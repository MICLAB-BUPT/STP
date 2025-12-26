from typing import Tuple, List, Union, Callable, Optional
from types import MethodType

import torch
import torch.nn as nn
import torch.distributed as dist

from . import comm
from . import schedule

from megatron.core.parallel_state import get_tensor_model_parallel_rank, \
    get_pipeline_model_parallel_prev_rank, get_pipeline_model_parallel_next_rank, get_pipeline_model_parallel_group, get_pipeline_model_parallel_rank
from .utils import WeightGradStore
from megatron.core import parallel_state

class Trainer(nn.Module):
    def __init__(
        self,
        modules: Tuple[nn.Module, nn.Module],
        num_microbatches: int = 0,
        process_group: Optional[dist.ProcessGroup] = None,
        rank_mapping: Optional[List[int]] = None,
        forward_step = True,
        vl_model = False,
        offload_enabled: bool = False,
        steady_peak = False,
    ) -> None:
        super().__init__()

        self.tp_rank = get_tensor_model_parallel_rank()
        self.module = modules

        self.num_microbatches = num_microbatches
        self.overlapped_forward_backward = True
        self.group = get_pipeline_model_parallel_group()
        self.num_ranks = self.group.size()

        # rank_mapping: Map rank in process_group to actual pp rank.
        # rank_inverse_mapping: Map actual pp rank to rank in process_group.
        if rank_mapping is None:
            rank_mapping = list(range(self.num_ranks))
        rank_inverse_mapping = [None] * (self.num_ranks + 1)
        for i in range(self.num_ranks):
            rank_inverse_mapping[rank_mapping[i]] = i

        self.rank = get_pipeline_model_parallel_rank()
        self.prev_rank = get_pipeline_model_parallel_prev_rank()
        self.next_rank = get_pipeline_model_parallel_next_rank()

        self.is_first_rank = self.rank == 0
        self.is_last_rank = self.rank == self.num_ranks - 1
        
        # custom foward step defined in training code, in vpp it trains model and here we use it for loading data
        self.model_forward_func = forward_step 
        self.print = False
        self.vl_model = vl_model
        self.cmds = schedule.STPSchedule(self.rank, self.num_ranks, num_microbatches).generate(steady_peak)
        # if self.tp_rank == 0:
        #     print(self.rank, self.cmds, flush=True)
        self.offloaders = [module.config.qwen2vl.offload for module in self.module]
        
        self.offload_enabled = offload_enabled
        
        self.tensor_shapes = [m.comm_tensor_shape for m in modules]
        self.ite = 0
        self.first_f = 0

    def _reset_states(self) -> None:
        WeightGradStore.clear()
        for offload in self.offloaders:
            offload.clear()
            offload.reset()

        self.input_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], []) # hidden states
        self.input_dataloader_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], []) # attn mask, position ids, ...
        
        self.output_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.input_grad_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.output_grad_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.loss_chunks: List[torch.Tensor] = []
        self.criterion: Callable = []

        self.current_send_f_chunk_id: List[int] = [0, 0]
        self.current_send_b_chunk_id: List[int] = [0, 0]
        self.comm_ops: List[dist.P2POp] = []
        self.first_f = 0
    
    def _load_microbatch(self, phase: int, mb_idx: int, enable_zb=False):
        args, kwargs, criterion = self.model_forward_func()
        if self.is_first_rank and phase == 1:
            self.criterion.append(criterion)
        if self.is_first_rank and phase == 0:
            self.input_chunks[0].append(args[0])
        
        inputs = self.input_chunks[phase][mb_idx]
        if self.is_first_rank and phase == 0:
            self.input_chunks[phase][mb_idx] = None
        # set hidden states
        if not (self.is_first_rank and phase == 0):
            set_input_tensor = getattr(self.module[phase], "set_input_tensor")
            set_input_tensor(inputs)
        
        dataloader = dict(args=args, kwargs=kwargs)
        inputs_args = dataloader['args']
        inputs_kwargs = dataloader['kwargs']
        return inputs_args, inputs_kwargs

    def _after_forward_compute(self, phase: int, outputs):
        outputs = [outputs] if isinstance(outputs, torch.Tensor) else outputs
        is_last_stage = (self.is_first_rank and phase == 1)
        
        if is_last_stage:
            loss = self.criterion.pop(0)(*outputs) / self.num_microbatches
            self.loss_chunks.append(loss)

        if self.is_last_rank and phase == 0:
            self.input_chunks[1].append([output.detach().requires_grad_() for output in outputs])
        if not is_last_stage:
            self.output_chunks[phase].append(outputs)
            
    def _forward_compute_chunk(self, phase: int, mb_idx: int, enable_zb=False, offload=0) -> None:
        parallel_state.set_virtual_pipeline_model_parallel_rank(phase)
        
        inputs_args, inputs_kwargs = self._load_microbatch(phase, mb_idx, enable_zb)
        
        if self.tp_rank == 0 and self.print:
            print(f'[{self.rank}]: forward comp phase: {phase}, id {mb_idx}', flush=True)
        
        offload_handler = self.offloaders[phase]
        with offload_handler.run_context(idx=mb_idx, offload_level=offload):
            outputs = self.module[phase](*inputs_args, **inputs_kwargs)
        self._after_forward_compute(phase, outputs)
    
    def _load_backward_data(self, phase: int, mb_idx: int):
        is_last_stage = (self.is_first_rank and phase == 1)
        
        if is_last_stage:
            loss = self.loss_chunks[mb_idx]
            self.loss_chunks[mb_idx] = None
            outputs = [loss]
            output_grads = None
            torch.autograd.backward(outputs)
        else:
            outputs = self.output_chunks[phase][mb_idx]
            self.output_chunks[phase][mb_idx] = None
            output_grads = self.output_grad_chunks[phase][mb_idx]
            self.output_grad_chunks[phase][mb_idx] = None
            
            non_empty = [(t, g) for t, g in zip(outputs, output_grads) if g is not None]
            for output, output_grad in non_empty:
                output.grad = output_grad
        
        # optional, when the first stage is visual encoder without lm layer
        if self.vl_model and (self.module[phase].visual_process and not self.module[phase].lm_process):
            torch.autograd.backward(output, output.grad)
        
        return outputs
    
    def _after_backward_compute(self, phase: int, mb_idx):
        inputs = self.input_chunks[phase][mb_idx]
        self.input_chunks[phase][mb_idx] = None
        if self.is_first_rank and phase == 0:
            self.input_grad_chunks[0].append(None)
            return
        input_grads = [t.grad for t in inputs]

        if self.is_last_rank and phase == 1:
            self.output_grad_chunks[0].append(input_grads)
        else:
            self.input_grad_chunks[phase].append(input_grads)
            
    def _backward_compute_chunk(self, phase: int, mb_idx: int, enable_zb=False) -> None:
        parallel_state.set_virtual_pipeline_model_parallel_rank(phase)
        
        if self.tp_rank == 0 and self.print:
            print(f'[{self.rank}]: backward comp phase: {phase}, id {mb_idx}, enable {enable_zb}', flush=True)

        outputs = self._load_backward_data(phase, mb_idx)
        with WeightGradStore.func_enabled(enable_zb):
            if len(outputs) > 0:
                backward = getattr(self.module[phase], "backward")
                backward(outputs)
        self._after_backward_compute(phase, mb_idx)

    def _forward_weight_compute_chunk(self, phase: int, mb_idx, enable_zb=False, offload=0) -> None:
        parallel_state.set_virtual_pipeline_model_parallel_rank(phase)
        
        inputs_args, inputs_kwargs = self._load_microbatch(phase, mb_idx, enable_zb)
        if self.tp_rank == 0 and self.print:
            print(f'[{self.rank}]: forward weight comp phase: {phase}, id {mb_idx}', flush=True)
        
        offload_handler = self.offloaders[phase]
        with offload_handler.run_context(idx=mb_idx, offload_level=offload):
            outputs = self.module[phase](*inputs_args, **inputs_kwargs, overlap_weight=True)
        self._after_forward_compute(phase, outputs)
            
    def _forward_forward_compute_chunk(self, phase: int, mb_idx, enable_zb=False, offload=0) -> None:
        parallel_state.set_virtual_pipeline_model_parallel_rank(phase)
        
        self._forward_compute_chunk(phase, mb_idx, enable_zb, offload)
        self._forward_compute_chunk(phase, mb_idx+1, enable_zb, offload)
        
    def _forward_backward_compute_chunk(self, phase: int, mb_idx, enable_zb=False, offload=0) -> None:
        parallel_state.set_virtual_pipeline_model_parallel_rank(phase)
        
        assert isinstance(mb_idx, (tuple, list))        
        mb_idx, bw_idx = mb_idx
        inputs_args, inputs_kwargs = self._load_microbatch(phase, mb_idx, enable_zb)

        if self.tp_rank == 0 and self.print:
            print(f'[{self.rank}]: forward backward phase: {phase}, id {mb_idx}, {bw_idx}, enable {enable_zb}', flush=True)
            
        self._load_backward_data(phase, bw_idx)
                   
        # forward & backward
        offload_handler = self.offloaders[phase]
        offload_context = offload_handler.run_context(idx=mb_idx, offload_level=offload)
        with WeightGradStore.func_enabled(enable_zb), offload_context:
            outputs = self.module[phase](*inputs_args, **inputs_kwargs, overlap_backward=True)

        self._after_forward_compute(phase, outputs)
        self._after_backward_compute(phase, bw_idx)

    def _forward_chunk(self, phase: int, mb_idx, enable_zb=False, offload=0, recv=True, send=True) -> None:
        if recv:
            self._recv_forward(phase)

        self._commit_and_wait_comm()
        self._forward_compute_chunk(phase, mb_idx[0], enable_zb, offload)

        self.first_f += 1
        
        if send:
            self._send_forward(phase)

    def _backward_chunk(self, phase: int, mb_idx, enable_zb=False, offload=0, recv=True, send=True) -> None:
        if recv:
            self._recv_backward(phase)
        self._commit_and_wait_comm()
        self._backward_compute_chunk(phase, mb_idx[0], enable_zb)
        if send:
            self._send_backward(phase)

    def _forward_forward_chunk(self, phase: int, mb_idx, enable_zb=False, offload=0, recv:Optional[int]=True, send=True) -> None:
        for i in range(recv):
            self._recv_forward(phase)
        self._commit_and_wait_comm()
        self._forward_forward_compute_chunk(phase, mb_idx[0], enable_zb, offload)
        if send:
            self._send_forward(phase)
            self._send_forward(phase)
    
    def _forward_weight_chunk(self, phase: int, mb_idx, enable_zb=False, offload=0, recv=True, send=True) -> None:
        if recv:
            self._recv_forward(phase)
        self._commit_and_wait_comm()
        self._forward_weight_compute_chunk(phase, mb_idx[0], enable_zb, offload)
        if send:
            self._send_forward(phase)
         
    def _forward_backward_chunk(self, phase: int, mb_idx, enable_zb=False, offload=0, recv=True, send=True) -> None:
        if recv:
            self._recv_forward(phase)

        self._recv_backward(phase)
        self._commit_and_wait_comm()
        self._forward_backward_compute_chunk(phase, mb_idx, enable_zb, offload)

        self._send_forward(phase)
        self._send_backward(phase)
        
    def _weight_chunk(self, *args, **kwargs) -> None:
        self._commit_and_wait_comm()
        torch.cuda.nvtx.range_push(f"weight_bw")
        WeightGradStore.pop()
        torch.cuda.nvtx.range_pop()
    
    def _acitvation_offload(self, phase, mb_idx, offload=0, *args, **kwargs):
        return
        # if self.offload_enabled and offload > 0:
        #     self.offloaders[phase].offload(mb_idx)
        
    def _activation_reload(self, phase, mb_idx, *args, **kwargs):
        if self.offload_enabled:
            self.offloaders[phase].reload(mb_idx)

    def _recv_forward(self, phase: int, *args, **kwargs) -> None:
        if (self.is_first_rank and phase == 0) or (self.is_last_rank and phase == 1):
            return

        peer_rank = self.prev_rank if phase == 0 else self.next_rank
        tensor_shape = self.tensor_shapes[phase]
        if isinstance(tensor_shape, dict):
            tensor_shape = tensor_shape[peer_rank]
        tensors = comm.append_irecv(self.comm_ops, peer_rank, tensor_shape, self.group)
        
        self.input_chunks[phase].append(tensors)

    def _send_forward(self, phase: int, *args, **kwargs) -> None:
        if (self.is_first_rank and phase == 1) or (self.is_last_rank and phase == 0):
            return

        chunk_id = self.current_send_f_chunk_id[phase]
        self.current_send_f_chunk_id[phase] += 1
        tensors = self.output_chunks[phase][chunk_id]
        comm.append_isend(self.comm_ops, tensors, self.next_rank if phase == 0 else self.prev_rank, self.group)

    def _recv_backward(self, phase: int, *args, **kwargs) -> None:
        if (self.is_first_rank and phase == 1) or (self.is_last_rank and phase == 0):
            return
        peer_rank = self.next_rank if phase == 0 else self.prev_rank
        tensor_shape = self.tensor_shapes[phase]
        if isinstance(tensor_shape, dict):
            tensor_shape = tensor_shape[peer_rank]
        tensors = comm.append_irecv(self.comm_ops, peer_rank, tensor_shape, self.group)
        self.output_grad_chunks[phase].append(tensors)

    def _send_backward(self, phase: int, *args, **kwargs) -> None:
        if (self.is_first_rank and phase == 0) or (self.is_last_rank and phase == 1):
            return

        chunk_id = self.current_send_b_chunk_id[phase]
        self.current_send_b_chunk_id[phase] += 1
        tensors = self.input_grad_chunks[phase][chunk_id]
        self.input_grad_chunks[phase][chunk_id] = None
        comm.append_isend(self.comm_ops, tensors, self.prev_rank if phase == 0 else self.next_rank, self.group)

    def _commit_and_wait_comm(self) -> None:
        if not self.comm_ops:
            return
        if self.print and self.tp_rank == 0:
            print(f'[{self.rank}]: commit and wait comm {self.comm_ops}', flush=True)
        reqs = dist.batch_isend_irecv(self.comm_ops)
        for req in reqs:
            req.wait()
        self.comm_ops = []

    def step(self, ):
        assert comm.TENSOR_SHAPES is not None and comm.TENSOR_DTYPE is not None, \
            "You need to call set_p2p_tensor_shapes and set_p2p_tensor_dtype before executing a step."
        for module in self.module:
            module.train()

        self._reset_states()
        memory_recoder_key = []
        memory_recoder = []
        for cmd in self.cmds:
            if self.tp_rank == 0 and self.rank == 1 and self.print:
                print(self.rank, cmd, flush=True)
            if self.ite == 0:
                if type(cmd) in [schedule.Forward, schedule.Backward, schedule.ForwardAndWeight, schedule.ForwardAndForward, schedule.ForwardAndBackward]:
                    instruction = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                    memory_recoder.append(torch.cuda.max_memory_allocated() / 1000**3)
                    instruction(**cmd.kwargs)
                    memory_recoder.append(torch.cuda.max_memory_allocated() / 1000**3)
                    memory_recoder_key.append((cmd, memory_recoder[-1]))
                    torch.cuda.reset_peak_memory_stats()
                else:
                    instruction = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                    instruction(**cmd.kwargs)
            else:
                instruction = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                instruction(**cmd.kwargs)
            
        assert WeightGradStore.funcs_queue.empty()

        self._commit_and_wait_comm()
        if parallel_state.get_tensor_model_parallel_rank() == 0 and self.ite == 0:
            for key in memory_recoder_key:
                print(f"{parallel_state.get_pipeline_model_parallel_rank()}, {key}", flush=True)

        store_result = []
        for loss in self.loss_chunks:
            store_result.append({'lm loss': loss.clone().detach()})
        
        for module in self.module:
            module.clear_saved_tensor()
            module.config.qwen2vl.offload.clear()
        self.ite += 1
        return store_result
    
    _INSTRUCTION_MAP = {
        schedule.Forward: _forward_chunk,
        schedule.Backward: _backward_chunk,
        schedule.ForwardAndWeight: _forward_weight_chunk,
        schedule.ForwardAndForward: _forward_forward_chunk,
        schedule.ForwardAndBackward: _forward_backward_chunk,
        schedule.RecvForward: _recv_forward,
        schedule.SendForward: _send_forward,
        schedule.Weight: _weight_chunk,
        schedule.ActOffload: _acitvation_offload,
        schedule.ActReload: _activation_reload,
    }