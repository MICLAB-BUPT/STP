import torch
import queue
from functools import partial
from collections import defaultdict
from contextlib import contextmanager

OFFLOAD_ENABLED = False
OFFLOAD_ACT_LEVEL = 2

def make_tensor_offload_enabled(tensor):
    if not OFFLOAD_ENABLED:
        return
    if isinstance(tensor, torch.Tensor):
        tensor.act_offload_enabled = True
        return
    elif isinstance(tensor, (list, tuple)):
        for t in tensor:
            make_tensor_offload_enabled(t)
    else:
        raise TypeError(f"{type(tensor)} is not supported")
    
def set_global_offload_enabled(enabled: bool):
    global OFFLOAD_ENABLED
    OFFLOAD_ENABLED = enabled

def set_offload_level(level: int):
    global OFFLOAD_ACT_LEVEL
    OFFLOAD_ACT_LEVEL = level

def get_offload_level():
    return OFFLOAD_ACT_LEVEL
    
class saved_tensors_hooks:
    def __init__(
        self,
        act_offload: 'ActivationOffload'
    ) -> None:
        if act_offload is not None:
            self.pack_hook = act_offload.after_forward_hook
            self.unpack_hook = act_offload.before_backward_hook
            self.offload_enabled = act_offload.enabled
        else:
            self.offload_enabled = False

    def __enter__(self) -> None:
        if self.offload_enabled and OFFLOAD_ENABLED:
            torch._C._autograd._push_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook
        )

    def __exit__(self, *args: object) -> None:
        if self.offload_enabled and OFFLOAD_ENABLED:
            torch._C._autograd._pop_saved_tensors_default_hooks()
            

# every model chunk owns this cls
class ActivationOffload:
    def __init__(self, ):
        self.D2H_stream = torch.cuda.Stream()
        self.H2D_stream = torch.cuda.Stream()
        self.offloading_gpu_tensors = defaultdict(list) # {batch id: [tensor, tensor, tensor]}
        self.saved_other_tensors = defaultdict(list)
        self.offload_tensor_counter = 0
        self.mb_id = 0
        self.enabled = False 
        self.offload_queue = queue.Queue()
    
    @contextmanager
    def run_context(self, idx=0, offload_level=0):
        try:
            set_offload_level(offload_level)
            enabled = OFFLOAD_ENABLED and offload_level > 0
            self.enabled = enabled
            if enabled:
                self.set_md_id(idx)
            yield
        finally:
            self.enabled = False
    
    def set_md_id(self, id):
        self.mb_id = id
        self.offload_tensor_counter = 0
        
    def clear(self,):
        self.offloading_gpu_tensors.clear()
        self.saved_other_tensors.clear()
    
    def reset(self,):
        self.offload_tensor_counter = 0
        self.mb_id = 0
    
    def get_tensor_store(self, key):
        assert key in ['gpu', 'cpu']
        if key == 'cpu':
            return self.offloading_gpu_tensors
        else:
            return self.saved_other_tensors        
    
    def after_forward_hook(self, gpu_tensor: torch.Tensor):
        device = gpu_tensor.device
        self.D2H_stream.wait_stream(torch.cuda.current_stream())
        if getattr(gpu_tensor, 'act_offload_enabled', False):
            with torch.cuda.stream(self.D2H_stream):
                cpu_backup = torch.empty(
                    gpu_tensor.size(),
                    dtype=gpu_tensor.dtype,
                    layout=gpu_tensor.layout,
                    device="cpu",
                    pin_memory=True,
                )
                cpu_backup.copy_(gpu_tensor, non_blocking=True)
            self.offloading_gpu_tensors[self.mb_id].append((cpu_backup, device))
            tensor_id = self.offload_tensor_counter
            self.offload_tensor_counter += 1
            store_dict = 'cpu'
        else:
            tensor_id = len(self.saved_other_tensors[self.mb_id])
            self.saved_other_tensors[self.mb_id].append((gpu_tensor, device))
            store_dict = 'gpu'
            
        return (store_dict, self.mb_id, tensor_id)
    
    def before_backward_hook(self, tensor_info):
        store_dict, mb_id, tensor_id = tensor_info
        saved_tensor = self.get_tensor_store(store_dict)
        tensor, device = saved_tensor[mb_id][tensor_id]
        saved_tensor[mb_id][tensor_id] = None
        assert tensor.device == device
        return tensor
    
    def offload(self, mb_id,):
        return
    
    def get_offload(self):
        return
    
    def offload_exec(self, start, end, pin_memory=True):
        gpu_tensors = self.offloading_gpu_tensors[self.mb_id]
        self.D2H_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.D2H_stream):
            for idx in range(start, end):
                gpu_tensor, deivce = gpu_tensors[idx]
                cpu_backup = torch.empty_like(
                    gpu_tensor,
                    device="cpu",
                    pin_memory=True,
                )
                gpu_tensors[idx] = (cpu_backup, deivce)
                cpu_backup.copy_(gpu_tensor, non_blocking=pin_memory)
            
    def reload(self, mb_id):
        if not OFFLOAD_ENABLED:
            return
        
        tensors = self.offloading_gpu_tensors.pop(mb_id, None)
        if tensors is None:
            return
        tmp = []
        self.offloading_gpu_tensors[mb_id] = tmp
        self.H2D_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.H2D_stream):
            for tensor, device in tensors:
                if tensor.device == device:
                    tmp.append((tensor, device))
                else:
                    tmp.append((tensor.to(device=device, non_blocking=True), device))
        
            