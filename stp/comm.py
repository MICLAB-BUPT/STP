from typing import List, Tuple

import torch
import torch.distributed as dist


TENSOR_SHAPES: List[Tuple[int]] = None
TENSOR_DTYPE: torch.dtype = None
TENSOR_ONE: torch.Tensor = None
VISUAL_TENSOR_ONE: torch.Tensor = None

def set_p2p_tensor_shapes(shapes: List[Tuple[int]]):
    global TENSOR_SHAPES
    TENSOR_SHAPES = shapes


def set_p2p_tensor_dtype(dtype: torch.dtype):
    global TENSOR_DTYPE
    TENSOR_DTYPE = dtype

def create_one_tensor():
    global TENSOR_ONE
    TENSOR_ONE = torch.ones(TENSOR_SHAPES[0], dtype=TENSOR_DTYPE, device="cuda", requires_grad=False)
    
def get_one_tensor():
    if TENSOR_ONE is None:
        create_one_tensor()
    return TENSOR_ONE


def create_visual_one_tensor(vit_len=2976, dim=1280):
    global VISUAL_TENSOR_ONE
    VISUAL_TENSOR_ONE = torch.ones((vit_len, dim), dtype=TENSOR_DTYPE, device="cuda", requires_grad=False)
    

def get_visual_one_tensor():
    if VISUAL_TENSOR_ONE is None:
        create_visual_one_tensor()
    return VISUAL_TENSOR_ONE

def build_from_tensor_shapes(tensor_shapes):
    return [torch.empty(s, dtype=TENSOR_DTYPE, device="cuda", requires_grad=True) for s in tensor_shapes]


def append_irecv(ops: List[dist.P2POp], src: int, tensor_shape,  group: dist.ProcessGroup) -> List[torch.Tensor]:
    if not isinstance(tensor_shape, list):
        tensor_shape = [tensor_shape]
    tensors = build_from_tensor_shapes(tensor_shape)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.irecv, tensor, src))
    return tensors


def append_isend(ops: List[dist.P2POp], tensors: List[torch.Tensor], dst: int, group: dist.ProcessGroup) -> None:
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.isend, tensor, dst))
