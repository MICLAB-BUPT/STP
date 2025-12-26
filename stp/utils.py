import queue
from typing import List, Callable
from contextlib import contextmanager
import functools
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu

class WeightGradStore:

    enabled: bool = False
    cache: List[Callable] = []
    args_cache: List[List] = []
    funcs_queue = queue.Queue()
    args_queue = []
    weight_per_layer: int = 0
    
    D2H_stream = torch.cuda.Stream()
    H2D_stream = torch.cuda.Stream()

    @classmethod
    def put(cls, func: Callable, *args) -> None:
        cls.cache.append(func)
        cls.args_cache.append(args)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)
        cls.args_queue.append(cls.args_cache)
        cls.cache = []
        cls.args_cache = []

    @classmethod
    def pop(cls, run=True) -> None:
        # assert not cls.funcs_queue.empty(), "Pop empty queue."
        if cls.funcs_queue.empty():
            return []
        funcs = cls.funcs_queue.get()
        args = cls.args_queue.pop(0)
        assert len(funcs) == len(args), "funcs and args mismatch."
        
        if not run:
            return_funcs = []
            for func, arg in zip(funcs, args):
                return_funcs.append(functools.partial(func, *arg))
            return return_funcs

        for func, arg in zip(funcs, args):
            func(*arg)
        
    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.funcs_queue = queue.Queue()
        cls.args_queue = []
        cls.args_cache = []
    
    @classmethod
    @contextmanager
    def func_enabled(cls, enabled):
        try:
            cls.enabled = enabled
            yield
        finally:
            cls.enabled = False
            if enabled:
                cls.flush()

class ForwardHandler:

    cache: List[Callable] = []
    funcs_queue = queue.Queue()
    enabled = False

    @classmethod
    def put(cls, func: Callable) -> None:
        cls.funcs_queue.put(func)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)

    @classmethod
    def pop(cls,) -> None:
        if mpu.get_tensor_model_parallel_world_size() == 1:
            return
        
        func = cls.funcs_queue.get()
        func.wait()

    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.funcs_queue = queue.Queue()

class BackwardHandler:

    cache: List[Callable] = []
    funcs_queue = queue.Queue()
    enabled = False

    @classmethod
    def put(cls, func: Callable) -> None:
        cls.funcs_queue.put(func)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)

    @classmethod
    def pop(cls,) -> None:
        if mpu.get_tensor_model_parallel_world_size() == 1:
            return
        func = cls.funcs_queue.get()
        func.wait()

    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.funcs_queue = queue.Queue()
        

