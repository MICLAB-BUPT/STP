
from abc import abstractmethod


class ScheduleBase:
    def __init__(self, rank, world_size, num_microbatches):
        self.rank = rank
        self.world_size = world_size
        self.mbs = num_microbatches
    
    def append_cmds(self, instruction, phase=0, counter=None, enable_zb=False, recv=True, send=True, offload=1, mb_idx=None):
        if counter is None:
            self.exec_instructions.append(instruction(phase=phase, mb_idx=mb_idx, enable_zb=enable_zb, recv=recv, send=send, offload=offload))
            return
        
        mb_idx = []
        for c in counter:
            mb_idx.append(c[phase])
        cmd = instruction(phase=phase, mb_idx=tuple(mb_idx), enable_zb=enable_zb, offload=offload, recv=recv, send=send,)
        self.exec_instructions.append(cmd)        
        for c in counter:
            c[phase] += 1
            if isinstance(cmd, ForwardAndForward):
                c[phase] += 1
    @abstractmethod
    def generate(self, ):
        pass


class STPSchedule(ScheduleBase):         
    def generate(self,):
        self.exec_instructions = []
        rank = self.rank
        world_size = self.world_size
        mbs = self.mbs
        assert mbs > 0 and mbs >= world_size * 2, f"{mbs=}, {world_size=}"

        fw_idx = [0, 0]
        bw_idx = [0, 0]
        
        # Step 1: nF0
        step_1 = (world_size - rank - 1) * 2
        for i in range(step_1):
            # offload = 0 if rank == 0 else 1
            offload = 1 # set offload level manually
            self.append_cmds(Forward, phase=0, counter=[fw_idx], offload=offload)
            self.append_cmds(ActOffload, phase=0, mb_idx=fw_idx[0]-1, offload=offload)


        # Step 2: nF0F1
        step_2 = rank + 1
        self.append_cmds(RecvForward, phase=0)
        for i in range(step_2):
            send = True
            self.append_cmds(Forward, phase=0, counter=[fw_idx], recv=False, send=False, offload=1)
            self.append_cmds(ActOffload, phase=0, mb_idx=fw_idx[0]-1, offload=1)
            self.append_cmds(RecvForward, phase=0)
            # last F1 don't send activation immediately
            if i == step_2 - 1:
                send = False
            offload = 0
            self.append_cmds(Forward, phase=1, counter=[fw_idx], send=send, offload=offload)
            self.append_cmds(SendForward, phase=0)
            self.append_cmds(ActOffload, phase=1, mb_idx=fw_idx[1]-1, offload=offload)

        step_3 = rank + 1
        single_F = step_3 % 2 == 1
        overlapped_F = step_3 // 2
        if single_F:
            self.append_cmds(Forward, phase=0, counter=[fw_idx], recv=False, offload=1)
            self.append_cmds(ActOffload, phase=0, mb_idx=fw_idx[0]-1,)
        for i in range(overlapped_F):
            recv = 2
            if i == 0 and not single_F:
                recv = 1
            if i == overlapped_F - 1:
                self.append_cmds(SendForward, phase=1)
                # self.append_cmds(ActReload, phase=1, mb_idx=bw_idx[1],)
                
            self.append_cmds(ForwardAndForward, phase=0, counter=[fw_idx], recv=recv, offload=1)
            self.append_cmds(ActOffload, phase=0, mb_idx=fw_idx[0]-2, offload=1)
            self.append_cmds(ActOffload, phase=0, mb_idx=fw_idx[0]-1, offload=1)
        
        # one F1B1, one F0W1
        step_4 = world_size - rank - 1
        for i in range(step_4):
            offload = 0
            self.append_cmds(ForwardAndBackward, phase=1, counter=[fw_idx, bw_idx], enable_zb=True, offload=offload)
            # self.append_cmds(ActReload, phase=1, mb_idx=bw_idx[1],)
            offload = 1 if rank == 0 else 1
            self.append_cmds(ForwardAndWeight, phase=0, counter=[fw_idx], offload=offload)
            self.append_cmds(ActOffload, phase=0, mb_idx=fw_idx[0]-1, offload=offload)
        
        # Step 5: nB1F1 or B0F0
        step_5 = 2 * (mbs - 3 * world_size + 1) + 1 + rank * 2
        full_steps = 2 * (mbs - 3 * world_size + 1) + 1 + rank
        for i in range(step_5):
            enable_zb = True if i >= full_steps else False
            phase = 1 if i % 2 == 0 else 0
            reload_phase = 1 if phase == 0 else 0
            if reload_phase == 0:
                self.append_cmds(ActReload, phase=reload_phase, mb_idx=bw_idx[reload_phase],)
            offload = 0 if phase == 1 else 1
            self.append_cmds(ForwardAndBackward, phase=phase, counter=[fw_idx, bw_idx], enable_zb=enable_zb, offload=offload)
            self.append_cmds(ActOffload, phase=phase, mb_idx=fw_idx[phase]-1, offload=offload)

        # Step 6: n B0&F1B1
        # finish overlapped comm and computation, backward
        step_6 = mbs - step_2 - step_4 - step_5//2 - 1
        for i in range(step_6):
            self.append_cmds(ActReload, phase=1, mb_idx=bw_idx[1],)
            self.append_cmds(Backward, phase=0, counter=[bw_idx])
            self.append_cmds(ActReload, phase=0, mb_idx=bw_idx[0],)
            offload = 0 if rank == 0 else 1
            self.append_cmds(ForwardAndBackward, phase=1, counter=[fw_idx, bw_idx], enable_zb=True, offload=offload)
            self.append_cmds(ActOffload, phase=1, mb_idx=fw_idx[1]-1, offload=offload)

        # # Step 7: n B0&B1
        step_7 = 1 + rank
        for i in range(step_7):
            self.append_cmds(ActReload, phase=1, mb_idx=bw_idx[1],)
            self.append_cmds(Backward, phase=0, counter=[bw_idx])
            self.append_cmds(ActReload, phase=0, mb_idx=bw_idx[0],)
            self.append_cmds(Backward, phase=1, counter=[bw_idx])

        # # Step 8: nB0&WW
        step_8 = 1 + (world_size - rank - 1) * 2
        for i in range(step_8):
            if i % 2 == 0:
                self.append_cmds(Backward, phase=0, counter=[bw_idx])
            else:
                self.append_cmds(ActReload, phase=0, mb_idx=bw_idx[0],)
                self.append_cmds(Weight)
                self.append_cmds(Weight)

        # # Step 9: nW
        step_9 = rank * 2
        for i in range(step_9):
            self.append_cmds(Weight)
        
        return self.exec_instructions


def call_to_str(base, *args, **kwargs):
    name = f'{base}('
    if args:
        name += ', '.join(repr(arg) for arg in args)
        if kwargs:
            name += ', '
    if kwargs:
        name += ', '.join(f'{key}={repr(arg)}' for key, arg in kwargs.items())
    name += ')'
    return name

class PipeInstruction:
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        return call_to_str(self.name, **self.kwargs)


# Compute
class Forward(PipeInstruction):
    pass

class Backward(PipeInstruction):
    pass

class Weight(PipeInstruction):
    pass

class ForwardAndForward(PipeInstruction):
    pass

class ForwardAndWeight(PipeInstruction):
    pass

class ForwardAndBackward(PipeInstruction):
    pass

class RecvForward(PipeInstruction):
    pass

class SendForward(PipeInstruction):
    pass

class ActOffload(PipeInstruction):
    pass

class ActReload(PipeInstruction):
    pass