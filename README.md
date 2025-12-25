# Synergistic Tensor and Pipeline Parallelism


> [Synergistic Tensor and Pipeline Parallelism](https://arxiv.org/abs/2510.27257)
> Authors: Mengshi Qi, Jiaxuan Peng, Jie Zhang, Juan Zhu, Yong Li, Huadong Ma



## Schedule
### Motivation
<img src="figs/overlap_comm.png" width="80%"/>

The forward and backward computations (e.g., Attention and MLP) are interleaved for overlapping computation and All-Reduce communication in tensor parallelism, enabling efficient utilization of GPU resources and reducing idle time.

### Comparison of existing schedules
<img src="figs/schedules.png" width="100%"/>

| Schedule     | PP Bubble                              | TP Bubble           | Peak Activation Memory     |
|--------------|----------------------------------------|---------------------|----------------------|
| 1F1B-I       | $(p-1)(T_F + T_{AR} + T_B + T_W)$      | $2mT_{AR}$          | $(3p - 2) M_a$       |
| ZB-V         | $(p-1)(T_F + 2T_{AR} + T_B - 2T_W)$    | $4mT_{AR}$          | $2p M_a$             |
| Ours         | $(p-1)(T_F + T_{AR} + T_B - T_W)$      | $(2p + 1)T_{AR}$    | $3p M_a$             |

| Symbol      | Description |
|-------------|-------------|
| $p$         | Number of pipeline parallel (PP) stages |
| $m$         | Number of microbatches ($p \ll m$) |
| $M_a$       | Activation memory per microbatch per model chunk |
| $T_F$       | Forward computation time per chunk |
| $T_B$       | Activation gradient computation time per chunk |
| $T_W$       | Weight gradient computation time per chunk |
| $T_{AR}$    | TP communication time per chunk |


We compare our synergistic schedule against two pipeline parallelism strategies: 1F1B-I and Zero Bubble V (ZB-V). While 1F1B-I suffers from large pipeline bubbles and ZB-V incurs excessive tensor parallel communication overhead due to its decoupled backward pass, our approach jointly reduces both PP and TP bubbles through fine-grained interleaving of computation and communication.

### Offloading Schedule
<img src="figs/offload_schedule.jpg" width="80%"/>

We introduce a CPU offloading strategy that overlaps tensor transfers with on-device computation. As illustrated in the schedule, activations are offloaded immediately after computation, reducing GPU memory pressure without stalling the training pipeline.
## Citation
```
@misc{qi2025synergistictensorpipelineparallelism,
      title={Synergistic Tensor and Pipeline Parallelism}, 
      author={Mengshi Qi and Jiaxuan Peng and Jie Zhang and Juan Zhu and Yong Li and Huadong Ma},
      year={2025},
      eprint={2510.27257},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2510.27257}, 
}
```