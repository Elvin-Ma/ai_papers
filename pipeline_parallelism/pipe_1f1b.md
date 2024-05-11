# PipeDream: Fast and Efficient Pipeline Parallel DNN Training
- [paper 论文](https://arxiv.org/pdf/1806.03377)

# 摘要
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PipeDream是一种用于GPU的深度神经网络（DNN）训练系统，通过在多台机器上进行流水线执行来并行计算。其流水线并行计算模型**避免了**数据并行训练中面临的由于大型模型和/或有限的网络带宽引起的**高通信与计算比例而导致的减速问题**。**相对于数据并行训练，PipeDream相对于大型DNNs减少了高达95%的通信，并允许完全重叠的通信和计算**。PipeDream通过系统地**将DNN层分配给所有可用的GPU**，以平衡工作并最小化通信，为向后传递的正确性版本化模型参数，并以循环方式安排不同输入的正向和反向传递，以优化“达到目标准确性所需的时间”。在两个不同的集群上对五个不同的DNN进行的实验表明，与数据并行训练相比，**PipeDream在时间达到准确性方面最多快5倍**。<br>


