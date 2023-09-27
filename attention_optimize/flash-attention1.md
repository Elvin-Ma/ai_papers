# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

# 摘要
Transformer在长序列上速度慢且内存不足，因为自我关注的时间和内存复杂性在序列长度上是二次的。近似注意力方法试图通过权衡模型质量以降低计算复杂度来解决这个问题，但通常无法实现挂钟加速(wall-clock speedup)。我们认为，一个缺失的原则是让注意力算法IO感知——考虑GPU内存级别之间的读写。我们提出了FlashAttention，这是一种IO感知的精确注意力算法，它使用平铺来减少GPU高带宽内存（HBM）和GPU片上SRAM之间的内存读/写次数。我们分析了FlashAttention的IO复杂性，表明它需要比标准关注更少的HBM访问，并且对于一系列SRAM大小来说是最佳的。我们还将FlashAttention扩展到block-sparse attention，从而产生比任何现有的近似注意力方法更快的近似注意力算法。FlashAttention训练transformer的速度比现有baseline更快：与MLPerf 1.1训练速度记录相比，BERT大型（序列长度512）的端到端时钟速度提高了15%，GPT-2（序列长度1K）的速度提高了3倍，远程竞技场(long-range arena序列长度1K-4K)的速度提升了2.4倍。FlashAttention和块稀疏FlashAttentention使Transformers中的上下文更长，产生了更高质量的模型（GPT-2的困惑度为0.7，长文档分类的提升度为6.4）和全新的能力：第一个在Path-X挑战（序列长度16K，准确率为61.4%）和Path-256（序列长度64K，准确度为63.1%）上实现优于偶然性能的transformer。
*注释："Wall-clock time"（挂钟时间）是指程序从开始执行到结束所经过的实际时间，通常以秒为单位。当对程序进行改进或优化时，如果程序的执行时间减少了，那么就可以说程序实现了"wall-clock speedup"。*
*SRAM（Static Random-Access Memory）是一种静态随机存取存储器，用于在计算机和其他电子设备中存储数据。它是一种易失性存储器，需要持续的电源供应来保持存储的数据。SRAM具有更快的访问速度和更低的访问延迟，因为SRAM使用了一种基于触发器的存储元件结构，而DRAM使用了电容器和电荷的存储机制。*
*DRAM（Dynamic Random-Access Memory）是一种动态随机存取存储器，用于在计算机和其他电子设备中存储数据。它是一种易失性存储器，需要持续的电源供应来保持存储的数据。*
*DRAM与另一种常见的存储器类型SRAM（Static Random-Access Memory）有所不同。与SRAM相比，DRAM具有较高的存储密度和较低的成本，但访问速度和延迟相对较高*
*计算机系统中的内存通常是指主存储器，它主要由DRAM（Dynamic Random-Access Memory）组成*

# 介绍(Introduction)
Transformer 模型[82] 已经成为自然语言处理和图像分类等应用中最常用的架构。Transformer 模型变得更大[5]、更深[83]，但要为它们提供更长的上下文仍然很困难[80]，因为它们核心的自注意力模块的时间和内存复杂度随序列长度呈二次增长。一个重要的问题是，将注意力机制变得更快速和内存更高效是否可以帮助 Transformer 模型解决长序列的运行时间和内存挑战。<br>
许多近似注意力方法旨在减少注意力计算和内存需求。这些方法包括稀疏近似[51, 74]、低秩近似[12, 50, 84]以及它们的组合[3, 9, 92]。尽管这些方法将计算需求降低为线性或接近线性，但其中许多方法并没有在与标准注意力相比的挂钟速度上显示出加速，并且没有得到广泛采用。一个主要原因是它们侧重于降低浮点运算量（与挂钟速度可能不相关），并且往往忽视了内存访问（IO）带来的开销。<br>
![figure1](./images/flash_attention1_figure1.jpg)
在本论文中，我们认为一个没被注意到的原则是使注意力算法具有IO感知性[1]，即仔细考虑对不同级别的快速和慢速存储器(memory)进行读写操作（例如，在快速GPU 芯片上SRAM和相对较慢的GPU高带宽存储器之间，如图1所示，左侧）。在现代GPU上，计算速度已经超过了内存速度[61, 62, 63]，而Transformer中的大多数操作都受到内存访问的瓶颈[43]。在读取和写入数据占据了运行时间的很大部分的内存受限操作中(IO密集型operator中)，IO感知算法至关重要，例如数据库连接[71]、图像处理[70]、数值线性代数[4]等等[40, 85]。然而，诸如PyTorch和Tensorflow等深度学习的常见Python接口并不允许对内存访问进行细粒度的控制。
