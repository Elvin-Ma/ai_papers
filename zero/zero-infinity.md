# zero-infinity：打破极端规模(scale)深度学习模型的内存墙

# 摘要
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在过去的三年里，最大的密集(dense)深度学习模型的规模增长了1000倍以上，达到了数千亿个参数，而GPU内存只增长了5倍（从16GB到80GB）。因此，模型规模的增长主要是通过系统创新来支持的，这些创新使得大型模型可以适应多个GPU的总体内存。然而，我们正在接近GPU内存的极限。为了训练一个**万亿参数的模型，需要800个NVIDIA V100 GPU**，而这样的集群对大多数数据科学家来说根本无法实现。此外，以这样的规模训练模型需要复杂的并行技术组合，给数据科学家增加了重构模型的巨大负担。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在本论文中，我们提出了一种名为ZeRO-Infinity的新型异构系统技术，它利用GPU、CPU和NVMe内存，在有限资源下实现了**前所未有**的模型规模扩展，而无需对模型代码进行重构。同时，它实现了出色的训练吞吐量和可扩展性，不受限于有限的CPU或NVMe带宽。ZeRO-Infinity可以在当前一代GPU集群上容纳数**万亿甚至数百万亿个参数**的模型进行训练。它可以用于在单个NVIDIA DGX-2节点上微调万亿参数模型，使大型模型更易于使用。在训练吞吐量和可扩展性方面，它在512个NVIDIA V100 GPU上可以维持超过25 petaflops的性能（达到峰值的40%），同时还展示了超线性的可扩展性。ZeRO-Infinity的开源实现可通过DeepSpeed获得。
*注释：NVMe代表非易失性内存扩展（Non-Volatile Memory Express），是一种高性能、低延迟的存储接口协议。NVMe利用并行性和高带宽特性，提供了更高的I/O性能和更低的延迟。这使得NVMe成为处理大规模数据和高性能计算应用的理想选择，尤其在需要快速数据读写和响应时间的场景中。* <br>
*注释：DeepSpeed（https://www.deepspeed.ai/）是一个旨在使分布式训练变得简单、高效和有效的深度学习优化库。DeepSpeed已经被深度学习社区广泛采用。*

# 1 背景(EXTENDED)介绍
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;近年来，深度学习在取得巨大(tremendous)进展，使其成为我们生活中不可或缺的一部分，从为搜索引擎提供动力到智能家居虚拟助手。这些进展的核心在于模型规模的增加[1-3]，而多项研究表明这一趋势将会持续下去[4, 5]。因此，人们已经大量投资于训练庞大的模型。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在过去的三年中，深度学习中最大的训练密集模型的规模增长了1000倍，从一亿个参数（ELMo [6]）增长到超过一千亿个参数（GPT-3 [4]）。相比之下，单个GPU的内存仅增加了5倍（从16GB到80GB）。因此，模型规模的增长主要通过系统技术的进步来实现大规模深度学习模型的训练，其中包括模型并行化 [7]、流水线并行化 [8-10] 和 ZeRO [11, 12] 等并行技术，正在为训练更大、更强大的模型铺平道路.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当前大模型训练技术的sota技术是三维并行化(3D parallelism [13, 14])，它将模型（张量切片）并行化、流水线并行化和数据并行化相结合，有效地将深度学习训练扩展到数万亿个参数，并在数百或数千个GPU上进行。例如，DeepSpeed实现的三维并行化可以利用集群的GPU内存，使得在**800个NVIDIA V100 GPU上可以扩展到超过一万亿个参数的规模** [15]。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;尽管三维并行化(3D parallelism)在大型模型训练方面具有很大的能力，但我们现在正面临GPU内存墙的挑战 [16]。**集群的GPU内存简单地不足以支持模型规模的增长**。即使使用最新的NVIDIA A100 GPU，其具有80GB的内存，为了训练一个万亿参数的模型，三维并行化需要320个GPU才能容纳，而要扩展到未来百万亿参数级别的模型，则需要超过6K个GPU，即使我们假设未来几年GPU内存增加5倍。我们无法再依靠GPU内存作为瓶颈来维持模型规模的持续增长。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;GPU 内存墙也限制了数据科学家甚至无法访问当今的大型模型，特别是用于微调。大型模型首先在大量的通用数据上进行预训练，通过微调，同一模型可以针对各种应用进行专门优化。虽然对一个拥有数千亿个参数的模型进行预训练可能需要数百万个GPU计算小时，但进行微调的成本要低得多，只需要较少的GPU计算小时，并且可以在单个计算节点(不是单个gpu)上完成，使用少量的GPU。虽然许多企业和用户可以获得此类计算资源，但不幸的是，它们受限于计算节点上可用的内存，这进而限制了可以进行微调的模型规模。这使得**大型模型的微调对于大多数没有大规模GPU集群资源的研究人员和公司来说是无法实现的**。例如，即使单个DGX-2节点（16个GPU）具备足够的计算能力在合理的时间内对GPT-3进行微调，但要将模型适应于训练，需要超过8个DGX-2节点（128个GPU）并使用三维并行化。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;除了GPU内存墙外，用于训练大规模模型的最新技术在可用性和灵活性方面也存在限制。如上所述，三维并行化需要以复杂的方式将数据并行化、模型并行化和流水线并行化相结合，才能达到数千亿或数万亿个参数的规模。虽然这样的系统可能非常高效，但它要求数据科学家进行重大的模型代码重构，将单个GPU运算符替换为张量切片版本，并将模型分割成负载平衡的流水线阶段。这也使得三维并行化在支持的模型类型方面缺乏灵活性。具有复杂依赖关系的模型不能轻易转换为负载平衡的流水线结构。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;给定大型模型训练的现状，我们提出了三个问题：<br>
- 展望未来，我们如何支持模型规模的下一次1000倍增长，从拥有1750亿参数的模型（如GPT-3）到拥有数万亿参数的模型？<br>
- 我们如何让当今的大型模型对更多没有数百个GPU资源的数据科学家可用？<br>
- 我们能否通过消除模型重构和多种形式的并行化(parallelism)的需求，使大型模型训练更加简单？<br>
在本论文中，我们从三维并行化迈出了一大步，提出了ZeRO-Infinity，这是一种新颖的系统，能够解决大型模型训练的所有上述挑战。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**前所未有的模型规模**：ZeRO-Infinity通过新颖的异构内存访问技术——**无限卸载引擎（infinity offload engine）**，扩展了ZeRO技术家族[11, 12]。这使得ZeRO-Infinity能够利用CPU和NVMe内存同时支持有限的GPU资源上的大规模模型。此外，ZeRO-Infinity还引入了一种称为**内存中心平铺（memory-centric tiling）的新颖GPU内存优化技术**，以支持那些即使逐层单独加载也无法适应GPU内存的**极大尺寸的单个层**。通过无限卸载引擎和内存中心平铺技术，ZeRO-Infinity不仅支持模型规模的下一次1000倍增长，而且还使得拥有有限GPU资源的数据科学家能够使用大型模型。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;出色的训练效率：ZeRO-Infinity引入了一种新颖的数据分区策略，利用(leveraging)所有设备的聚合(aggregate)内存带宽，我们称之为**带宽中心分区(bandwidth-centric partitioning)**，并将其与**强大的通信重叠设计**以及在**无限卸载引擎**中进行**高性能NVMe访问**的优化相结合。尽管将数据卸载到CPU或NVMe，但ZeRO-Infinity提供了出色的训练效率，不受它们有限带宽的限制。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**易于使用**：有了ZeRO-Infinity，数据科学家不再需要根据三维并行化等多种并行化形式来调整他们的模型。这是由上面讨论的ZeRO-Infinity中的**内存中心平铺技术(memory-centric tiling)实现的**，该技术旨在减少单个大型层(layer)所需的GPU内存，否则需要模型并行化（张量切片）来适应GPU内存。此外，ZeRO-Infinity通过一种启发式易于实现的方法，自动化了训练任意模型架构所需的所有通信和数据分区，从而消除了手动模型代码重构的需要，即使在扩展到数万亿个参数时也是如此。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 本文的主要贡献如下：<br>
- 大型模型训练的内存和性能特征描述了不同组成部分的内存需求（第3节）以及训练的带宽需求（第4节），以确保训练的高效性。<br>
- ZeRO-Infinity（第5、6和第7节）是一种新颖的深度学习训练系统技术，包括五项创新技术，以解决内存和带宽需求，提供前所未有的模型规模，并且易于使用，同时实现出色的训练效率：**i) 无限卸载引擎**：充分利用现代集群上的异构架构，同时利用GPU、CPU和NVMe内存以及GPU和CPU计算能力。**ii) 内存中心平铺(memory-centric tiling)**：处理大规模运算符(超大数据层)，无需模型并行化。**iii) 带宽中心分区(bandwidth-centric partitioning)**：利用所有并行设备的聚合内存带宽。**iv) 重叠中心设计(overlap-centric design)**：将计算和通信重叠，提高效率。**v) 启发式实现**：避免模型代码重构，简化使用过程。<br>
- ZeRO-Infinity经过广泛(extensive)的评估，展示了以下内容：i) 在32个NVIDIA DGX-2节点（512个V100 GPU）上运行32万亿参数的前所未有的规模；ii) 在相同硬件上实现出色的训练效率，吞吐量超过25 petaflops( $10^{15}$ 一千万亿); iii) 兆(万亿)参数模型的超线性可扩展性；iv) 可访问性和易用性：在单个DGX-2节点上对万亿参数模型进行微调，无需使用任何模型并行化或模型代码重构；v) ZeRO-Infinity中不同技术对模型规模和效率的影响（第8节）。<br>
- 论文讨论了ZeRO-Infinity及其对未来硬件系统设计的潜在影响(在第9节)。
- 在DeepSpeed2中有一个ZeRO-Infinity的开源实现，DeepSpeed(https://www.deepspeed.ai/)是一个深度学习优化库，旨在使分布式训练变得简单、高效和有效，已经在深度学习社区广泛采用。<br>

# 2 背景和相关工作
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**数据、模型、流水线和3D并行化** 并行化是在大规模训练中训练大型模型的重要策略。对于适合设备内存进行训练的模型，可以使用数据并行化（DP）将训练扩展到多个设备。当模型无法适合(fit)设备内存时，可以使用模型并行化（MP）[7, 17, 18]和流水线并行化（PP）[7–9]将模型在进程之间进行垂直和水平分割。3D并行化[14, 15]结合了数据、模型和流水线并行化的优点，使其能够高效地扩展到数万亿个参数。虽然3D并行化可以非常高效，但它需要：i）对模型进行重构，将模型分割为模型和流水线并行组件；ii）具有复杂依赖关系图的模型难以表达为负载平衡的流水线阶段；iii）模型大小受限于可用的GPU内存总量。我们建议读者参考Ben-Nun和Hoefler [19]的详尽调查报告，了解深度学习中的并行化技术。<br>
*注释：在本文中，我们区分了模型并行化和流水线并行化，其中前者具体指基于张量切片的方法，并不包括流水线并行化。* <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**ZeRO(Zero Redundancy Optimizer)** ZeRO[11]通过在数据并行进程之间分割三个模型状态（即优化器状态、梯度和参数）而不是复制它们，消除了数据并行进程之间的内存冗余。通过这样做，与传统的数据并行化相比，它提高了内存效率，同时保持了计算粒度和通信效率。ZeRO有三个阶段，对应于三个模型状态：第一阶段（ZeRO-1）仅分割优化器状态，第二阶段（ZeRO-2）分割优化器状态和梯度，最后阶段（ZeRO-3）分割所有三个模型状态。在ZeRO-3中，模型每一层的参数归属于唯一的数据并行进程。在训练过程中，ZeRO-3确保在运算符执行之前，通过从所有者进程发出广播通信集合，提供前向或后向传递操作所需的参数。在运算符执行之后，ZeRO-3还会**移除参数**，因为它们在下一个运算符的前向或后向传递之前不再需要。此外，在训练的参数更新阶段，ZeRO-3确保每个数据并行进程只更新其所拥有的参数对应的优化器状态。因此，除了即时计算所需的参数外，ZeRO-3可以在整个训练过程中保持所有模型状态的**分割状态**。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**异构训练方法**，基于CPU内存的多种异构方法[20-26]中，ZeRO-Offload [12] 是用于多GPU的大型模型训练的最先进技术。ZeRO-Offload建立在ZeRO-2(2 means stage 2)的基础上，将梯度(grad)和优化器状态(optim states)存储在CPU内存中。在GPU设备不足以存储优化器状态和梯度时，ZeRO-Offload利用CPU内存。然而，它仍然需要将参数(parameters)存储在GPU内存中，并在所有设备之间进行复制。因此，ZeRO-Offload的模型规模受限于单个GPU设备内存可以容纳的参数总数。由于子优化(suboptimal)的数据划分和有限的PCIe带宽，ZeRO-Offload还需要较大的批量大小以保持高效。我们通过ZeRO-Infinity解决了ZeRO-Offload的这些限制。在基于NVMe的方法方面，Zhao等人 [27] 使用分层参数服务器设计将稀疏参数卸载到SSD上，以创建一个庞大规模的DL广告系统。相比之下，ZeRO-Infinity旨在成为一个通用的DL系统，用于训练庞大的**稠密模型**。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**减少激活内存** 激活是在前向传播过程中产生的中间结果，需要保留以计算后向传播中的梯度。通过压缩[28]、激活检查点[29, 30]或实时分析[31]等多种方法，已经进行了多项努力来**减少激活所需的内存**。ZeRO-Infinity与激活检查点结合使用，以减少激活内存。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Adam Optimizer和混合精度训练** 自适应优化方法 [32-35] 对于有效地训练大型模型以实现最先进的性能和准确性至关重要。与SGD相比，它通过在每个模型参数和梯度上维护细粒度的一阶和二阶统计信息，以显著的内存开销为代价。Adam [33] 是在大型模型训练中最常用的优化器。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通常情况下，大型模型的训练采用混合精度训练的方式，其中前向传播和反向传播使用FP16进行计算，而参数更新则使用FP32进行计算 [36]。这利用了现代GPU上可用的张量核心单元的性能加速 [37]。<br>

# 3 内存需求
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本节描述了深度学习训练的内存需求。虽然我们的方法是通用的，但我们将具体分析重点放在基于Transformer [38] 的架构上，因为所有超过十亿参数的最先进模型都遵循该架构。我们的分析假设使用Adam优化器进行混合精度训练，因为这是训练基于Transformer模型的事实上的标准方法。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;训练所需的内存可以分为两个组成部分：i) **模型状态，包括优化器状态、梯度和模型参数**；ii) **残差状态主要指激活内存**。为了研究在异构资源上的训练，我们还对GPU的工作内存进行了描述，描述了GPU上必须可用的最小内存量以支持训练，假设模型和残差状态可以成功地从GPU内存中卸载。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;模型状态的内存：模型状态由优化器状态、梯度和参数组成。对于使用Adam优化器的混合精度训练，参数和梯度以FP16格式存储，而优化器状态由FP32的动量、方差、参数和梯度组成。每个参数需要**20字节**的内存。基于Transformer的模型中的参数总数主要取决于隐藏维度 (ℎ𝑑) 和Transformer层数 (𝑛𝑙)。一个Transformer块中几乎所有的参数来自于每个块内的四个线性层，其大小分别为：(ℎ𝑑, 3ℎ𝑑)、(ℎ𝑑, ℎ𝑑)、(ℎ𝑑, 4ℎ𝑑)和(4ℎ𝑑, ℎ𝑑)。因此，可以近似计算基于Transformer的模型的总参数数目为：<br>
$$12 \times n l \times h d^{2} \ldots\ldots(1)$$
所需的总内存大小为：<br>
$$240 \times n l \times h d^{2} \ldots\ldots(2)$$
字节，来存储模型状态。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;根据图2a第5列所示，存储类似于GPT-3的基于Transformer的模型的模型状态所需的内存，其参数数量从1000亿到1万亿不等，这是通过变化隐藏维度和层数得到的。为了将内存需求放入上下文中，图2b第3列显示了单个NVIDIA V100 DGX-2盒子以及DGX2 SuperPOD集群中可用的总GPU内存。请注意，仅为了适应**1000亿参数模型的模型状态需要64个GPU**。适应**1万亿参数模型需要超过512个GPU**，而拥有**10万亿参数模型甚至超出了一个庞大的1536个GPU集群的范围**。<br>

![figure2](images/zero-infinity-figure2.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**残差状态的内存**：残差状态主要包括激活内存，其大小取决于模型架构、批次大小 (𝑏𝑠𝑧) 和序列长度 (𝑠𝑒𝑞)，可能会非常大。从积极的一面来看，可以通过激活检查点 [29] 显著减少激活所需的内存，但这会以**0.33倍**的额外计算为代价，换取激活内存的减少。例如，Turing-NLG 17.2B和GPT-3 175B等大型模型都是使用**激活检查点**进行训练的。存储激活检查点所需的内存(bytes)估计为：<br>
$$2 \times bsz \times seq \times h d \times nl / ci \ldots\ldots(3)$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其中，𝑐𝑖是两个激活检查点之间的Transformer块数量，𝑏𝑠𝑧 × 𝑠𝑒𝑞 × ℎ𝑑是每个Transformer块的输入大小。图2a的第7列显示了存储激活检查点所需的内存，假设批次大小为32，序列长度为1024，并**假设我们在每个Transformer块中存储一个激活**。许多现代GPU集群每个节点有8-16个GPU，因此我们选择每个GPU的批次大小为2-4，从而保守地估计每个节点中的激活批次大小为32。虽然生成的激活检查点(第7列)比完整的激活集合（第6列）小几个数量级，但在超过一万亿参数的情况下，对于考虑的批次大小和序列长度，它们仍然变得太大，无法适应GPU内存中。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **模型状态(不考虑激活)工作内存(MSWM)** 是在所有模型状态已经卸载到CPU或NVMe后，在模型中执行前向或反向传播所需的GPU内存的最小数量。这**大致等于模型中最大单个运算符的参数和梯度的大小**，因为**至少需要足够的内存来保存参数及其梯度用于反向传播**。对于基于Transformer的模型，最大的运算符是将隐藏状态从ℎ𝑑变换为4ℎ𝑑的线性层。该线性层的参数和梯度的大小(以字节为单位)为：<br>
$$4 \times hd \times 4hd \ldots\ldots(4)$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;请注意，MSWM(model states working memory)(图2a的第8列)在超过1,000亿个参数时显著增长，需要多个连续内存中的多个GB，这可能导致在训练过程中由于缺乏足够的连续内存来满足这些要求而耗尽内存。3D并行性等最先进的方法通过**模型并行性**来解决这个问题，通过将单个运算符分割到多个GPU上。在第5.1.3节中，我们将讨论一种新颖的方法来解决这种大规模模型状态工作内存(working memory)的问题，而无需使用模型并行性。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**激活工作内存(AWM)** 是在执行实际的反向传播之前，用于重新计算激活的反向传播过程中所需的内存。它表示两个连续激活检查点之间的激活大小。例如，如果我们在每个Transformer块创建一个激活检查点，那么**内存大小由每个Transformer块的总激活大小决定**。这个大小以字节为单位，可以近似(approximately)表示为:
$$bsz \times seq \times ci \times\left(16 \times hd+2 \times attn-heads \times seq\right) \ldots\ldots(4)$$
图2a的第8列显示，即使在𝑐𝑖 = 1的情况下，AWM在超过10万亿个参数时也会变得很大。与只包含单个参数和梯度的MSWM不同，AWM由几十个激活组成，并且只要总的AWM可以适应GPU内存，就不会因为缺乏连续内存而导致内存问题。<br>

# 4. 带宽需求
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将模型状态(model states)卸载到CPU和NVMe内存的一个关键问题是它们有限的带宽是否会影响训练效率。本节将描述带宽对训练效率的影响。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们首先定义一个效率度量指标(efficiency metric)。假设工作负载在没有任何计算和通信重叠(没有重叠)的情况下执行，我们可以使用峰值计算吞吐量( $𝑝𝑒𝑎𝑘_{𝑡𝑝}$ )、数据移动带宽（𝑏𝑤）以及其算术强度(intensity)(𝑎𝑖𝑡)来估算训练效率。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;工作负载的算术强度(AIT)是总计算量与计算所需数据之间量的比值。**它描述了每个数据移动所需的计算量**。较高的算术强度意味着对数据移动带宽的要求较低，因为对于每个加载的数据，加速器可以执行更多的计算。效率度量指标可以如下计算: <br>

$$ \text{computed time} = \frac{\text{𝑡𝑜𝑡𝑎𝑙 𝑐𝑜𝑚𝑝𝑢𝑡𝑎𝑡𝑖𝑜𝑛}}{peak_{tp}}$$

$$ 𝑎𝑖𝑡 = \frac{\text{𝑡𝑜𝑡𝑎𝑙 𝑐𝑜𝑚𝑝𝑢𝑡𝑎𝑡𝑖𝑜𝑛}}{\text{𝑡𝑜𝑡𝑎𝑙 𝑑𝑎𝑡𝑎 𝑚𝑜𝑣𝑒𝑚𝑒𝑛t}}$$

$$ \text {𝑐𝑜𝑚𝑚𝑢𝑛𝑖𝑐𝑎𝑡𝑖𝑜𝑛 𝑡𝑖𝑚𝑒} = \frac{\text{𝑡𝑜𝑡𝑎𝑙 𝑑𝑎𝑡𝑎 𝑚𝑜𝑣𝑒𝑚𝑒𝑛𝑡}}{bw} = \frac{\text{total computation}}{𝑎𝑖𝑡 \times 𝑏w}$$

$$𝑒𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑐𝑦 = \frac{\text{𝑐𝑜𝑚𝑝𝑢𝑡𝑒 𝑡𝑖𝑚𝑒}}{\text{𝑐𝑜𝑚𝑝𝑢𝑡𝑒 𝑡𝑖𝑚𝑒} + \text{𝑐𝑜𝑚𝑚𝑢𝑛𝑖𝑐𝑎𝑡𝑖𝑜𝑛 𝑡𝑖𝑚e}}$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;效率可以写成 $𝑝𝑒𝑎𝑘_{𝑡𝑝}$ 、𝑏𝑤和𝑎𝑖𝑡的函数:
$$𝑒𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑐𝑦 = \frac{𝑎𝑖𝑡 \times 𝑏𝑤}{𝑎𝑖𝑡 \times 𝑏𝑤 + 𝑝𝑒𝑎𝑘_{𝑡𝑝}} \ldots\ldots(6)$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们将使用这个简单的效率方程来描述训练大规模模型所需的数据移动带宽。但在此之前，我们将首先对DL训练工作负载进行𝑎𝑖𝑡的量化。<br>

## 4.1 在DL训练中量化AIT
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;模型状态和激活检查点可能具有不同的𝑎𝑖𝑡(算术强度)。我们可以通过首先确定DL训练**每次迭代中的总计算量**，然后确定**每个模型状态和激活的数据移动量**来对它们进行量化。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;每次迭代的总计算量主要由Transformer的线性层计算决定。对于前向传播(forward propagation)，可以近似表示为参数数量(params)、序列长度(seq)和批大小(bsz)的函数，即2 × 𝑏𝑠𝑧 × 𝑠𝑒𝑞 × 𝑝𝑎𝑟𝑎𝑚𝑠。反向传播的计算成本大约是前向传播的**两倍**。此外，激活检查点在反向传播期间需要进行额外的重新计算，因此每次迭代的总计算量为：<br>

$$computation per iter = 2 \times 4 \times bsz \times seq \times params \ldots\ldots(7)$$
$$= 2 \times 4 \times 12 \times 𝑏𝑠𝑧 \times 𝑠𝑒𝑞 \times 𝑛𝑙 \times ℎ𝑑 \ldots\ldots(8)$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**关于参数和梯度的算术强度AIT** 在前向传播和反向传播过程中，模型参数必须至少加载两次，即在前向传播期间和实际的反向传播期间，从源位置加载到GPU寄存器(registers)，导致数据移动量为2×𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠。在存在激活检查点的情况下，参数可能会额外加载一次，用于在反向传播过程中进行重新计算(re-computation)，增加了另外的1×𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠的数据移动量。此外，梯度必须至少从GPU寄存器存储到最终位置一次，增加了1×𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠的数据移动量。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;因此，假设参数和梯度存储在相同的最终位置上，前向传播和反向传播期间的总数据移动量将为4×𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠，即2×4×𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠字节(bf16)。每次迭代的总计算量由第4.1节(上述)给出。因此，相对于(w.r.t --> with respect to)参数和梯度的𝑎𝑖𝑡为：<br>

$$𝑠𝑒𝑞 \times bsz \ldots\ldots(9)$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**关于优化器状态的算术强度(AIT)** 在优化器步骤中，优化器状态必须至少被读取一次，并且优化器状态必须至少被写入一次。因此，总的数据移动量为2×𝑜𝑝𝑡𝑖𝑚𝑖𝑧𝑒𝑟_𝑠𝑡𝑎𝑡𝑒𝑠，大约为2×16×𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠字节。每次迭代的总计算量由第4.1节给出。因此，在完整的训练迭代过程中，相对于优化器状态的𝑎𝑖𝑡为：

$$ 𝑠𝑒𝑞 \times bsz / 4 \ldots\ldots(10)$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**关于激活检查点的AIT** 在前向传播期间，激活检查点必须保存到它们的最终位置，并在反向传播期间检索。因此，相对于激活检查点的总数据移动量，以字节为单位，由2×𝑡𝑜𝑡𝑎𝑙_𝑎𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛_𝑐ℎ𝑒𝑐𝑘𝑝𝑜𝑖𝑛𝑡𝑠_𝑖𝑛_𝑏𝑦𝑡𝑒𝑠给出，根据方程(3)，它等于4 × 𝑛𝑙/𝑐𝑖 × ℎ𝑑 × 𝑠𝑒𝑞 × 𝑏𝑠𝑧。每次迭代的总计算量由第4.1节给出。因此，相对于激活检查点的𝑎𝑖𝑡为：

$$ 24 \times hd \times ci ldots\ldots(11)$$

## 4.2 带宽需求
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由于AIT的变化，为了达到良好的效率，模型状态和激活检查点具有非常不同的带宽需求。前者仅取决于批量大小和序列长度，而后者仅取决于激活检查点的频率和模型的隐藏维度大小。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;除了AIT之外，efficiency的带宽需求还取决于 $𝑝𝑒𝑎𝑘_{𝑡𝑝}$ ，如方程(6)所示。利用 $𝑝𝑒𝑎𝑘_{𝑡𝑝}$ 和𝑎𝑖𝑡，我们首先展示了效率如何随着不同模型和残差状态的带宽变化而变化，然后讨论了这些状态对于DL训练效率的带宽需求。我们的方法是通用的，可以应用于理解对于任何当前或未来一代集群的带宽需求。在这里，我们以NVIDIA V100 DGX-2 SuperPOD集群作为示例平台.<br>

![figure3](images/zero-infinity-figure3.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;利用第4.1节中的𝑎𝑖𝑡表达式和基于方程(6)的效率指标，图3展示了相对于参数和梯度、优化器状态以及激活检查点的**可用带宽与效率**之间的关系。为了生成这些图表，我们根据第4.1节中推导的表达式计算了𝑎𝑖𝑡，针对不同的批量大小、序列长度和模型配置进行了变化。具体来说，我们使用了序列长度为1024，与GPT-2 [2]、Megatron-LM [7]和Turing-NLG [39]所使用的序列长度相同。我们将批量大小范围从1变化到16，分别捕捉大型GPU和小型GPU的实验。当在大量GPU上运行时，每个GPU使用较小的批量大小，而在相对较少的GPU上训练时，每个GPU使用较大的批量大小，以保持合理的有效批量大小进行训练。我们的隐藏大小(hidden size)范围从8K到64K，代表具有数千亿参数的模型，到拥有数万亿参数的模型,如图2a所示。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了确定此分析中的 $𝑝𝑒𝑎𝑘_{𝑡𝑝}$ ，我们采用了经验方法(看注释)。我们在单个NVIDIA V100 DGX-2盒子上运行了具有上述配置的模型，并关闭了所有非GPU通信，以模拟几乎无限带宽的情况。根据隐藏大小(hidden-size)为8K-64K，所实现的性能范围为每个GPU的62-78 TFlops。为了本分析的目的，我们使用平均值70 TFlops/GPU来表示 $𝑝𝑒𝑎𝑘_{𝑡𝑝}$ (注释)。

*请注意，𝑝𝑒𝑎𝑘_{𝑡𝑝} 并不是理论硬件峰值，而是在没有任何通信瓶颈的情况下可以实现的峰值性能。* <br>
*另请注意，最终结果结果将根据所使用的𝑝𝑒𝑎𝑘_𝑡𝑝值而有所不同，而此分析只是一个数据点，旨在指导我们特定在NVIDIA V100 DGX-2集群上理解DL工作负载的效率和带宽之间的关系。此外，该结果仅考虑了模型状态和激活之间的效率和带宽关系，一次只考虑一个状态，假设其他状态具有无限带宽，以分离出每个状态的带宽需求。* <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**关于参数(with respect to)和梯度的带宽**，图3a显示，即使在最小的批量大小情况下，如果参数和梯度的带宽超过70 GB/s，我们可以实现**超过50%的效率**。在这个带宽下，理论上**可以完全重叠数据传输和计算**，以达到100%的效率。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**关于优化器状态的带宽**，图3b显示，**为了实现50%的效率**，优化器状态需要近**4倍**的更高带宽，相比之下，参数和梯度的带宽要求较低。此外，优化器状态在前向和反向传播的结束时进行更新，**无法与计算重叠**。因此，为了保持整体DL工作负载的高效性，它们需要更大的带宽。例如，要实现每个GPU的批量大小为2的90%效率，需要近1.5 TB/s的有效带宽，这甚至超过了GPU的内存带宽。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**关于激活内存的带宽**，图3c还显示，启用激活检查点的情况下，仅仅2 GB/s的带宽就能够维持超过50%的效率，即使hidden size为2K。一旦隐藏大小超过8K，带宽需求降低到不到1 GB/s。<br>

# 5 ZERO-INFINITY设计概述
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在本节中，我们将介绍ZeRO-Infinity的设计选择概述，这些选择使其能够实现前所未有的模型规模，并提供出色的训练效率和易用性。ZeRO-Infinity的整体设计如图4所示，并在下文中进行讨论。<br>

![figure4](images/zero-infinity-figure4.jpg)

## 5.1 前所未有规模的设计
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;现代GPU集群在存储器方面具有高度异构性(heterogeneous)。除了GPU内存外，它们还具有CPU内存以及比GPU内存大50倍、比CPU内存大近20倍的大规模NVMe存储器（参见图2b）。
*注释：NVMe（Non-Volatile Memory Express）是一种高性能、低延迟的存储器接口协议，专为固态硬盘（SSD）和闪存存储器设计*
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们开发了ZeRO-Infinity，这是一个用于DL训练的并行系统，通过利用现代GPU集群中的异构内存系统，可以突破GPU内存的限制。图1比较了3D并行性和ZeRO-Infinity所能实现的最大模型大小。ZeRO-Infinity在每个NVIDIA V100 DGX-2节点上支持万亿个参数，**比3D并行性增加了50倍**。<br>

![figure1](images/zero-infinity-figure1.jpg)

### 5.1.1 模型状态的无限卸载引擎(Infinity offload engine for model states)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ZeRO-Infinity是在ZeRO-3 [11] 的基础上构建的，它将所有模型状态进行分区，以消除内存冗余，如2节所述。与现有的任何ZeRO技术系列不同，ZeRO-Infinity设计了一个强大的卸载机制，称为无限卸载引擎，可以根据内存需求将所有分区的模型状态卸载到CPU或NVMe内存，或者保留在GPU上。请注意从图2a和图2b可以看出，即使是一个拥有100万亿参数的模型所需的模型状态，也可以适应DGX-2集群（1536个GPU，96个节点）的总体NVMe内存。因此，无限卸载引擎使得ZeRO-Infinity可以适应具有数万亿参数的模型的模型状态。有关更多详细信息，请参阅第6节。<br>

### 5.1.2 激活的CPU卸载
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;除了模型状态之外，ZeRO-Infinity还可以在必要时将激活内存卸载到CPU内存中。需要注意的是，一个拥有1万亿参数的模型所需的激活**检查点(0.76 TB)** 可以轻松适应DGX-2系统上可用的1.5TB CPU内存，而一个拥有100万亿参数的模型所需的3 TB激活检查点也可以适应下一代硬件的CPU内存。因此，通过将激活检查点卸载到CPU内存，ZeRO-Infinity可以适应具有数万亿参数的模型的激活检查点。<br>
### 5.1.3 工作内存的以内存中心切片技术
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了减少大型模型的DL训练对工作内存的需求，ZeRO-Infinity引入了一种称为内存中心切片的新技术，该技术利用了ZeRO-3的数据获取和释放模式，通过**将大型运算符分解为可以按顺序执行的较小切片来减少工作内存的需求**。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;例如，为了减少大型线性运算符(linear 算子)的工作内存，ZeRO-Infinity将该运算符表示为数学上等价的由原始运算符的参数切片组成的较小线性运算符序列，并按顺序执行它们。结合ZeRO-3使用时，每个切片的参数和梯度可以逐个获取和释放，从而将工作内存减少与切片数量成比例的量。因此，ZeRO-Infinity可以支持任意大小的运算符，而无需依赖模型并行性将其适应于有限的GPU内存中。<br>

## 5.2 优秀的训练效率设计
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 将所有模型状态和激活内存卸载到CPU或NVMe只有在ZeRO-Infinity能够**在卸载的情况下实现高效率**才是切实可行的。实际上，这是非常具有挑战性的，因为CPU内存的带宽比GPU内存慢一个数量级，而NVMe带宽比CPU内存带宽慢一个数量级。此外，从GPU读取和写入这些内存的速度更慢（参见图2b）。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在像DGX-2这样的系统上，根据我们在第4节的分析，为了实现高效的DL训练，参数和梯度、优化器状态以及激活检查点的带宽必须分别大于70GB/s、1.5TB/s和1-4GB/s。下面我们将讨论ZeRO-Infinity如何实现所需的带宽以达到优秀的效率。<br>
### 5.2.1 关于参数和梯度的效率
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;参数和梯度的数据传输带宽必须大于**70GB/s**，接近DGX-2集群上可用的GPU-GPU带宽[40]。因此，像ZeRO3 [11]这样的深度学习并行训练解决方案，在正向或反向传播之前将参数从拥有者GPU广播到其他GPU，**只要通信重叠**，就可以高效运行。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;相反，单个GPU到CPU内存或NVMe之间微弱的12GB/s的PCIe带宽（见图2b）或反之(CPU 到 GPU)，根本无法支持大规模异构训练。因此，现有的异构解决方案(如ZeRO-Offload)要求在广播之前必须先将参数从CPU移动到拥有者GPU，这就需要每个GPU非常大的批量大小才能达到足够的活跃度以在有限的带宽下实现高效。这带来了两个问题：i)对于庞大的模型，激活内存的大小甚至超过CPU内存的容量；ii)当扩展到数百或数千个GPU进行有效收敛时，有效批量大小变得过大。<br>
*(注释：CPU和NVMe的带宽分别约为100GB/s和25GB/s，但从CPU或NVMe读取数据到单个GPU的速度受限于可实现的PCIe带宽，大约在10-12GB/s左右。)* <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ZeRO-Infinity通过两种方式解决了这些挑战：**i)带宽为中心的分区**：一种新颖的数据映射和并行数据检索策略，用于被卸载的参数和梯度，使得ZeRO-Infinity能够实现几乎无限的异构内存带宽（详见第6.1节），**ii)基于重叠的设计**: 使得ZeRO-Infinity不仅可以将GPU之间的通信与计算重叠，而且还可以**通过PCIe重叠NVMe-CPU和CPU-GPU之间的通信**（详见第5.1.3节）。<br>
### 5.2.2 关于优化器状态的效率
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;与在正向和反向传播过程中按顺序使用和生成的参数和梯度不同，优化器状态可以同时并行更新。这个特性被ZeRO-3和ZeRO-Offload所利用，它们分别将优化器状态存储和更新在GPU(存储)和CPU(更新)内存中，并且**跨所有可用的GPU和CPU同时进行**。因此，随着GPU或CPU数量的增加，聚合的GPU或CPU内存带宽**可以远远高于所需的1.5TB/s**。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由于ZeRO-Infinity是基于ZeRO-3构建的，它也可以利用聚合的GPU和CPU内存带宽以及聚合的CPU计算资源来进行优化器步骤，当将优化器状态卸载到CPU内存时。然而，通过NVMe卸载，**需要将数据以适合CPU内存的块大小从NVMe传输到CPU内存**，并**逐块**进行优化器步骤。因此，优化器步骤(optimizer step)**受限于NVMe-CPU内存带宽**：虽然ZeRO-Infinity可以实现跨多个节点的聚合NVMe带宽，但关键是要实现每个节点接近峰值的NVMe带宽，以支持超过1.5TB/s的必要带宽，并且**使用尽可能少的节点和尽可能小的批量大小**。此外，将数据从NVMe传输到CPU内存，或从CPU内存传输到GPU内存的过程可能导致GPU和CPU内存的碎片化，即使仍有大量内存可用，也可能导致内存不足。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**无限卸载引擎(Infinity offload engine)** 不仅可以实现接近峰值的NVMe带宽，还可以允许ZeRO-Infinity将NVMe到CPU的读取与CPU到NVMe的写入以及优化器步骤的**CPU计算**同时重叠进行，从而使得ZeRO-Infinity在少量GPU上使用适度的批量大小时保持高效，在大量GPU上使用小批量大小时也能高效运行。同时，它通过仔细重用(内存池)用于数据传输的临时缓冲区来最小化内存碎片化。我们在第6节详细讨论了Infinity offload engine的优化措施。<br>
### 5.2.3 关于激活的效率
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在DGX-2节点上，**每个GPU可以以大约3GB/s的速度通过PCIe并行读写数据到CPU内存**，这使得可以将激活检查点卸载到CPU内存，同时对于大于8K或更大的隐藏大小仍保持超过80%的效率。为了在较小的隐藏大小下也能保持高效率，ZeRO-Infinity可以降低激活检查点的频率，并且**有效地将激活检查点的通信与GPU上的正向和反向计算重叠进行**，包括与CPU内存之间的通信。<br>

## 5.3 便于使用的设计
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用ZeRO-Infinity，数据科学家不再需要将他们的模型适应多种形式的并行性，比如3D并行。这是因为ZeRO-Infinity中的以**内存为中心的切片设计**(在第5.1.3节中讨论)旨在降低大型单个层所需的GPU内存，否则需要使用模型并行（张量切片）来适应GPU内存中的层。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;此外，ZeRO-Infinity在PyTorch中的实现方式消除了在扩展到数万亿个参数时需要手动重构模型代码的需求。这是通过两个自动化功能实现的：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i)在训练期间，在参数被使用之前和之后，自动进行数据移动来收集和分区(partition)参数。ZeRO-Infinity通过向PyTorch子模块(submodule)注入&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i)pre forward/backward钩子来触发allgather集合操作，以在前向/反向传递之前收集所需的参数，以及ii）post forward/backward 钩子来触发参数/梯度分区，并可选择将它们卸载到CPU或NVMe（详见第7.1节）。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ii)在初始化期间自动进行模型分区，以便不能适应单个GPU或CPU内存的模型仍然可以进行初始化，而无需手动将模型分区到数据并行进程中。ZeRO-Infinity通过包装(wrapping)所有模块类的构造函数来实现这一点，以便在初始化过程中创建每个子模块的参数后**立即对其进行分区和卸载**。整个模型不会完全实例化在单个数据并行进程上（详见第7.2节）。<br>

# 6 效率优化
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在本节中，我们深入探讨了在第5节中介绍的优化措施，这些措施使得ZeRO-Infinity能够实现出色的效率。<br>

## 6.1 带宽为中心的分区
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ZeRO-Infinity实现了一种新颖的数据映射和检索策略，以解决NVMe和CPU内存带宽的限制。与ZeRO [11]和ZeRO-Offload [12]不同，这两种方法中每个层的参数由单个数据并行进程拥有，并在需要时向其他进程广播，ZeRO-Infinity将各个参数分区到所有数据并行进程，并在需要访问参数时**使用allgather而不是广播**。需要注意的是，无论是广播还是allgather通信收集，如果数据位于GPU上，其通信成本是相同的。因此，在仅使用GPU进行训练时，这没有区别。然而，当数据位于NVMe或CPU上时，这将产生重大影响。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在基于广播的方法中，由于每个参数完全由一个数据并行进程拥有，参数**必须首先通过PCIe从其源位置传输到GPU内存**，然后才能进行广播。需要注意的是，这个过程只有一个PCIe可以活动，而连接到其他所有GPU的所有PCIe链路都处于空闲状态。相反，在ZeRO-Infinity中，**通过分区参数和基于allgather的方法，所有PCIe链路都可以并行活动**，每个链路带来参数的 $1/𝑑𝑝_{𝑡ℎ}$ 部分，其中𝑑𝑝是数据并行度。因此，NVMe或CPU与GPU之间的有效通信带宽随着𝑑𝑝的增加呈线性增长。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;例如，使用基于广播的方法，在DGX-2盒子上进行16路数据并行时，CPU/NVMe到GPU的带宽保持在大约**12GB/s**（使用PCIe Gen 3）。然而，使用基于allgather的方法，有效可达到的带宽分别增加到约**48/25GB/s**（每个GPU为3.0/1.6GB/s）（参见图2b），仅受限于每个DGX-2节点的最大聚合PCIe带宽和最大NVMe带宽。从这里开始，带宽随着节点数量的增加呈线性增长。当在大规模进行大型模型的训练时，ZeRO-Infinity可以提供比训练所需的更多异构内存带宽（几乎无限）。例如，在64个DGX-2节点上，ZeRO-Infinity可以获得超过**3TB/s的CPU内存带宽**和超过**1.5TB/s的NVMe带宽**。<br>

# 6.2 重叠中心的设计
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;虽然ZeRO-Infinity可以在多节点设置中利用足够的异构内存带宽，但在单个GPU或单个节点设置中，带宽仍然可能成为瓶颈。即使是GPU-GPU的allgather通信，在使用小批量大小时也会对效率产生很大影响（图3）。此外，访问NVMe内存需要一个三步过程：i)从**NVMe读取数据到CPU内存**（nc-transfer），ii)将数据从**CPU内存复制到GPU内存**(cg-transfer)，iii)**执行allgather以在所有GPU上构建完整的参数**（gg-transfer）。这些数据移动的顺序性意味着，如果简单地进行，总通信时间将是这三个数据移动成本的总和，即使每个阶段的数据移动带宽单独足够，也会导致效率低下。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了解决这些问题，ZeRO-Infinity拥有一个重叠引擎，它不仅可以将GPU-GPU通信与GPU计算重叠，还可以同时重叠NVMe到CPU和CPU到GPU的通信。重叠引擎包括两个组件：i）一个动态预取器，用于在前向或反向传递中消耗参数之前重叠重构参数所需的数据移动；ii）一个通信和卸载重叠机制，用于并行执行梯度所需的数据移动和反向计算。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在ZeRO-Infinity中，动态预取器(dynamic prefetcher)会即时跟踪前向和反向计算过程，并构建每次迭代的操作符序列的内部映射。在每次迭代中，预取器会**记录**当前所处的**操作符序列位置**，**并预取**未来操作符所需的参数。预取器(prefetcher)了解三步通信过程，因此可以将一个参数的nc-transfer与其他参数的cg-transfer和gg-transfer重叠。例如，在执行第𝑖个操作符之前，预取器可以为第𝑖+3、𝑖+2和𝑖+1个操作符所需的参数分别调用nc、cg和gg-transfer。需要注意的是，所有这些数据移动可以与执行第𝑖个操作符并行进行。此外，ZeRO-Infinity可以在动态工作流的情况下**更新操作符序列映射**，以便在迭代过程中根据前向和反向传播的变化进行适当的预取.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;类似地，在反向传递中，ZeRO-Infinity可以将第𝑖+1个操作符的参数梯度的reduce-scatter与第𝑖个操作符的计算(computation)重叠，同时将第 𝑖+2个操作符的梯度的reduce-scatter分区梯度传输到CPU或NVMe。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;借助这种强大的以重叠为中心的设计，即使在使用少量GPU和每个GPU的小批量大小进行训练时，ZeRO-Infinity也可以隐藏大部分数据移动的时间。<br>
*(https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)* <br>

## 6.3 无限卸载引擎
无限卸载引擎由两个主要组件组成：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**DeepNVMe** : 一个Infinity Offload引擎中的强大C++ NVMe读写库，DeepNVMe支持异步完成批量读写请求，并支持显式同步请求以刷新正在进行的读写操作。异步支持使得ZeRO-Infinity能够将这些请求**与GPU/GPU或GPU/CPU通信或计算重叠**。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;最重要的是，DeepNVMe能够在NVMe存储设备上实现接近峰值的顺序读写带宽。它通过一系列优化措施实现高性能，包括对I/O请求的积极并行化（无论是来自单个用户线程还是跨多个用户线程）、智能工作调度、避免数据复制和内存固定等。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**固定(pinned)内存管理层**为了确保从（到）NVMe/CPU存储的张量高性能读取（或写入），源（或目标）张量必须位于**固定内存缓冲区**中。然而，固定内存缓冲区是稀缺的系统资源，单个进程过度使用会降低整体系统性能或导致系统不稳定。该层通过重复使用少量（数十GB）固定内存来管理有限的固定内存供应，以将整个模型状态（高达数十TB）卸载到CPU或NVMe。内存缓冲区的重复使用可以防止CPU和GPU内存的内存碎片化。此层还为PyTorch张量提供固定内存数据，允许原地计算张量，然后将其写入NVMe，无需进一步复制，从而提高带宽。<br>
*(固定内存（pinned memory）是一种特殊类型的内存，它在物理内存中被固定，不会被操作系统进行页面交换或移动。相比于常规的内存分配，固定内存的主要优势在于它可以提供更高的内存访问性能和数据传输速度。)* <br>

# 7 方便的实施(EASE INSPIRED IMPLEMENTATION)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ZeRO-Infinity是基于PyTorch实现的，并且设计为在**不需要对模型代码进行重构的情况下使用**，类似于PyTorch中的标准数据并行训练。本节详细介绍了在实现这样一个系统时面临的一些挑战。<br>
## 7.1 自动化数据传输
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ZeRO-Infinity必须协调模型参数、梯度和优化器状态所组成的张量的移动。当一个张量不在活动使用中时，它会被分区存储在不同的工作节点上，并且可能会被卸载(offload)到CPU或NVMe内存中。系统必须确保这些张量在需要使用时能够及时存在于GPU内存中，并在后续重新进行分区。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PyTorch模型被表示为一个层次结构的模块(module)，代表神经网络的各个层。例如，Transformer架构中包含了自注意力和前馈网络等子模块。自注意力子模块又由线性变换和其他子模块组成。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ZeRO-Infinity通过递归(recursively)地向模型的子模块(module)注入钩子来自动化所需的数据传输。在子模块的前向传递(forward pass)开始时，**这些钩子确保子模块的参数可用于计算**，否则会执行适当的allgather收集操作，并阻塞(block)直到参数可用。本文第6.2节详细介绍的重叠中心设计对于减少由于参数通信而引起的停顿非常关键。在子模块的前向传递结束时，我们再次对参数进行分区(partition)，并可选择将它们卸载(offload)。反向传递的处理方式(be handled)类似。<br>
*(ZeRO-Infinity通过向模型注入钩子，并在子模块的前向传递和反向传递过程中进行参数的传输和重分区，实现了自动化的数据移动。这样可以确保在计算过程中所需的参数及时可用，并充分利用重叠计算和通信来最小化性能停顿。)* <br>
### 7.1.1 自动注册外部参数
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 在理想情况下，一个子模块的参数和梯度只在其自身的前向传递和反向传递中被访问，这样可以很容易地识别和自动化数据移动，如上文所讨论的。然而，某些模型架构是例外情况，其中在一个子模块中定义和分配的参数在另一个子模块的前向传递和反向传递中被使用。例如，像GPT [41] 这样的语言模型在网络的开头和结尾共享嵌入层(embedding)的权重，以将单词映射到向量(反之亦然)。我们将**跨模块边界使用的参数称为外部参数**。在存在外部参数的情况下，很难确定在一个子模块的前向传递和反向传递的开始时要收集哪些参数。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了解决这个问题，一种方法是将外部参数注册到ZeRO-Infinity中，以便在访问它们的子模块的前向传递和反向传递过程中收集它们。注册后，外部参数将像其他参数一样处理，并包含在前文第6.2节中描述的预取系统中(prefetching system)。我们提供了用于手动注册外部参数的API。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了改善用户体验，我们还提供了机制来检测这些情况并自动注册外部参数，这样用户就不需要进行任何代码更改：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**拦截(Intercepting)分区参数访问** PyTorch模块将其张量参数存储在哈希表中。在初始化时，我们用一个子类化的类型替换哈希表，并覆盖张量访问操作。当访问一个分区参数时，我们对该参数执行阻塞的allgather操作，将其注册为外部参数，然后返回收集到的参数。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**激活函数内省** 一个子模块可能会在其前向传递中返回一个参数，供另一个子模块的前向传递和反向传递使用。例如，Megatron-LM在线性层的前向传递中返回偏置向量，并由父级Transformer层模块使用。我们检查每个子模块的前向传递返回的激活输出，查找分区参数。如果发现一个分区参数，我们会收集并注册它作为外部参数。<br>
## 7.2 初始化期间的自动模型分区
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果模型很大，使用传统的数据并行方法在ZeRO-Infinity进行分区之前可能无法完全初始化模型，即在每个数据并行进程上复制模型。例如，一个5000亿参数的模型在半精度下将占用1TB的内存，因此一个每个节点有8个GPU的系统仅用于初始数据并行分配步骤就需要8TB的总CPU或GPU内存。这超出了节点上可用的GPU或CPU内存。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了解决这个限制，必须**在初始化时**对模型的每一层参数进行分区，而不是在整个模型初始化完成后再进行分区。为了实现这一点，我们提供了一个Python的ZeRO-Infinity上下文(context)，它修饰(decorates)了torch.nn.Module的__init__方法，这样在每个模块/子模块的初始化之后，分配在该模块/子模块下的参数会立即在数据并行进程组中进行分区。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;因此，在进行分区之前，只有各个子模块被完全初始化，而整个模型永远不会在所有数据并行进程上进行复制。在上述示例中，这个拥有5000亿参数的模型因此可以在初始化过程中完全分区，只需要**1TB的总CPU内存**，无论数据并行进程的总数是多少。<br>

# 8 评估
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本节评估了ZeRO-Infinity，展示了它在具有数万亿参数的模型上实现了出色的训练效率和可扩展性。我们还展示了ZeRO-Infinity中各种技术对模型规模和性能的影响。<br>
## 8.1 方法论
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**硬件**。我们在一台由512个V100 SXM3 32 GB GPU(32个DGX-2节点-每节点16个GPUs)组成的集群上进行了实验，**节点间通信带宽为800 Gbps**。
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**基准线**。对于没有模型并行性(mp)的实验，我们使用torch的分布式数据并行(DDP [42])作为基准线。对于具有模型并行性的实验，我们使用Megatron-LM [7]。对于每个实验，我们使用3D并行性 [13]、ZeRO [11]或ZeRO-Offload [12]中的相关最先进方法作为基准线(baseline)。
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**模型配置**。我们使用基于GPT的Transformer模型。我们将序列长度固定为1024，并根据隐藏维度和层数的不同来获得具有不同参数数量的模型。表格1提供了我们在整个评估过程中使用的具体模型配置。附录A中提供了其他配置的详细信息。

![table1](images/zero-infinity-table1.jpg)

## 8.2 模型规模和速度
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**模型规模**：ZeRO-Infinity可以训练具有超过32万亿参数的模型，而3D并行性的参数规模约为6500亿，这是目前的最先进水平，ZeRO-Infinity的模型规模比3D并行性提升了50倍（图1）。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**模型速度**：图5a展示了ZeRO-Infinity在**512个GPU**上训练高达20万亿参数模型的性能。对于5000亿参数的模型（接近3D并行性在这些资源上可以运行的最大规模），ZeRO-Infinity和3D并行性的吞吐量几乎相同，**表明ZeRO-Infinity的训练效率与最先进方法相当**。当进一步增加模型规模时，3D并行性会因为内存不足而无法继续运行，而ZeRO-Infinity可以训练高达20万亿参数（比之前大40倍）的模型，并且具有高达**49 TFlops/GPU**的出色吞吐量。在极端规模下，图5a展示了性能从10万亿参数（43 TFlops/GPU）降至20万亿参数（34 TFlops/GPU）的下降。这种下降不是由于NVMe带宽引起的，因为这两种模型尺寸都使用了NVMe卸载，而是由于每个GPU的批量大小非常小（表格1），这是**由于有限的CPU内存存储激活检查点所导致的**。这个问题可以通过增加CPU内存或在未来的实现中将激活检查点卸载到NVMe上来改善。<br>
## 8.3 超线性可扩展性
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;图5b展示了当训练一个1万亿参数的模型时，ZeRO-Infinity从4个节点（64个GPU）到32个节点（512个GPU）实现了超线性的可扩展性。这是一种弱可扩展性的结果，我们保持每个节点的批量大小不变，并随着节点数量的增加增加总批量大小。ZeRO-Infinity通过有效**利用聚合PCIe和NVMe带宽的线性增加**来加速参数和优化器状态的卸载，并利用额外节点的CPU计算进行参数更新，超过了完美的线性扩展。此外，ZeRO-Infinity仅使用4个节点就已经实现了超过2.8 petaflops（44 Tflops/GPU）的性能，这表明即使在适度规模下，聚合的NVMe带宽已足够实现良好的效率。<br>
## 8.4 大规模模型训练的民主化
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;图5c展示了在单个节点（16个GPU）上使用ZeRO-Infinity在没有任何模型并行性的情况下训练10亿至1万亿参数的模型的性能。对于高达1000亿参数的模型，ZeRO-Infinity实现了超过40 TFlops/GPU的出色性能，使得只使用一个DGX-2机箱就可以对GPT-3等模型进行**微调**成为可能。相比之下，**3D并行性无法扩展到超过200亿参数的模型**。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这些结果展示了ZeRO-Infinity的两个方面：i）在单个NVIDIA DGX-2节点上，可以轻松对具有高达1万亿参数的大模型进行微调，使得那些没有大型GPU集群的用户也能够使用。ii）易用性：使用ZeRO-Infinity可以训练这种规模的模型，无需结合模型并行性或流水线并行性，也无需对模型代码进行改写，使得数据科学家能够轻松地扩大他们的模型规模。<br>

![figure6](images/zero-infinity-figure6.jpg)

![table2](images/zero-infinity-table2.jpg)

## 8.5 系统特性对模型规模的影响
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们展示了不同设备放置策略对模型规模的影响，以及内存中心平铺（Sec. 5.1.3）对使用单个DGX-2系统（16个GPU）的最大隐藏层大小的影响。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**最大模型规模** 图6a展示了不同设备放置和划分策略（参见表2）对最大模型规模的影响。仅使用数据并行性时，由于GPU内存受限和模型状态冗余，我们只能达到14亿参数的规模。当我们引入ZeRO-2和ZeRO-Offload的优化器/梯度划分和CPU卸载时，我们能够在单个节点上将模型规模扩大9倍，达到130亿参数。在ZeRO-Infinity中，将参数状态划分和卸载到CPU，使我们几乎达到1000亿参数。然而，**最终的规模跃升是通过将模型状态卸载到NVMe来实现的**，最终达到1万亿参数的规模，相对于仅使用数据并行性，模型规模增加了700倍。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**最大隐藏层大小** 我们评估了内存中心平铺在存在内存碎片化的情况下实现大隐藏层大小(large hidden sizes)的影响。我们使用不同的隐藏层大小和平铺因子训练单层Transformer模型，以确定可以使用和不使用平铺训练的最大隐藏层大小。为了保持所有实验中的内存碎片一致，我们将总GPU内存预先划分为2GB的连续块，以便所有大于2GB的内存分配请求都会失败。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;图6b显示，**不使用内存中心平铺时可以训练的最大隐藏层大小为8K**，而使用平铺因子为16的内存中心平铺甚至可以训练具有64K的庞大隐藏层大小。通过内存中心平铺，ZeRO-Infinity通过避免对模型并行性的需求，极大地简化了DL系统堆栈，使数据科学家能够轻松地使用大隐藏层大小进行训练。<br>

# 8.6 系统特性对性能的影响
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们评估了无限卸载引擎（Sec. 5）、带宽中心划分（Sec. 6.1）、重叠中心设计（Sec. 6.2）和激活检查点卸载（Sec. 4.1）对训练速度的影响。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**ZeRO-Infinity与ZeRO-Offload的比较** 图6c展示了在一个具有8B参数的模型的反向传播时间上，使用ZeRO-Infinity与ZeRO-Offload将梯度卸载到CPU内存的影响。ZeRO-Infinity利用跨多个GPU的总PCIe带宽来卸载梯度，相对于受限于单个PCIe带宽的ZeRO-Offload，可以在64个GPU上实现近2倍的加速。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**预取和重叠** 图6d展示了在一个具有8B参数的模型和64个GPU的情况下，打开和关闭通信重叠和预取时的相对吞吐量差异。图中显示，对于每个GPU的小批量大小，预取和重叠对于实现良好的性能至关重要，而在大批量大小时，其影响逐渐减弱。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**激活检查点卸载** 图6e显示了在ZeRO-Infinity中将激活检查点卸载到CPU内存时，对训练吞吐量的影响。对于小的隐藏层大小，激活检查点的CPU卸载可以将训练吞吐量**降低**最多1.2倍，但对于32K和64K的隐藏层大小，影响很小，这表明可以在不影响大的隐藏层大小的效率的情况下将激活检查点卸载到CPU内存中。<br>

# 9 结论与未来影响
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在本文中，我们提出了ZeRO-Infinity，这是一种新颖的异构系统技术，利用GPU、CPU和NVMe存储器，实现了前所未有的模型规模，同时保持了优秀的效率，且易于使用。它对于我们如何考虑大型模型训练的内存提供了一种范式转变。不再需要将DL训练适配到速度极快但昂贵且有限的HBM2等内存中。ZeRO-Infinity证明，通过在多个设备上并行利用廉价且较慢但容量巨大的CPU或NVMe存储器，可以超越GPU内存瓶颈，实现对当前一代GPU集群进行高效训练所需的聚合带宽。<br>

![table3](images/zero-infinity-table3.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;展望未来，GPU和其他加速器的性能将变得更加强大，而对于高效训练所需的聚合带宽也将增加。表3显示，即使与NVIDIA V100 GPU相比，加速器(accelerators)的计算能力增加了10倍，在一个拥有512个加速器的集群中，ZeRO-Infinity只需要每个加速器和慢速内存之间的30GB/s带宽，就能保持高效性。实际上，通过使用当前的技术，通过NVLink [43]将加速器连接到慢速内存已经是可能的。例如，2018年推出的Summit超级计算机[44]每个GPU将NVIDIA V100 GPU与CPU内存连接速度达到40GB/s。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过ZeRO-Infinity，加速器设备的内存已经不再是模型规模或训练效率的限制因素。然而，要在合理的时间内训练具有数万亿或数百万亿参数的模型仍然需要计算能力的巨大飞跃，并且要在未来的设备上高效运行，需要相应提升的设备间带宽（表3）。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们希望，随着设备内存不再是限制因素，ZeRO-Infinity将激发未来更多关注计算能力和设备间带宽的创新，以支持模型规模的1000倍增长和相关的进步。这将推动未来超强加速器设备和超级计算集群的发展，为深度学习模型的发展和创新提供支持。<br>

![table4](images/zero-infinity-table4.jpg)

![table5](images/zero-infinity-table5.jpg)

![table6](images/zero-infinity-table6.jpg)

![table7](images/zero-infinity-table7.jpg)

![table8](images/zero-infinity-table8.jpg)
