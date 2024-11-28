# TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training
- [github](https://github.com/pytorch/torchtitan)

# 0 摘要
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;大型语言模型（LLM）的开发对于推动最先进的自然语言处理(sota)应用的发展起到了至关重要的作用。训练拥有数十亿参数(billons)和数万亿(trillons)标记的大型语言模型需要**复杂的分布式系统**，这些系统能够组合和比较多种最先进的技术，以便**在数千个加速器上实现高效扩展**。然而，**现有的解决方案复杂且分散在多个库/存储库中，缺乏互操作性，且维护起来非常繁琐**。因此，整理和实证比较训练方案需要相当大的工程投入。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本文介绍了TorchTitan，这是一个开源的、基于PyTorch的原生分布式训练系统，它**统一并推进了最先进的技术，简化了集成过程并降低了工程开销**。TorchTitan以模块化和可组合的方式实现了**3D并行性的无缝应用**，同时具备**弹性扩展能力(elastic scaling)**，以适应不断变化的计算需求。该系统提供了全面的日志记录、高效的检查点保存和调试工具，确保训练过程具备生产就绪性。此外，TorchTitan还融入了创新的**软硬件协同设计方案，利用诸如Float8训练和SymmetricMemory等尖端特性**，以最大限度地提高硬件利用率。作为一个灵活的实验测试平台(flexible experimental test bed)，TorchTitan促进(facilitates)了针对不同训练场景的自定义训练方案的整理和比较。通过利用TorchTitan，我们为Llama 3.1系列开发了优化的训练方案，并基于我们的实践经验，就如何选择和结合分布式训练技术以最大化训练效率提供了可行的指导。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们在包含80亿(8B)至4050亿(405B)参数的Llama 3.1系列大型语言模型（LLM）上对TorchTitan进行了全面评估，并展示了其卓越的性能、模块化组合能力和弹性扩展性(elastic scalability)。通过叠加训练优化技术，我们在NVIDIA H100 GPU上相对于优化后的基线实现了显著加速：在128个GPU规模下（Llama 3.1 8B）使用1D并行性获得了65.08%的加速，在256个GPU规模下（Llama 3.1 70B）额外使用2D并行性获得了12.59%的加速，以及在512个GPU规模下（Llama 3.1 405B）再额外使用3D并行性获得了30%的加速。<br>

# 1 介绍
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;大型语言模型（LLM）（Devlin, 2018; Liu et al., 2019; Radford et al., 2019; Chowdhery et al., 2023; Anil et al., 2023; Achiam et al., 2023; Dubey et al., 2024; Jiang et al., 2024; Abdin et al., 2024）处于自然语言处理（NLP）发展的前沿。它们是推动语言翻译、内容/代码生成、对话式人工智能、文本数据分析、创意写作与艺术、教育与研究等众多NLP应用发展的核心力量。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了达到最先进的性能，大型语言模型（LLM）需要数十亿个参数并训练超过万亿个标记。实现最先进的LLM性能需要庞大的规模，例如表现顶尖的模型Llama 3.1（4050亿参数，15万亿标记，3084万GPU小时，**1.6万个H100 GPU**）（Dubey等，2024）和谷歌的PaLM（5400亿参数，0.8万亿标记，940万TPU小时，6144个TPUv4芯片）（Chowdhery等，2023）。这些模型展现出了卓越的自然语言理解和生成能力，但需要大量的计算资源、内存和时间来训练，这凸显了推进自然语言处理所需的巨大投入。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;大型语言模型（LLM）的训练挑战正在从各个方面得到解决。在大规模上训练大型语言模型是一项艰巨的任务，它需要在**并行性、计算和通信之间取得微妙的平衡**，同时还要处理好复杂的内存和计算之间的权衡。训练所需的大量资源使得其容易受到GPU故障的影响，这强调了**需要高效的恢复机制和检查点策略来最大限度地减少停机时间**（Eisenman等，2022；Wang等，2023；Gupta等，2024；Maurya等，2024；Wan等，2024）。为了优化资源利用率并实现弹性扩展，关键在于结合多种并行技术，包括数据并行（Li等，2020；Rajbhandari等，2020；Zhang等，2022；Zhao等，2023）、张量并行（Narayanan等，2021；Wang等，2022；Korthikanti等，2023）、上下文并行（Liu等，2023；Liu和Abbeel，2024；NVIDIA，2023；Fang和Zhao，2024）以及流水线并行（Huang等，2019；Narayanan等，2019, 2021；Qi等，2023）。通过将这些并行技术与内存和计算优化技术相结合(stacking)，如激活重计算（Chen等，2016；Korthikanti等，2023；He和Yu，2023）、混合精度训练（Micikevicius等，2018, 2022）以及深度学习编译器（Bradbury等，2018；Yu等，2023；Li等，2024；Ansel等，2024），可以最大限度地提高硬件利用率。<br>

**采用最先进技术的现有系统的局限性**虽然最先进的分布式训练技术已经显著推动了该领域的发展，但采用这些技术的现有系统在解决阻碍研究者和行业从业者使用、采纳和有效应用的关键挑战方面仍存在不足。<br>

1.不可组合性：现有系统难以结合和叠加各种并行技术，限制了多维并行性的探索。进一步将它们与内存和计算优化相结合也充满挑战，从而阻碍了训练效率的提升。<br>
2.架构僵硬且单一：当前系统不具备模块化或可扩展性，导致难以集成和比较新技术、优化方法和硬件，也限制了它们对不断发展的机器学习环境的适应能力。<br>
3.硬件利用率低下：现有系统**未能充分利用高级硬件功能**，导致GPU效率不佳(MFU)，并且缺乏**可定制的激活检查点策略(checkpoint)** 来平衡内存和计算之间的权衡。<br>
4.对生产级训练的支持不足：现有系统缺乏可扩展且高效的分布式检查点功能，使得**故障恢复和模型保存变得繁琐**，并且通常没有提供足够的**调试工具和日志指标**，导致问题识别和修复变得困难，特别是对于那些没有丰富专业知识的人来说。<br>
5.未能充分利用PyTorch等框架的潜力：现有系统未能充分利用PyTorch等框架的全部潜力，错过了**错误修复、优化内核、新功能和编译器支持**。它们还依赖于外部依赖项，这些依赖项往往缺乏彻底测试，并且可能由于维护不足而变得过时或不兼容。<br>

**根本原因: 缺乏表达力强的张量抽象**。分布式系统不可组合性和不灵活性的根本原因在于没有将表达力强的张量和设备抽象作为核心组件来使用，而所有的分布式并行性、检查点以及效率优化都应该建立在这个基础之上。<br>

**设计原则：统一的分布式张量和设备抽象作为构建模块**。统一的设备抽象**将分布式系统表示为多维数组**，其中**每个维度对应一种并行技术**，负责管理设备间的通信和处理集体进程组(collective process groups)。一个互补的(complementary)张量抽象使得张量能够在这个数组上进行分片，保持分片规范(maintaining sharding specification)并支持自动分片传播。这些抽象共同实现了并行技术的无缝组合，确保了正确的语义，并促进了分布式操作中集体的调度。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们通过采用PyTorch的分布式张量(DTensor)和设备网格(DeviceMesh)[Wanchao Liang, 2023](https://github.com/pytorch/pytorch/issues/88838.)作为TorchTitan的基础组件，解决了(address)统一张量抽象的技术挑战。在与DTensor和DeviecMesh的合作中，我们发现了关键限制并进行了解决。通过**使用和扩展DTensor，我们开发了TorchTitan**，这是一个生产就绪(production-ready)的系统，能够在分布式训练中实现可组合性(composability)、模块化(modularity)、灵活性(flexibility)和可扩展性(extensibility)。TorchTitan支持3D并行性的组合(composition)、训练优化、可扩展的分布式检查点，并充分利用(harnesses)了PyTorch生态系统的全部优势。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了开发和评估TorchTitan的能力，我们采取了几个关键步骤，这些步骤代表了本工作的核心贡献，总结如下：<br>
1. 我们通过扩展DTensor的分片(sharding)功能以支持n维并行性，增加了与torch.compile的兼容性以实现编译器优化，并通过状态字典支持实现了n维模型的高效检查点功能，从而推进了DTensor的发展。我们还解决了关键错误，以增强DTensor的生产就绪性。
2. 我们展示了如何组合和堆叠(stack)各种并行技术，从而便于在大型语言模型训练中探索(facilitatating)多维并行性（第2.1节）。<br>
3. 我们实现了创新(novel)的软硬件协同设计方案，利用高级硬件特性提高GPU效率，提供**可定制的激活检查点策略**以平衡内存与计算之间的权衡，并利用**torch.compile进一步优化内存、计算和通信**（第2.2节）。<br>
4. 我们通过引入可扩展且高效的**分布式检查点功能**来支持快速故障恢复，集成如Flight Recorder等调试工具来调试崩溃或卡住的work，并提供广泛的日志指标，从而提供了生产级别的训练环境（第2.3节）。<br>
5. 我们在Llama 3.1系列模型（分别为8B、70B和405B，采用1D、2D和3D并行性）上，对TorchTitan进行了广泛的评估，评估规模从8个到512个GPU不等，以展示其在确保效率、收敛性和准确性的同时所具备的弹性扩展能力。总的来说，在最新NVIDIA H100 GPU上，与优化后的基线相比，我们在128个GPU规模下（Llama 3.1 8B）使用1D并行性实现了65.08%的训练加速，在256个GPU规模下（Llama 3.1 70B）使用2D并行性实现了额外的12.59%加速，在512个GPU规模下（Llama 3.1 405B）使用3D并行性实现了额外的30%加速（第3.2节）。<br>
6. 我们提供了系统的训练配方和指南，帮助用户应对分布式训练的复杂性，助力他们为各种模型大小和集群配置优化训练效率（第3.3节）。<br>
7. 我们展示了我们的模块化和可扩展架构如何允许新技术、优化和硬件的无缝集成和比较，从而确保对不断发展的机器学习领域的适应性（第4节）。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过提供一个易用且可扩展的平台，TorchTitan使大型语言模型（LLM）的预训练更加普及，使更广泛的研究人员和开发人员能够挖掘LLM的潜力，并加速该领域的创新。<br>

# 2 通过可组合性实现弹性(Elastic through Composability)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TorchTitan以模块化的方式融入了多种并行技术，使用户能够轻松选择多维并行性的组合。这种可组合性通过提高前沿探索的便捷性，解决了复杂的扩展挑战，从而优化了大规模训练的效率。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TorchTitan的代码库组织得很有目的性，以实现可组合性和可扩展性。我们有意将三个主要组件分开，并尽可能使它们相互独立：
1. 模型定义，它与并行性无关，且设计易于阅读；
2. 并行辅助工具(parallelism helpers)，它将数据并行、张量并行和流水线并行应用于特定模型；
3. 一个通用的训练循环(training loop)。所有这些组件都可以通过TOML文件进行配置，并允许通过命令行进行覆盖，而且基于现有的代码库，很容易添加新的模型和并行技术。<br>

## 2.1 可组合N维并行训练
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在本节中，我们将逐步介绍在大型集群上扩展模型训练的整个过程(entire regime)，包括元设备初始化和核心的可组合多维并行性，以展示在TorchTitan中如何组合这些技术来高效地训练规模不断增大(increasing scale)的大型语言模型（LLM）。TorchTitan中相应的实际代码片段可以在附录A中找到。<br>

### 2.1.1 使用元设备进行大规模模型初始化
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;鉴于大型语言模型（LLM）的尺寸呈指数级增长，实际训练开始之前就出现了第一个扩展问题。这就是需要在集群中分片部署一个大型模型，同时又不溢出(overflowing)CPU或GPU内存的需求。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了解决(tackle)这一问题，我们在TorchTitan中为模型启用了元设备初始化(meta device initialization)功能，即模型首先在一个“meta”设备类型上进行初始化。meta device tensor仅持有元数据信息，而非实际数据，这使得初始化过程极快。之后，我们执行模型分片，并将模型参数转换为分布式张量（DTensors），其中每个参数都持有一个位于元设备上的本地分片。最后，我们根据用户定义的初始化函数来初始化参数。_我们利用(leverage)分布式张量(Distributed Tensor)来正确同步随机数生成器(RNG)的种子，并根据参数的分片布局(sharding layouts)来初始化它们_。这**确保了参数在开始时具有与在整个模型在一个设备上初始化后分片前相同的值**，从而便于比较不同并行配置下的收敛情况。

### 2.1.2 完全分片数据并行
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;原始的完全分片数据并行([Fully Sharded Data Parallel，FSDP](https://dl.acm.org/doi/10.14778/3611540.3611569))（Zhao等人，2023）是ZeRO的一种有效实现，它提供了在PyTorch中进行大型模型训练的能力。然而，由于其在PyTorch中的FlatParameter实现（详见附录B.1），原始实现（FSDP1）存在诸多限制。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;鉴于这些限制，TorchTitan集成了完全分片数据并行的新版本（FSDP2）。FSDP2采用per-parameter的分布式张量分片表示，因此**与模型并行技术(model parallelism)和其他需要操作单个(individual)参数的功能具有更好的可组合性**。TorchTitan集成了FSDP2并将其作为默认的1D并行性加以利用，从而受益于改进的内存管理（与FSDP1相比，每个GPU的内存需求通常降低7%）和轻微的性能提升（与FSDP1相比，平均提升1.5%）。关于FSDP2的更多细节和使用示例，请参见附录B.1。TorchTitan通过嵌入适当的默认设置(包括根据您的world size自动进行分片(auto-sharding))来简化与FSDP2的运行。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了扩展(scaling)到更大的world sizes，TorchTitan还集成了**混合分片数据并行(Hybrid Sharded Data Parallel，HSDP)**, 它通过 _创建分片组_ 来扩展FSDP2。详细信息请参见附录B.2。<br>

### 2.1.3 张量并行
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;张量并行([Tensor Parallel，TP](https://dl.acm.org/doi/10.1145/3458817.3476209))（Narayanan等人，2021）与序列并行（[Sequence Parallel，SP](https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf)）（Korthikanti等人，2023）是实现大规模大型模型训练的关键模型并行技术。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在TorchTitan中，张量并行（TP）是通过PyTorch的 _RowwiseParallel_ 和 _ColwiseParallel_ API实现的，其中模型参数被分割为DTensors，并与其进行sharded computation。通过利用(leveraging) DTensor，TP的实现**无需修改模型代码**，这使得 _在不同模型上能够更快地启用_ ，并且与本文中提到的其他功能具有更好的组合性(composability)。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;张量并行（TP）与序列并行（SP）：张量并行负责将计算量最大的部分进行分割，而序列并行（SP）则在序列维度上对归一化或dropout layers执行分片计算。如果不进行分片，这些层会生成大量重复的激活张量，从而可能超出每个GPU的内存限制。有关TP和FSDP+TP的更多详细信息、图示及用法，请参阅附录B.3。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由于张量并行（TP）和序列并行（SP）之间的协同关系，TorchTitan原生地将这两者结合在一起，并且它们共同由TP度数设置来控制。<br>


