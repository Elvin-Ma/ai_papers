# LLaMA：开放高效的基础语言模型

# 摘要
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们介绍了LLaMA，一系列参数从**7B到65B**的基础语言模型。我们在数**万亿**个tokens上训练我们的模型，并展示了只使用公开可用的数据集进行训练的可能性，而不依赖专有和不可访问的数据集。特别是，LLaMA-13B在大多数基准测试中优于GPT-3（175B），而LLaMA-65B与最佳模型Chinchilla-70B和PaLM-540B相媲美。我们将所有模型发布给研究社区[社区链接](https://github.com/facebookresearch/llama)。<br>

# 1 引言
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;于大规模语料(corpora)库训练的大型语言模型（LLM）展示了它们从文本指令或少量示例中执行新任务的能力（Brown等，2020）。这些少样本特性首次出现在将模型扩展到足够大的规模时（Kaplan等，2020），从而衍生出一系列工作，专注于进一步**扩展**这些模型（Chowdhery等，2022；Rae等，2021）。这些努力基于一个假设，即更多的参数会带来更好的性能。**然而**，Hoffmann等（2022）的最新研究表明，在给定的计算预算下，最佳性能并不是由最大的模型实现的，而是由在**更多数据**上训练的较小模型实现的。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Hoffmann等人（2022）提出的扩展定律的目标是确定如何在**特定的训练计算预算下**最佳地扩展数据集和模型大小。然而，这个目标忽视了推理计算预算，在大规模使用语言模型时变得至关重要。在这个背景下，考虑到目标性能水平，首选的模型不是训练速度最快的模型，而是**推理速度最快**的模型，尽管训练一个大模型达到一定水平的成本可能更低，但**在实际使用时推理速度的快慢才是关键**。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本研究的重点是训练一系列语言模型，在**不同的推理计算预算下**实现最佳性能，通过使用比通常使用的tokens更多的tokens进行训练。结果产生的模型称为LLaMA，参数范围从7B到65B，与最佳的现有LLM相比具有竞争力的性能。例如，LLaMA-13B在大多数基准测试中优于GPT-3，尽管体积小了10倍。我们相信，这个模型将有助于普及LLM的使用和研究，因为它可以在单个GPU上运行(inference)。在更高级别的规模上，我们的65B参数模型也与最佳的大型语言模型（如Chinchilla或PaLM-540B）相媲美。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;与Chinchilla、PaLM或GPT-3不同的是，我们只使用公开可用的数据，使我们的工作与开源兼容，而大多数现有模型则依赖于不公开可用或未记录的数据（例如，“Books - 2TB”或“Social media conversations”）。存在一些例外，特别是OPT（Zhang等，2022）、GPT-NeoX（Black等，2022）、BLOOM（Scao等，2022）和GLM（Zeng等，2022），但没有一个与PaLM-62B或Chinchilla相竞争。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在本文的剩余部分，我们将概述我们对Transformer架构（Vaswani等，2017）所做的修改，以及我们的训练方法。然后，我们将报告我们模型的性能，并与其他LLM在一组标准基准上进行比较。最后，我们将使用来自负责任的AI社区的最新基准之一，揭示我们模型中编码的一些偏见和有害性。<br>

# 2 方法
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们的训练方法类似于先前研究中描述的方法（[Brown等，2020；Chowdhery等，2022](https://doi.org/10.48550/ARXIV.2005.14165)），并受到Chinchilla的扩展定律的启发（Hoffmann等，2022）。我们使用标准优化器在大量文本数据上训练大型Transformer模型。

## 2.1 预训练数据

![table1](images/llama-table1.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们的训练数据集是由多个来源的混合组成，如表1所示，涵盖了多个领域。在很大程度上，我们重复使用了已经用于训练其他LLM的数据源，但限制只使用公开可用且适用于开源的数据。这导致了以下数据混合以及它们在训练集中所占比例：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;英文Common Crawl [67%]。我们使用CCNet流水线（Wenzek等，2020）对2017年至2020年的五个CommonCrawl数据集进行预处理。该过程在行级别进行数据去重，使用fastText线性分类器进行语言识别以删除非英语页面，并使用ngram语言模型过滤低质量内容。此外，我们训练了一个线性模型来将页面分类为维基百科引用页面和随机抽样页面，并丢弃未被分类为引用的页面。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;C4 [15%]。在探索性实验中，我们观察到使用多样的预处理的CommonCrawl数据集可以提高性能。因此，我们将公开可用的C4数据集（Raffel等，2020）包含在我们的数据中。C4的预处理也包括去重和语言识别步骤：与CCNet的主要区别在于质量过滤，这主要依赖于标点符号的存在或网页中的单词和句子数量等启发式方法。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Github [4.5%]。我们使用Google BigQuery上公开可用的GitHub数据集。我们只保留在Apache、BSD和MIT许可下分发的项目。此外，我们使用基于行长度或包含字母数字字符比例的启发式方法过滤低质量文件，并使用正则表达式删除头部等模板内容。最后，我们使用精确匹配在文件级别进行数据去重。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;维基百科 [4.5%]。我们添加了涵盖20种语言的维基百科转储，时间跨度为2022年6月至8月，这些语言使用拉丁或西里尔字母脚本: bg、ca、cs、da、de、en、es、fr、hr、hu、it、nl、pl、pt、ro、ru、sl、sr、sv、uk。我们对数据进行处理，删除超链接、注释和其他格式化模板内容。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;古腾堡计划和Books3 [4.5%]。我们在训练数据集中包括两个**图书语料库**：古腾堡计划，其中包含公共领域的图书，以及ThePile的Books3部分（Gao等，2020），这是一个用于训练大型语言模型的公开可用数据集。我们在图书级别进行去重，删除内容重叠超过90%的图书。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ArXiv [2.5%]。我们处理arXiv的LaTeX文件，以将科学数据添加到我们的数据集中。根据Lewkowycz等人（2022）的方法，我们删除第一节之前的所有内容以及参考文献部分。我们还从.tex文件中删除注释，并对用户编写的内联扩展定义和宏进行了处理，以增加论文之间的一致性。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Stack Exchange [2%]。我们包括Stack Exchange的一个转储，这是一个包含高质量问题和答案的网站，涵盖了从计算机科学到化学等多个领域。我们保留了28个最大网站的数据，从文本中删除了HTML标签，并按得分（从高到低）对答案进行了排序。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;分词器(tokenizer)。我们使用字节对编码（BPE）算法（Sennrich等，2015）对数据进行分词，使用了SentencePiece（Kudo和Richardson，2018）的实现。值得注意的是，我们将所有数字拆分为单个数字，并在无法识别的UTF-8字符上使用字节进行分解。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;总体而言，我们整个训练数据集在分词后大约包含**1.4T个tokens**。对于我们的大部分训练数据，每个token在训练过程中只使用一次，但维基百科和图书领域是个例外，我们在这两个领域上进行了大约两个epoch的训练。<br>

## 2.2 架构
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在大型语言模型的最新工作中，我们的网络基于Transformer架构（Vaswani等，2017）。我们利用了随后提出的各种改进，这些改进在不同的模型中得到了应用，比如PaLM。以下是与原始架构的主要区别，以及我们从哪里得到了这种变化的灵感（括号中）：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**预归一化(Pre-normalization) [GPT3]**。为了改善训练的稳定性，我们对每个Transformer子层的**输入进行归一化**，而不是对输出进行归一化。我们使用了**RMSNorm归一化函数**，由Zhang和Sennrich（2019）引入。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**SwiGLU激活函数 [PaLM]**。我们用SwiGLU激活函数代替ReLU非线性函数，该函数由Shazeer（2020）引入，以提高性能。我们使用的维度是 $\frac{2}{3} 4d$ 而不是PaLM中的4d。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**旋转嵌入 [GPTNeo]**。我们移除了绝对位置嵌入，而是在网络的每一层添加了**旋转位置嵌入**（RoPE），该嵌入由Su等人（2021）引入。我们不同模型的超参数细节见表2。<br>

![table2](images/llama-table2.jpg)

## 2.3 优化器
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们使用AdamW优化器（Loshchilov和Hutter，2017）对模型进行训练，具体的超参数如下：β1 = 0.9，β2 = 0.95。我们采用余弦学习率调度，使得最终学习率等于最大学习率的10%。我们使用0.1的权重衰减和1.0的梯度裁剪。我们使用2,000个预热步骤，并根据模型的大小调整学习率和批量大小（详见表2）。<br>

## 2.4 高效实现
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了提高模型的训练速度，我们进行了几项优化。首先，我们使用了有效的**因果多头注意力的实现**，以减少内存使用和运行时间。这个实现在xformers库中可用，受到了[Rabe和Staats(2021)](https://arxiv.org/pdf/2112.05682.pdf)的启发，并使用了Dao等人（2022）的反向传播方法。这是通过不存储注意力权重和不计算由于语言建模任务的因果性质而被屏蔽的键/查询分数来实现的。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了进一步提高训练效率，我们通过检查点技术减少了在反向传播过程中需要重新计算的激活值数量。具体来说，我们保存了计算成本较高的激活值，例如线性层的输出。这是通过手动实现Transformer层的反向函数来实现的，而不是依赖于PyTorch的自动求导。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了充分受益于这个优化，我们需要通过使用模型和序列并行化来减少模型的内存使用，如[Korthikanti等人（2022)](https://arxiv.org/abs/2205.05198)所描述的。此外，我们还尽可能地重叠激活值的计算和GPU之间的网络通信（由于all_reduce操作）。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当训练一个拥有650亿参数的模型时，我们的代码在拥有80GB RAM的2048个A100 GPU上每秒处理大约380个tokens。这意味着在包含1.4T个标记的数据集上训练大约需要21天的时间。<br>













