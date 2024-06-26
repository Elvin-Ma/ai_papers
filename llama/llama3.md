# llam3
## 1 模型架构
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Llama 3 采用一个相对标准的Decoder-only的Transformer架构。与 Llama 2 相比，进行了几个关键改进：
- 采用词汇量为 128K 的分词器（Tokenizer），能更有效地编码语言，大大提高了模型性能；
- 为提高推理效率，采用了分组查询注意力 (GQA，示意图见上图)；
- 使用掩码来确保自注意力不会跨越文档边界，在长度为 8,192 个词元（token）的序列上训练模型。

## 2 训练数据（Training Data）
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Llama 3 在超过 **15T个token**上进行预训练，这些数据都是从公开可用的来源收集而来的。这个训练数据集比用于 Llama 2 的数据集大**七倍**，其中包括四倍于以往的code数据。同时，预训练数据集包含超过 5% 的高质量**非英语数据组成**，涵盖了超过 30 种语言。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为确保在最高质量的数据上进行训练，作者开发了一系列数据过滤流程。包括使用启发式过滤器、不适宜工作场合的内容过滤器、语义去重方法和文本分类器来预测数据质量。作者发现以往版本的 Llama 在识别高质量数据方面出奇地优秀，因此使用 Llama 2 生成用于文本质量分类器的训练数据。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;作者还进行了大量实验，评估了在最终的预训练数据集中混合来自不同来源的数据的最佳方法。这些实验使作者能够选择一种**数据混合方式**，确保 Llama 3 在各种用例中都表现出色，包括日常问题、STEM、编码、历史知识等。<br>

## 3 扩大预训练规模（Scaling up Pretraining）
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在开发 Llama 3 的过程中，作者做出了几项新的扩展行为观察。例如，虽然对于 8B 参数的最佳训练计算量对应约为200B词元（参见上图，数据来自 [Chinchilla论文](https://link.juejin.cn/?target=https%3A%2F%2Farxiv.org%2Fpdf%2F2203.15556.pdf), 但作者发现，即使在模型训练了两个数量级更多的数据后，模型性能仍然会继续提升。8B和70B参数模型在进行高达 15T 词元的训练后，继续呈现对数线性的改善。更大的模型可以以更少的训练计算量达到与这些较小模型相匹配的性能，但通常更喜欢较小的模型，因为它们在推理过程中更加高效。<br>




# 参考文档
- [blog](https://kili-technology.com/large-language-models-llms/llama-3-guide-everything-you-need-to-know-about-meta-s-new-model-and-its-data)
