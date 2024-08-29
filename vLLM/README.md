# Efficient Memory Management for Large Language Model Serving with PagedAttention
- [论文链接](https://arxiv.org/pdf/2309.06180)

# 0 摘要
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对大型语言模型（LLMs）进行高吞吐量的服务需要一次性批量处理足够多的请求。然而，现有系统存在困难，因为每个请求的键-值缓存（KV缓存）内存很大并且会动态增长和收缩。当管理效率低下时，这种内存可能会因碎片化和冗余复制而被显著浪费，从而限制批处理大小。为了解决这个问题，我们提出了PagedAttention，这是一种受**传统虚拟内存和分页技术**启发的注意力算法，类似于操作系统中的技术。在此基础上，我们构建了vLLM，一个LLM服务系统，实现了（1）在KV缓存内存中几乎零浪费和（2）在请求内部和跨请求之间灵活共享KV缓存，以进一步减少内存使用。我们的评估显示，与现有的state-of-the-art系统（如FasterTransformer和Orca）相比，vLLM将流行的LLMs的吞吐量提高了2-4倍，并且具有相同水平的延迟。这种改进在序列更长、模型更大和更复杂的解码算法下更加明显。vLLM的源代码可以在https://github.com/vllm-project/vllm 公开获取。<br>

# 1 


# 2 

## 2.2 LLM服务与自回归生成
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一旦训练完成，LLMs通常作为**条件生成服务**而部署（例如，完成API [34]或聊天机器人 [19, 35]）。向LLM服务发送请求时，会提供一组输入提示标记（𝑥1, . . . , 𝑥𝑛），LLM服务根据方程式1生成一组输出标记（𝑥𝑛+1, . . . , 𝑥𝑛+𝑇）。我们将提示和输出列表的串联称为序列。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由于方程式1的分解，LLM只能**逐个样本化和生成新的标记**，每个新标记的生成过程取决于该序列中所有先前标记，特别是它们的key和value向量。在这个顺序生成过程中，现有标记的键和值向量通常被缓存以生成未来的标记，称为KV缓存。请注意，**一个标记的KV缓存取决于其所有先前标记**。这意味着出现在序列中不同位置的相同标记的KV缓存将不同。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;考虑到一个请求提示，LLM服务中的生成计算可以分解为两个阶段：<br>



# 5 参考链接
- [csdn blog](https://blog.csdn.net/yjw123456/article/details/141090361)
- [vllm blog](https://blog.vllm.ai/2023/06/20/vllm.html)
