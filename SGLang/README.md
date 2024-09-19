# SGLang: Efficient Execution of Structured Language Model Programs
- [论文链接](https://arxiv.org/pdf/2312.07104)

# 3 使用 RadixAttention 实现高效的 KV 缓存重用
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SGLang 程序可以通过 "fork" 原语链接多个生成调用并创建并行副本。此外，不同的程序实例通常共享一些公共部分（例如系统提示）。在执行过程中，这些情况会导致许多共享的提示前缀，从而为重用 KV Cache创造了许多机会。在 LLM 推理过程中，KV 缓存存储来自前向传递的中间张量，用于解码未来的标记。它们以自注意力机制中的键值对命名 [51]。KV 缓存的计算仅取决于前缀标记。因此，具有相同提示前缀的请求可以重用 KV 缓存，减少冗余计算和内存使用。附录 A 中提供了更多背景信息和一些示例。<br>
