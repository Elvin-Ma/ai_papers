# KV Cache

# 1 概述
在生成式Transformer中，缓存(Caching) Key(K)和 Value(V)状态的技术已经存在一段时间了。这种技术可以显著提高推理速度，在注意力机制中，Key和Value状态用于计算带缩放的点积注意力机制(scaled dot-product attention)，如下图所示。

![figure1](images/kv-cache-figure1.jpg)

KV Cache发生在多个tokens生成步骤中，只在Decoder中进行（即在仅解码器的模型如GPT中，或者在编码器-解码器模型如T5中的解码器部分）。像BERT这样的模型不是生成模型，因此没有KV Cache。<br>



