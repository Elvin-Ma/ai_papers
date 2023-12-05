# KV Cache

# 1 概述
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在生成式Transformer中，缓存(Caching) Key(K)和 Value(V)状态的技术已经存在一段时间了。这种技术可以显著提高推理速度，在注意力机制中，Key和Value状态用于计算带缩放的点积注意力机制(scaled dot-product attention)，如下图所示。<br>

![figure1](images/kv-cache-figure0.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;KV Cache发生在多个tokens生成步骤中，只在Decoder中进行（即在仅解码器的模型如GPT中，或者在编码器-解码器模型如T5中的解码器部分）。像BERT这样的模型不是生成模型，因此没有KV Cache。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;解码器以自回归(auto-regressive)的方式工作，就像下图GPT-2文本生成示例所示的那样。<br>

![figure1](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*sexO6adGhaKr7aH0.gif)

*(figrue 1: 在Encoder的自回归生成中，给定一个输入，模型会预测下一个token，然后在下一步中使用组合的输入进行下一个预测。)* <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这种自回归行为会重复(repeats)一些操作，我们可以通过放大(zoom in) Encoder 中计算的带掩码的缩放点积注意力(masked scaled dot-product attention)来更好地理解这一点。<br>

![figure2](images/kv-cache-gif1.gif)
*(解码器中缩放点积注意力的逐步可视化。emb_size表示embedding size.)* <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由于解码器是因果的（即令牌的注意力仅依赖于其前面的令牌），在每个生成步骤中，我们重新计算了相同的先前令牌的注意力，而实际上我们只想计算新令牌的注意力。<br>

# 2 KV Cache
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这就是KV缓存发挥作用的地方。通过缓存先前的键(Key)和值(Value)，我们可以只专注于计算新token的注意力。<br>
![figure2](images/kv-cache-gif2.gif)

*(缩放点积注意力的比较，带有和不带有KV缓存。emb_size表示嵌入大小。图片由作者创建。)*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这种优化为什么重要呢？如上图所示，使用KV缓存得到的矩阵要小得多，这导致矩阵乘法更快。唯一的缺点是它需要更多的GPU VRAM（或者如果没有使用GPU，则需要更多的CPU RAM）来缓存键(Key)和值(Value)的状态。
# 5 参考链接
[参考链接1](https://jalammar.github.io/illustrated-gpt2/)

