# Scalable Diffusion Models with Transformers

# 摘要
&nsps;&nsps;&nsps;&nsps;&nsps;&nsps;&nsps;&nsps;我们探索了一种基于Transformer架构的新型扩散模型(diffusion model)。我们训练了图像的潜在扩散模型，将常用的U-Net骨干网络替换为在潜在patch上操作的Transformer。我们通过Gflops衡量前向传递复杂度，分析了我们的Diffusion Transformers (DiTs)的可扩展性。我们发现，具有更高Gflops的DiTs（通过增加Transformer的深度/宽度或增加输入令牌的数量）始终具有较低的FID。除了具有良好的可扩展性属性外，我们最大的DiT-XL/2模型在基于类条件的ImageNet 512×512和256×256基准测试中胜过了所有以前的扩散模型，后者在256×256基准测试上达到了2.27的最新FID。 <br>



# reference

![论文地址](https://arxiv.org/pdf/2212.09748.pdf)
