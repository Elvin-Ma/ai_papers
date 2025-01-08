# 0 摘要
- [论文地址](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们推出了DeepSeek-V3，这是一个强大的混合专家（MoE）语言模型，总参数量达到6710亿，每个词汇激活370亿参数。为了实现高效推理和成本效益高的训练，DeepSeek-V3采用了多头潜在注意力（MLA）和DeepSeekMoE架构，这些架构在DeepSeek-V2中已得到了充分验证。此外，DeepSeek-V3开创性地采用了无辅助损失的负载均衡策略，并设定了多词汇预测训练目标，以实现更强的性能。我们在14.8万亿个多样且高质量的词汇上对DeepSeek-V3进行了预训练，随后进行了监督微调和强化学习阶段，以充分发挥其能力。综合评估显示，DeepSeek-V3的性能超越了其他开源模型，并达到了与领先的闭源模型相当的水平。尽管性能卓越，但DeepSeek-V3的完整训练仅需278.8万H800 GPU小时。此外，其训练过程非常稳定。在整个训练过程中，我们没有遇到任何不可恢复的损失尖峰，也没有进行任何回滚。模型检查点可在https://github.com/deepseek-ai/DeepSeek-V3上获取。<br>

