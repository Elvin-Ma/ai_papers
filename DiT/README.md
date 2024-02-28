# Scalable Diffusion Models with Transformers

# 摘要
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们探索了一种基于Transformer架构的新型扩散模型(diffusion model)。我们训练了图像的潜在扩散模型，将常用的U-Net骨干网络替换为在潜在patch上操作的Transformer。我们通过Gflops衡量前向传递复杂度，分析了我们的Diffusion Transformers (DiTs)的可扩展性。我们发现，具有更高Gflops的DiTs（通过增加Transformer的深度/宽度或增加输入令牌的数量）始终具有较低的FID。除了具有良好的可扩展性属性外，我们最大的DiT-XL/2模型在基于类条件的ImageNet 512×512和256×256基准测试中胜过了所有以前的扩散模型，后者在256×256基准测试上达到了2.27的最新FID。 <br>

# 1 引言
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;机器学习正在经历由Transformer驱动的复兴。在过去的五年中，用于自然语言处理 [8, 42]、视觉 [10]和其他几个领域的神经网络架构主要被Transformer所取代 [60]。然而，许多种类的图像级生成模型仍然没有采用这一趋势。尽管Transformer在自回归模型 [3,6,43,47]中得到广泛应用，但在其他生成建模框架中的应用较少。例如，扩散模型(diffusion models)一直是图像级生成模型最新进展的前沿 [9,46]，然而它们都采用了卷积U-Net架构作为事实上的骨干选择。<br>

*注释：扩散模型是一种生成模型，用于生成高质量的图像。它通过在一系列迭代的步骤中逐渐将图像从一个噪声图像转变为目标图像。在每个步骤中，模型通过引入噪声来模糊图像，然后尝试通过迭代过程逐渐减小噪声并还原图像的细节。这种逐渐减小噪声的过程被称为扩散，因此这种模型被称为扩散模型。* <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ho等人的开创性工作 [DDPM](https://arxiv.org/pdf/2006.11239.pdf) 首次引入了U-Net骨干网络用于扩散模型。最初在像素级自回归模型和条件生成对抗网络（GAN）[23]中取得成功后，U-Net从PixelCNN++ [52, 58] 中继承并进行了一些修改。该模型是卷积模型，主要由ResNet [15] 块组成。与标准的U-Net [49] 相比，额外的**空间自注意力块（在Transformer中是重要组件）被插入到较低分辨率的位置**。Dhariwal和Nichol [9] 对UNet的几个架构选择进行了剖析，例如使用自适应归一化层 [40] 注入条件信息和卷积层的通道数。然而，Ho等人提出的U-Net的高层设计在很大程度上保持不变。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过这项工作，我们旨在揭示扩散模型中架构选择的重要性，并为未来的生成建模研究提供经验基准。我们展示了U-Net的**归纳偏差**对于扩散模型的性能并不关键，可以轻松地用标准设计(如Transformer)进行替换。因此，扩散模型有望从最近的**架构统一趋势**中受益，例如通过继承其他领域的最佳实践和训练方法，同时保留可扩展性、稳健性和效率等有利属性。标准化的架构还将为跨领域研究开辟新的可能性。<br>

*注释：归纳偏差是指机器学习模型在学习过程中对数据的归纳和推理的偏好或倾向。它反映了模型在没有足够数据的情况下对数据的预先假设或先验知识的影响。* <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在本文中，我们专注于一种基于Transformer的新型扩散模型。我们将其称为Diffusion Transformers，简称为DiTs。DiTs遵循了(adhere)视觉Transformer([ViTs](https://arxiv.org/pdf/2010.11929.pdf) )[10]的最佳实践，已经证明在视觉识别方面比传统的卷积网络（例如ResNet [15]）具有更好的可扩展性。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;具体而言，我们研究了Transformer在网络复杂度与样本质量之间的规模化行为。我们展示了通过在Latent Diffusion Models(LDMs : [stable-diffusion paper](https://arxiv.org/pdf/2112.10752.pdf))[48]框架下构建和评估DiT设计空间的方式，即在VAE的潜在空间中训练扩散模型，我们可以成功地用Transformer替代U-Net骨干网络。我们进一步展示了**DiT是适用于扩散模型的可扩展架构**：网络复杂度（以Gflops衡量）与样本质量（以FID衡量）之间存在着很强的相关性。通过简单地扩大DiT的规模并使用高容量骨干网络（118.6 Gflops）训练LDM，我们能够在条件为类别的256×256 ImageNet生成基准测试中实现2.27的FID，达到了最先进的结果。<br>

# 2 相关工作
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Transformer。Transformer [60] 已经在语言、视觉 [10]、强化学习(reinforcement learning) [5, 25] 和元学习 [39] 等领域取代了特定领域的架构。它们在**增加模型大小、训练计算资源和语言领域的数据方面展现出了显著的扩展性属性 [26]**，可以作为通用的自回归模型 [17] 和 ViTs [63]。除了语言领域，Transformer 已经被训练用于自回归地预测像素 [6, 7, 38]。它们还被用于discrete codebooks(离散的 编程书籍) [59] 的训练，既作为自回归模型 [11, 47]，也作为掩码生成模型 [4, 14]；前者在参数规模达到了200亿规模时展示出了出色的扩展行为 [62]。最后，Transformer 在 DDPMs 中被探索用于合成非空间数据，例如在 DALL·E 2 中生成 CLIP 图像嵌入 [41, 46]。**在本文中，我们研究了将 Transformer 作为图像扩散模型的骨干网络时的扩展性质。** <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**去噪扩散概率模型 (DDPMs)。** 扩散模型 [19, 54] 和基于分数的生成模型 [22, 56] 在图像生成方面取得了特别成功的结果 [35, 46, 48, 50]，在许多情况下超过了此前的最先进技术生成对抗网络 (GANs) [12]。过去两年中，DDPMs 的改进主要是通过改进的**采样技术** [19, 27, 55] 驱动的，尤其是**无分类器指导 [21]、重新定义扩散模型以预测噪声而不是像素 [19]，以及使用级联的 DDPM 管道**，其中低分辨率的基础扩散模型与上采样器并行训练 [9, 20]。对于上述所有扩散模型，**Conv U-Net [49] 是事实上的骨干架构选择**。同时进行的研究 [24] 提出了一种基于注意力机制的高效架构用于 DDPMs；**我们探索纯 Transformer 模型**。 <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**架构复杂度。** 在图像生成领域评估架构复杂度时，**通常使用参数数量**作为常见做法。一般来说，参数数量可能不足以准确反映图像模型的复杂性，因为它们未考虑到例如图像分辨率对性能的显著影响 [44, 45]。相反，本文中的大部分模型复杂性分析是通过理论上的Gflops进行的。这使我们与架构设计文献保持一致，其中**广泛使用Gflops来衡量复杂性**。在实践中，作为复杂性的黄金指标仍存在争议，因为它通常取决于具体的应用场景。Nichol和Dhariwal关于改进扩散模型的开创性工作 [9, 36] 与我们最相关，在那里，他们分析了U-Net架构类的可扩展性和Gflop特性。本文中，我们专注于Transformer类。<br>

*注释：Model Gflops = (模型浮点运算数 / 模型执行时间) x 10^9* <br>

# 3 扩散 Transformer
## 3.1 准备工作
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**扩散公式。** 在介绍我们的架构之前，我们简要回顾一些基本概念，以便了解扩散模型 (DDPMs) [19, 54]。高斯扩散模型假设存在一个前向加噪过程，逐渐将噪声应用于真实数据 $x_{0}: q(x_{t} \mid x_{0})= \mathcal{N} (x_{t} ; \sqrt{ \bar \alpha_{t}} x_{0}, (1-\bar \alpha_{t}) \mathbf{I})$ , 这里 $\bar \alpha_{t}$ 是超参数。通过应用重参数化技巧，我们可以进行采样 $x_{t}=\sqrt{\bar \alpha_{t}} x_{0}+\sqrt{1-\bar \alpha_{t}} \epsilon_{t}$ , 这里  $\epsilon_{t} \sim \mathcal{N}(0, \mathbf{I})$ 。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;扩散模型通过训练来学习逆前向过程的噪音： $p_{θ}(x_{t−1} | x_{t}) = \mathcal{N}(µ_{θ}(x_{t}), Σθ(x_{t}))$ ，其中神经网络用于预测 $p_{θ}$ 的统计信息。逆处理模型是使用 $P(x_{0})$ 的变分下界[30]训练的，该下界可简化为 $\mathcal L(θ) = -p(x0|x1) + Σ_{t} \mathcal D_{K L}(q*(x_{t−1} | x_{t}, x_{0}) || p_{θ}(x_{t−1} | x_{t}))$ ，其中排除了训练无关的额外项。 由于 $q*$ 和 $p_{θ}$ 都是高斯分布，可以使用两个分布的均值和协方差来评估 $\mathcal D_{KL}$ 。通过将 $µ_{θ}$ 重新参数化为噪声预测网络 $\varepsilon_{θ}$ ，可以使用预测的噪声 $\varepsilon_{θ}(x_{t})$ 和采样的高斯噪声 $\varepsilon_{t}$ 之间的简单均方误差来训练模型： $\mathcal L_{simple}(\theta)=||\epsilon_{\theta} (x_{t}) - \epsilon_{t}||_{2}^{2}$ 。为了使用学习的反向过程协方差 $\Sigma _{\theta}$ 来训练扩散模型，需要优化完整的 $ \mathcal D _{KL}$ 项。我们遵循Nichol和Dhariwal的方法[36]：使用 $\mathcal L _{simple}$ 训练 $\epsilon _{\theta}$ ,使用完整的 $\mathcal L$ 训练Σθ。一旦 $p _{θ}$ 训练完成，可以通过初始化 $x _{t _{max}} ∼ N(0, I)$ 并使用重参数化技巧从 $p _{θ}(x _{t−1} | x_{t})$ 中采样 $x _{t−1}$来生成新的图像。 <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;无分类器指导。条件扩散模型将额外的信息作为输入，例如类别标签c。在这种情况下，反向过程变为 $pθ(x_{t−1} |x_{t}, c)$ ，其中 $\epsilon_{θ}$ 和 $Σ_{θ}$ 以c为条件。在这种设置下，可以使用无分类器指导来鼓励采样过程找到 $log p(c|x)$ 较高的x。根据贝叶斯法则: $\log p(c \mid x) \propto \log p(x \mid c)-\log p(x)$ , 因此 $\nabla_{x} \log p(c \mid x) \propto \nabla_{x} \log p(x \mid c)-\nabla_{x} \log p(x)$ . 通过将扩散模型的输出解释为得分函数，可以通过以下方式来引导DDPM采样过程以高概率p(x|c)采样x: $\hat{\epsilon}_{\theta} (x_{t}, c) = \epsilon_{\theta}(x_{t}, \emptyset) + s \cdot \nabla_{x} \log p(x \mid c) \propto \epsilon_{\theta}(x_{t}, \emptyset) + s \cdot(\epsilon_{\theta}(x_{t}, c)-\epsilon_{\theta}(x_{t}, \emptyset))  $ ，其中s > 1表示指导的尺度（注意，s = 1表示恢复标准采样）。<br>








# reference

![论文地址](https://arxiv.org/pdf/2212.09748.pdf)
