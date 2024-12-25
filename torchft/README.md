# torchft – Fault Tolerance DDP/HSDP in OSS

# 1 背景
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这是torchft的设计文档，torchft是PyTorch开源社区（OSS）中实现的一种每步容错（per step fault tolerance）机制，其设计重点在于灵活性和与现有模型及库的兼容性。在最近的2024年PyTorch大会上，我们的实验性容错高可扩展性分布式处理（HSDP，Highly Scalable Distributed Processing）项目引起了广泛关注。许多不同公司都表示对使用类似机制非常感兴趣，因此，这是一份全新的、仅针对开源社区的设计和实现方案。<br>


