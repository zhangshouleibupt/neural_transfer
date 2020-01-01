### 神经风格转换

摘要：本博客为torch官方教程的实现以及论文<A Neural Algorithm of Artisitc Style>的解读

### 基本的想法

想法：如何实现两幅图片的融合呢？这里的融合是指：假如我们有两幅图片，如果我们想将其中一幅的风格融合到另外一幅图片上，当然我们希望的是生成的图片满足以下条件：

- 风格上尽量接近目标风格图片
- 内容上尽量接近目标内容（提供的欲修改图片）图片



参考文献：

[A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)

[NEURAL TRANSFER USING PYTORCH](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)