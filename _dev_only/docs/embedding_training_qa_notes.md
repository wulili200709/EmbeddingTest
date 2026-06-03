# Embedding 小样本注册与 CNN 训练问答整理

这份文档整理的是 AI Local System / Embedding 小样本注册方案中，围绕“CNN 是否训练、embedding 从哪一层来、GAP/L2 是什么、OK/NG 原型怎么判定、特征分析图怎么理解”等问题的汇报口径。

## 1. 这个系统有没有重新训练 CNN？

严格来说，**没有重新训练 CNN**。

系统使用的是 `torchvision` 官方预训练模型权重，例如：

- `EfficientNet-B0`
- `MobileNetV3-small`
- `MobileNetV3-large`

这些模型已经在 ImageNet 等大规模数据集上完成预训练。现场使用时，系统不更新模型权重，也不微调分类头，只把它们作为固定的特征提取器使用。

更准确的说法是：

> 本系统中的“训练 / 学习”不是深度学习网络训练，而是基于预训练 CNN embedding 的小样本注册。

也就是说：

```text
CNN 不是被现场训练的对象
CNN 是固定的通用视觉特征提取器
```

## 2. 系统里的“训练”到底是哪一步？

界面中点击“学习 / 训练”时，实际做的是当前产品的 OK/NG 特征注册。

流程如下：

```text
添加 OK/NG 样本
      ↓
选择 ROI
      ↓
点击学习 / 训练
      ↓
用官方预训练 CNN 提取每张样本的 embedding
      ↓
分别计算 OK 原型、NG 原型
      ↓
保存 OK/NG 特征库、原型、margin、topk 等参数
```

所以，这里的训练产物不是一个重新训练后的 CNN 模型，而是：

- `ok_bank`：OK 样本 embedding 特征库
- `ng_bank`：NG 样本 embedding 特征库
- `ok_proto`：OK 原型
- `ng_proto`：NG 原型
- `margin / topk / score_mode` 等判定参数

一句话总结：

> 系统中的训练是“样本注册训练”，不是“CNN 网络训练”；它训练的是当前产品的 OK/NG 特征原型和判定规则。

## 3. 用的是官方模型权重吗？

是的。

系统使用的是官方预训练 CNN 权重，只把它当作固定特征提取器来推理。

```text
官方预训练模型权重
        ↓
固定不训练
        ↓
输入当前产品 ROI 图像
        ↓
输出 CNN features
        ↓
GAP + L2
        ↓
得到 embedding 特征
        ↓
用于 OK/NG 相似度注册和判断
```

现场注册时：

```text
不更新模型权重
不训练 CNN
只保存当前产品 OK/NG 样本对应的 embedding 特征库和原型
```

汇报时可以这样说：

> 本系统使用官方预训练 CNN 权重作为通用视觉特征提取器。现场小样本注册阶段只进行前向推理，提取当前产品 ROI 的 embedding 特征，并建立 OK/NG 特征原型；不会重新训练或微调 CNN 权重。

## 4. `model.features` 应该怎么介绍？

`model.features` 可以理解成 CNN 里专门负责“看图、提特征”的部分。

完整的分类模型一般分两段：

```text
CNN 分类模型
= features 特征提取器 + classifier 分类头
```

在 ImageNet 预训练模型里：

```text
features：提取边缘、纹理、形状、局部结构等视觉特征
classifier：把这些特征分类成猫、狗、车等 ImageNet 类别
```

本系统只用前半段：

```text
ROI 图像
  ↓
model.features
  ↓
最后卷积特征图
  ↓
GAP + L2
  ↓
embedding
```

可以这样对别人解释：

> 我们没有使用 CNN 最后的 ImageNet 分类结果，而是只保留模型前面的 `features` 特征提取部分。`model.features` 会把 ROI 图像转换成一组通用视觉特征，比如边缘、纹理、形状等。然后通过 GAP 和 L2 归一化，把这些特征图变成一个固定长度的 embedding 向量，用来和 OK/NG 样本做相似度比较。

更口语化一点：

> `model.features` 就像 CNN 的“视觉感知器”，负责把图片变成一串能代表图片外观的数字。我们不要它最后判断“猫狗车”的分类头，只要这串视觉特征数字，也就是 embedding。

## 5. embedding 来自 CNN 的哪一层？

代码里没有手工选择某个中间层，例如没有指定 `layer3`、`layer4` 或某个 block。

实际做法是：

> 取预训练模型 `model.features` 的最后输出，也就是分类头之前的最后一个卷积特征模块输出。

在代码中：

```python
feat = model.features
wrapper = _NormalizedEmbeddingBackbone(feat)
```

然后：

```python
feat = self.feature_extractor(batch)
feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
feat = F.normalize(feat, dim=1)
```

统一流程：

```text
ROI 图像
 ↓
预训练 CNN 的 features 部分
 ↓
最后卷积特征图 C x H x W
 ↓
GAP 全局平均池化
 ↓
C 维 embedding
 ↓
L2 归一化
 ↓
用于 OK/NG 相似度判断
```

## 6. EfficientNet-B0 具体是第几层？

更严谨地说，这里不应该把 `features[8]` 叫成“CNN 的第 8 层”。`features[8]` 是 `torchvision` 把 EfficientNet-B0 的 `model.features` 组织成列表后，里面的第 8 号顶层模块。

如果一定要回答“取到哪里”，按 `torchvision` 的 EfficientNet-B0 结构编号，可以说：

> 对 EfficientNet-B0，本系统取的是 `model.features[8]` 的输出，也就是分类头之前最后一个卷积特征模块的输出。

注意：系统不是只运行 `features[8]`，而是从 `features[0]` 顺序执行到 `features[8]`，最后取 `features[8]` 的输出。

大致结构可以理解成：

```text
features[0]  初始卷积层
features[1]  MBConv stage 1
features[2]  MBConv stage 2
features[3]  MBConv stage 3
features[4]  MBConv stage 4
features[5]  MBConv stage 5
features[6]  MBConv stage 6
features[7]  MBConv stage 7
features[8]  最后的 1x1 卷积特征层
```

它输出的不是分类结果，而是特征图：

```text
features[8] 输出 ≈ 1280 x 7 x 7
```

然后：

```text
GAP: 1280 x 7 x 7 -> 1280
L2: 归一化 -> 1280 维 embedding
```

注意：

> 1280 不是第 1280 层，而是特征通道数 / embedding 维度。
> `features[8]` 也不是 CNN 的第 8 个实际计算层，而是 `features` 列表里的最后一个顶层特征模块。

## 7. MobileNetV3 具体是第几层？

MobileNetV3 也是同样逻辑：取 `model.features` 的最后输出。

这里的 `features[12]`、`features[16]` 也不是 CNN 的第 12 层、第 16 层，而是 `torchvision` 中 `model.features` 列表里的模块编号。

按 `torchvision` 的常见结构编号：

| 模型 | 取的模块 | 输出通道数 | embedding 维度 |
|---|---|---:|---:|
| MobileNetV3-small | `model.features[12]` | 576 | 576 维 |
| MobileNetV3-large | `model.features[16]` | 960 | 960 维 |

MobileNetV3-small：

```text
model.features[12] 输出
≈ 576 x 7 x 7
   ↓ GAP
576 维向量
   ↓ L2
576 维 embedding
```

MobileNetV3-large：

```text
model.features[16] 输出
≈ 960 x 7 x 7
   ↓ GAP
960 维向量
   ↓ L2
960 维 embedding
```

> 对 MobileNetV3-small，本系统取 `features[12]` 的输出；对 MobileNetV3-large，取 `features[16]` 的输出。它们都是分类头之前最后一个卷积特征模块的输出。576 和 960 表示特征通道数，不是网络层数。


取 12 和 16，是因为它们分别是 MobileNetV3-small / large 在 torchvision 中 classifier/head 之前的最后一个 features 模块。
因为 torchvision 对不同模型的拆分方式不同。MobileNetV3 把更多 InvertedResidual block 直接平铺在 features 列表里，所以列表更长；EfficientNet-B0 把多个 MBConv block 按 stage 分组，所以外层编号更短。

## block和stage的理解
layer  = 最小的计算层
block  = 几个 layer 组合成的功能小单元
stage  = 多个 block 组成的一个大阶段
举个简单例子：

text

Conv + BatchNorm + ReLU
这几个东西可以打包成一个小单元，这个小单元就可以叫一个 block。

再比如 MobileNetV3 里的 InvertedResidual block，里面可能包含：

text

1x1 卷积
3x3 depthwise 卷积
SE 注意力
1x1 卷积
BatchNorm
激活函数
残差连接
这一整组结构就叫一个 block。

stage 比 block 更大。它通常表示网络里的一个阶段：

text

stage 1：处理 128 x 128 的特征图
stage 2：处理 64 x 64 的特征图
stage 3：处理 32 x 32 的特征图
stage 4：处理 16 x 16 的特征图
stage 5：处理 8 x 8 的特征图
一个 stage 里面可以有多个 block。

关系像这样：

text

CNN 模型
│
├─ stage 1
│  ├─ block 1
│  └─ block 2
│
├─ stage 2
│  ├─ block 1
│  ├─ block 2
│  └─ block 3
│
└─ stage 3
   ├─ block 1
   └─ block 2
所以：

text

layer < block < stage < model
对应到 EfficientNet-B0：

text

features[1]  MBConv stage 1
features[2]  MBConv stage 2
features[3]  MBConv stage 3
...
features[7]  MBConv stage 7
这里每个 features[i] 可以看成一个 stage，里面可能包含多个 MBConv block。

对应到 MobileNetV3：

text

features[1]
features[2]
features[3]
...
features[12] 或 features[16]
它更像是把很多 InvertedResidual block 直接放在 features 列表里，所以编号更多。

可以用文件夹比喻：
layer  = 一个文件
block  = 一个小文件夹，里面有几个文件
stage  = 一个大文件夹，里面有几个小文件夹
model  = 整个项目文件夹
一句话：
>block 是一组层打包成的基本结构单元；stage 是一组 block 组成的网络阶段，通常对应一个特征图尺寸

## Conv + BatchNorm + ReLU 这什么？features[12] 这是一个stage?
Conv + BatchNorm + ReLU 是 CNN 里非常常见的一个基础组合。

拆开看：

Conv        卷积层，负责提取局部特征
BatchNorm   批归一化，让特征数值更稳定
ReLU        激活函数，引入非线性，让网络能表达复杂关系
可以理解成：

Conv：看图，提取边缘/纹理等特征
BatchNorm：把特征数值整理得更稳定
ReLU：把有用响应留下，把负值压掉
组合起来就是：

Conv + BatchNorm + ReLU
= 一个基础卷积特征提取单元
在 PyTorch / torchvision 里，这种组合经常被打包成一个 module。

第二个问题：features[12] 是不是一个 stage？

对 MobileNetV3-small 来说，features[12] 更准确地说是：

model.features 列表里的最后一个顶层模块
它通常不是一个完整 stage，而是最后的卷积模块。

MobileNetV3-small 的 features 大致可以理解成：

features[0]   初始 Conv + BN + Hardswish
features[1]   InvertedResidual block
features[2]   InvertedResidual block
...
features[11]  InvertedResidual block
features[12]  最后的 Conv + BN + Hardswish
所以：

features[12] 不是 CNN 的第 12 层
features[12] 也不太应该叫 stage
features[12] 是最后一个顶层特征模块
对比 EfficientNet-B0：

EfficientNet-B0 的 features[1]~features[7] 更像 stage
MobileNetV3 的 features[1]~features[11/15] 更像一个个 block
MobileNetV3 的 features[12/16] 是最后的卷积特征模块
为什么会这样？因为 torchvision 对不同模型的组织方式不同：

EfficientNet-B0：更多按 stage 分组
MobileNetV3：更多把 block 平铺在 features 列表中
一句话：

Conv + BatchNorm + ReLU 是基础卷积单元；MobileNetV3-small 的 features[12] 不是 stage，

## Hardswish 这是什么

Hardswish 是一种激活函数，主要用在 MobileNetV3 里。

你可以先把它理解成：

Hardswish 是 ReLU 的一种更平滑、更适合轻量模型的改进激活函数。
常见激活函数有：

ReLU
SiLU / Swish
Hardswish
ReLU 很简单：

x <= 0 -> 输出 0
x > 0  -> 输出 x
公式：

ReLU(x) = max(0, x)
它像一个开关：

负数不要，正数保留
Swish / SiLU 更平滑：

SiLU(x) = x * sigmoid(x)
它不是直接把负数全部砍成 0，而是平滑地压低一部分负值。表达能力更好，但计算里有 sigmoid，对移动端轻量模型不够省。

Hardswish 是 Swish 的近似版本：

Hardswish(x) = x * ReLU6(x + 3) / 6
其中：

ReLU6(y) = min(max(0, y), 6)
它的作用是：

用更简单的分段线性计算，近似 Swish 的效果
为什么 MobileNetV3 用它？

因为 MobileNetV3 目标是轻量、快、适合移动端或边缘设备。Hardswish 相比 Swish/SiLU：

计算更简单
速度更快
对移动端更友好
效果又比普通 ReLU 更好一些
可以这样解释：

ReLU 是简单开关；
Swish/SiLU 是更平滑的开关；
Hardswish 是 Swish 的轻量近似版本，尽量保留效果，同时降低计算成本。
在你的文档里可以写：

Hardswish 是 MobileNetV3 常用的激活函数，可以看成 Swish/SiLU 的轻量近似版本。它用于在卷积和归一化之后引入非线性，使网络能表达更复杂的图像特征，同时保持较低计算成本。
## 8. GAP 是什么？

GAP = Global Average Pooling，全局平均池化。

它的原理很简单：

> 对每一个特征通道，把整张特征图上的数值求平均。

假设 CNN 最后输出一个特征图：

```text
C x H x W
```

比如 EfficientNet-B0 可能是：

```text
1280 x 7 x 7
```

含义是：

- 1280 个特征通道
- 每个通道有 7x7 个响应值
- 每个通道可以理解为一种视觉模式响应，例如边缘、纹理、形状、局部结构等

GAP 对每个通道单独求平均：

```text
第 1 个通道：7x7 求平均 -> 1 个数
第 2 个通道：7x7 求平均 -> 1 个数
...
第 1280 个通道：7x7 求平均 -> 1 个数
```

所以：

```text
1280 x 7 x 7
   ↓ GAP
1280 x 1 x 1
   ↓ flatten
1280 维向量
```

数学表达：

```text
y_c = 1 / (H × W) × Σ x_c(i, j)
```

意思是：

```text
第 c 个通道的输出 = 这个通道所有空间位置响应值的平均值
```

可以再用更直观的方式理解：

> 1280 个通道，每个通道经过 GAP 后得到 1 个数。  
> 1280 个通道就得到 1280 个数。  
> 这 1280 个数拼在一起，就是 1280 维特征向量；后面再经过 L2 归一化，成为 1280 维 embedding。

假设 EfficientNet-B0 最后输出：

```text
1280 x 7 x 7
```

这里不是 1280 张图片，而是 **1280 个特征通道**。

每个通道里面有一个 `7 x 7` 的小矩阵，例如通道 1：

```text
a11 a12 ... a17
a21 a22 ... a27
...
a71 a72 ... a77
```

GAP 对通道 1 的 49 个数求平均：

```text
通道 1 平均值 = 这 49 个数的平均
```

得到 1 个数：

```text
f1
```

通道 2 也是一样，得到：

```text
f2
```

一直到通道 1280，得到：

```text
f1280
```

最后拼起来：

```text
[f1, f2, f3, ..., f1280]
```

这就是：

```text
1280 维特征向量
```

所以完整变化是：

```text
1280 x 7 x 7
   ↓ 对每个通道的 7x7 求平均
1280 x 1 x 1
   ↓ flatten
1280
   ↓ L2 normalize
1280 维 embedding
```

一句话总结：

> 1280 个通道 -> 1280 个平均值 -> 1280 维特征向量 -> L2 归一化后得到 1280 维 embedding。

## 9. 代码里是怎么实现 GAP 的？

代码里 GAP 是用 PyTorch 的 `adaptive_avg_pool2d` 实现的。

核心代码在 `algorithms/embedding.py`：

```python
feat = self.feature_extractor(batch)
feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
feat = F.normalize(feat, dim=1)
return feat
```

这句就是 GAP：

```python
F.adaptive_avg_pool2d(feat, 1)
```

含义是：不管输入特征图空间尺寸是多少，都平均池化成 `1 x 1`。

例如：

```text
batch x 1280 x 7 x 7
        ↓ adaptive_avg_pool2d(feat, 1)
batch x 1280 x 1 x 1
        ↓ flatten(1)
batch x 1280
        ↓ L2 normalize
batch x 1280 embedding
```

一句话：

> 代码里的 GAP 就是 `F.adaptive_avg_pool2d(feat, 1)`，它把每个通道的空间响应求平均，压缩成一个固定长度的特征向量。

## 10. L2 归一化是什么？

L2 归一化就是把 embedding 向量除以自己的长度，让它变成单位向量。

```text
embedding = embedding / ||embedding||
```

这样做的好处是：

- 降低特征幅值大小的影响
- 更关注特征方向是否相似
- 方便用余弦相似度比较 OK/NG 样本

归一化之后：

```text
||embedding|| = 1
```

两个 embedding 越接近，说明它们在视觉特征上越相似。

## 11. 特征图和 embedding 的关系是什么？

特征图是 CNN 内部输出，embedding 是给检测算法使用的最终特征向量。

关系如下：

```text
CNN 特征图 C x H x W
        ↓ GAP
C 维特征向量
        ↓ L2
embedding
```

例如 EfficientNet-B0：

```text
最后卷积特征图：1280 x 7 x 7
GAP 后：1280 维向量
L2 后：1280 维 embedding
```

所以：

- 特征图仍然保留空间位置响应
- embedding 是压缩后的整体视觉特征表示
- 检测阶段使用的是 embedding，不是直接使用原始特征图

更详细地说，二者的区别在于：

| 项目 | CNN 特征图 | embedding |
|---|---|---|
| 位置 | CNN `features` 的最后输出 | 特征图经过 GAP + L2 后的结果 |
| 形状 | `C x H x W` | `C` 维向量 |
| 是否有空间位置 | 有，例如 `7 x 7` 每个位置都有响应 | 没有空间网格，只保留整体特征强度 |
| 表达内容 | 每个通道在不同位置上的响应 | 整个 ROI 的综合视觉特征 |
| 是否直接用于检测 | 不直接用于最终相似度判断 | 直接用于 OK/NG 相似度判断 |
| 举例 | `1280 x 7 x 7` | `1280` 维向量 |

可以把特征图理解成“还带位置的局部响应图”：

```text
第 1 个通道：ROI 上某类边缘/纹理在 7x7 位置上的响应
第 2 个通道：ROI 上另一类结构在 7x7 位置上的响应
...
第 1280 个通道：另一种视觉模式的响应
```

此时模型不仅知道“有没有这个特征”，还大致保留这个特征出现在 ROI 的哪个空间位置。

而 embedding 是把这些空间响应压缩成“整体外观身份证”：

```text
第 1 个数：第 1 类视觉特征在整个 ROI 中的整体强度
第 2 个数：第 2 类视觉特征在整个 ROI 中的整体强度
...
第 1280 个数：第 1280 类视觉特征在整个 ROI 中的整体强度
```

GAP 做的事情就是把每个通道的 `7 x 7` 空间响应求平均：

```text
某个通道的 7 x 7 响应
        ↓ 求平均
这个通道的 1 个整体响应值
```

所以 EfficientNet-B0 中：

```text
1280 个通道 × 每个通道 7×7 个位置
        ↓ GAP
1280 个通道 × 每个通道 1 个平均值
        ↓ flatten
1280 维特征向量
        ↓ L2
1280 维 embedding
```

为什么检测阶段不用原始特征图，而用 embedding？

1. **维度更固定**  
   特征图有空间尺寸，处理起来更复杂；embedding 是固定长度向量，方便保存和比较。

2. **更适合做相似度比较**  
   OK/NG 注册模型需要比较“新图和 OK/NG 样本有多像”，向量之间的余弦相似度更直接。

3. **降低对局部位置的敏感性**  
   对错漏装、反装、有无这类任务，很多时候更关心 ROI 整体外观是否相似，而不是每个局部响应的精确位置。

4. **便于保存注册模型**  
   `.npz` 中只需要保存每张样本的 embedding，而不需要保存完整特征图，文件更小、计算更简单。

需要注意的是：

> embedding 不是原始图像，也不是 CNN 的最终分类结果，而是由最后卷积特征图压缩得到的整体视觉特征向量。它保留了模型认为重要的外观差异，但不再保留详细的 `H x W` 空间网格。

汇报时可以这样说：

> 特征图是 CNN 内部还带空间位置的响应结果，例如 EfficientNet-B0 最后得到 `1280 x 7 x 7` 的特征图。系统不会直接拿这个特征图做 OK/NG 判断，而是通过 GAP 把每个通道的空间响应平均成一个数，再经过 L2 归一化得到 `1280` 维 embedding。这个 embedding 可以理解成 ROI 外观的特征身份证，后续用它和 OK/NG 原型计算余弦相似度。

## 12. OK 原型 / NG 原型是什么？

OK 原型和 NG 原型就是 OK 样本、NG 样本的平均特征代表。

每张样本都会先变成一个 embedding：

```text
OK 样本 1 -> embedding 1
OK 样本 2 -> embedding 2
OK 样本 3 -> embedding 3
```

然后对所有 OK embedding 求平均：

```text
OK 原型 = 所有 OK 样本 embedding 的平均值
```

NG 同理：

```text
NG 原型 = 所有 NG 样本 embedding 的平均值
```

可以理解成：

```text
OK 原型 = 合格品的特征中心
NG 原型 = 不合格品的特征中心
```

检测新图片时：

```text
新图片 embedding
     ↓
和 OK 原型算相似度
和 NG 原型算相似度
     ↓
更像 OK -> 判 OK
更像 NG -> 判 NG
```

汇报说法：

> 原型不是新训练出来的神经网络参数，而是由当前产品样本 embedding 计算出的特征中心。OK 原型代表合格样本的平均视觉特征，NG 原型代表不合格样本的平均视觉特征。

## 13. `.npz` 注册模型文件保存的是什么？

例如：

```text
cam1_roi1_register_model_efficientnet_b0.npz
```

或实际文件中可能使用短码：

```text
cam1_roi1_register_model_b0.npz
```

其中：

```text
b0 = efficientnet_b0
b1 = mobilenet_v3_small
b2 = mobilenet_v3_large
```

这个 `.npz` 文件保存的是某个相机、某个 ROI、某个 backbone 下的小样本注册结果。

它**不是 CNN 模型权重**，也**不是重新训练出来的神经网络**。它保存的是当前产品 OK/NG 样本提取后的 embedding 特征和判定参数。

常见字段如下：

| 字段 | 含义 |
|---|---|
| `backbone` | 使用的特征模型，例如 `b0` / `efficientnet_b0` |
| `score_mode` | 判定模式，例如 `proto` 或 `topk` |
| `margin` | OK/NG 判定阈值 |
| `topk` | topk 相似度模式下使用的近邻数量 |
| `label_name` | ROI 名称，例如 `roi1` |
| `label_names` | 参与注册的 ROI 列表 |
| `device` | 注册时使用的设备，例如 `cpu` |
| `ok_proto` | OK 样本的平均 embedding，也就是 OK 原型 |
| `ng_proto` | NG 样本的平均 embedding，也就是 NG 原型 |
| `ok_bank` | 所有 OK 样本的 embedding 特征库 |
| `ng_bank` | 所有 NG 样本的 embedding 特征库 |

如果是 EfficientNet-B0，embedding 维度一般是 `1280`，所以形状类似：

```text
ok_proto: (1, 1280)
ng_proto: (1, 1280)
ok_bank:  (OK样本数, 1280)
ng_bank:  (NG样本数, 1280)
```

检测时流程：

```text
Cam1 ROI1 新图片
   ↓
EfficientNet-B0 提 embedding
   ↓
和 ok_proto / ng_proto 算相似度
   ↓
根据 margin 判断 OK 或 NG
```

一句话：

> `.npz` 文件保存的是某个相机 ROI 的 OK/NG 注册特征模型，包括 OK/NG 特征库、OK/NG 原型和判定参数；它不保存完整 CNN，也不保存新训练的 CNN 权重。

## 14. 余弦相似度是怎么计算的？

这里的余弦相似度本质上是两个 embedding 向量的点积。

因为代码里 embedding 已经做过 L2 归一化，所以点积等价于 cosine similarity。

标准余弦相似度公式：

```text
cos(e, ok_proto) = e · ok_proto / (||e|| × ||ok_proto||)
```

由于：

```text
||e|| = 1
||ok_proto|| = 1
```

所以变成：

```text
cos(e, ok_proto) = e · ok_proto
```

代码逻辑：

```python
sim_ok = float(e @ ok_proto[0])
sim_ng = float(e @ ng_proto[0])
diff = sim_ok - sim_ng
pred = "OK" if diff >= margin else "NG"
```

也就是：

```text
sim_ok = 新图 embedding 和 OK 原型的余弦相似度
sim_ng = 新图 embedding 和 NG 原型的余弦相似度
diff = sim_ok - sim_ng
```

判断：

```text
diff >= margin -> OK
diff < margin  -> NG
```

## 15. 每一张新图都要和 OK、NG 都算一遍相似度吗？

对。

每一张新图检测时，都会先提取一次 embedding，然后分别和 OK 侧、NG 侧计算相似度。

流程：

```text
新图 ROI
  ↓
CNN 提 embedding
  ↓
和 OK 特征算相似度
和 NG 特征算相似度
  ↓
比较哪个更像
  ↓
输出 OK / NG
```

如果是 `proto` 模式：

```text
新图 embedding
  ↓
和 OK 原型算余弦相似度 -> sim_ok
和 NG 原型算余弦相似度 -> sim_ng
  ↓
diff = sim_ok - sim_ng
```

然后：

```text
diff >= margin -> OK
diff < margin  -> NG
```

更口语化：

```text
新图不是直接问“是不是 OK”，
而是同时问：
它像 OK 样本有多像？
它像 NG 样本有多像？
然后比较两边分数。
```

## 16. topk 怎么快速理解？

`topk` 可以理解成：

> 不拿新图和“平均代表”比，而是拿新图和样本库里最像的几个样本比。

比如 OK 样本库里有 10 张 OK 样本：

```text
OK1 OK2 OK3 ... OK10
```

新图来了以后，系统会算：

```text
新图 vs OK1
新图 vs OK2
新图 vs OK3
...
新图 vs OK10
```

得到 10 个相似度，然后取最高的 `k` 个。

如果 `topk = 3`，就取最像的 3 个：

```text
0.91, 0.88, 0.86
```

再求平均：

```text
sim_ok = (0.91 + 0.88 + 0.86) / 3
```

NG 也一样：

```text
新图 vs 所有 NG 样本
取最像的 3 个
求平均
得到 sim_ng
```

最后比较：

```text
diff = sim_ok - sim_ng
diff >= margin -> OK
diff < margin  -> NG
```

和 `proto` 的区别：

| 模式 | 怎么比 | 快速理解 |
|---|---|---|
| `proto` | 和 OK/NG 平均原型比 | 看它更像哪一类的平均代表 |
| `topk` | 和 OK/NG 样本库里最像的 k 个样本比 | 看它在样本库里更像哪一边的近邻 |

一句话：

> `proto` 看“像不像平均脸”，`topk` 看“最近的几个熟人像不像”。

topk 的好处：

- 如果 OK 或 NG 样本内部差异比较大，平均原型可能不够代表全部情况
- topk 会更关注最接近的几个真实样本

topk 的风险：

- 如果样本库里有错误样本或边界样本太乱，topk 也可能被这些样本影响
- 所以样本质量很重要

## 17. 特征分析图是什么？

“打开特征分析图”不是重新取某一层，也不是显示 CNN 原始特征图。

它做的是：

```text
读取 register_model_xxx.npz
      ↓
取 ok_bank / ng_bank
      ↓
拼成 all_features
      ↓
用 PCA 或 TSNE 降到 2 维
      ↓
画 OK/NG 散点图
```

也就是说，特征分析图显示的是已经注册好的 embedding 特征库在二维空间中的分布关系。

区别如下：

| 项目 | 注册 / 检测 | 特征分析图 |
|---|---|---|
| 数据来源 | ROI 图像经过 CNN 得到 embedding | 读取已保存的 `ok_bank / ng_bank` |
| 是否重新取模型层 | 是，走模型推理 | 否，直接读取保存的 embedding |
| 是否参与判定 | 是，直接用于 OK/NG 相似度计算 | 否，只是可视化分析 |
| 数据维度 | 1280 / 576 / 960 等高维向量 | PCA/TSNE 降到 2 维 |
| 图上坐标含义 | 无 | 二维投影坐标，不是图像坐标 |

需要注意：

> 特征分析图不是算法输入，也不是模型层输出；它是对注册后 embedding 特征库的二维可视化，用来辅助判断 OK/NG 样本是否大致分离、是否存在异常样本。

## 18. 特征分析图是二维的吗？能准确看出 OK/NG 分布吗？

是的，特征分析图是二维的。

但它不是原始 embedding 空间，而是把高维 embedding 通过 PCA 或 TSNE 降维到 2 维后画出来。

比如 EfficientNet-B0 的 embedding 是：

```text
1280 维
```

特征分析图显示的是：

```text
1280 维 embedding
     ↓ PCA / TSNE
2 维散点图
```

它能做什么：

| 用途 | 是否适合 |
|---|---|
| 看 OK/NG 是否大致分开 | 适合 |
| 发现明显混在一起的异常样本 | 适合 |
| 判断样本质量是否有问题 | 适合 |
| 精确决定检测阈值 | 不建议只靠它 |
| 证明模型一定可靠 | 不可以 |
| 替代实际测试集验证 | 不可以 |

原因：

- 原始特征是高维的，真实判断在 576 / 960 / 1280 维 embedding 空间里做
- 二维图是降维投影，会压缩信息
- TSNE 更偏可视化，坐标和全局距离不一定能直接解释
- 最终判断用的是 `sim_ok - sim_ng >= margin`，不是图上的二维距离

汇报时可以说：

> 特征分析图是把高维 embedding 降到 2 维后的可视化工具，可以辅助观察 OK/NG 样本是否大致分离、是否存在异常样本，但不能完全代表真实高维空间的判定效果。最终可靠性仍然要看高维相似度分数、margin，以及实际测试样本验证。

## 19. 特征分析图的横坐标和纵坐标代表什么？

特征分析图的横坐标和纵坐标没有直接物理含义。

它们不是：

```text
不是图像 X/Y 坐标
不是宽度/高度
不是亮度/颜色
不是某两个固定 CNN 特征通道
```

它们只是把高维 embedding 降维后得到的两个投影维度。

如果用 PCA：

```text
横坐标：第一主成分方向
纵坐标：第二主成分方向
```

它们是很多 embedding 维度的线性组合，不是某一个具体特征。

如果用 TSNE：

```text
横坐标和纵坐标更不能直接解释成具体含义
```

TSNE 主要表达：

```text
哪些样本在高维空间里比较接近
哪些样本大致形成一簇
OK 和 NG 是否大致分开
```

汇报时可以说：

> 特征分析图的横纵坐标是高维 embedding 降维后的两个可视化坐标，不代表图像位置或具体尺寸。我们主要看 OK/NG 点是否大致分簇、是否混在一起，而不是解释横坐标或纵坐标各自代表什么物理量。

## 20. 这种方式检测靠不靠谱？

靠谱，但要讲清楚边界：

> 它适合“外观相似性判断类”的错漏装检测，不适合替代所有视觉检测。

可靠的原因：

1. CNN 特征比传统灰度 / 边缘特征更稳  
   它不是只看单个灰度阈值，而是综合纹理、边缘、形状、局部结构。

2. 用的是预训练模型，不是现场从零训练  
   官方预训练权重已经学过大量通用视觉特征，现场只做 embedding 注册。

3. 检测逻辑是相似度判断，比较直观  
   新图像更像 OK 原型还是 NG 原型，适合错装、漏装、反装、部件有无等外观差异。

4. 现场可持续补样本  
   如果发现某种边界情况误判，可以补 OK/NG 样本重新注册，不需要重新写代码或训练网络。

风险和边界：

1. 样本覆盖不够会不稳  
   如果 OK 样本只覆盖一种光照、一种位置，实际生产变化很大，原型代表性不足。

2. ROI 必须稳定  
   如果定位偏了、ROI 框错了，提出来的 embedding 就不对应同一个区域。

3. OK/NG 差异太小会困难  
   比如微小尺寸差异、极细划伤、边缘间距超差，更适合尺寸测量或传统几何算法。

4. 二维特征分析图只能辅助判断  
   最终检测不是靠图上点分不分开，而是靠高维 embedding 相似度。

汇报说法：

> 这种方法不是“拍脑袋 AI 判断”，而是固定预训练 CNN 提取 embedding，再用当前产品 OK/NG 样本建立特征原型。只要 ROI 稳定、样本覆盖了现场变化、OK/NG 外观差异明确，它在错漏装、反装、有无检测这类任务上是可靠的。对于精密尺寸、间隙、角度等确定性测量，仍然应使用尺寸测量模块或传统视觉算法。

## 21. 面对“CNN 训练”问题时的简洁回答

如果别人一直问“CNN 是怎么训练的”，可以直接区分：

| 对方说的 CNN 训练 | 本系统的注册学习 |
|---|---|
| 重新训练神经网络分类器 | 使用预训练 CNN 做特征提取 |
| 需要大量 OK/NG 图片 | 只需要少量样本 |
| 会更新 CNN 权重 | 不更新 CNN 权重 |
| 训练分类头或整网 | 只建立 embedding 特征库和原型 |
| 训练时间较长 | 注册速度快 |
| 更依赖算法工程师 | 车间人员可界面操作 |

最简洁的回答：

> 我们用的是官方预训练 CNN 权重，只作为固定特征提取器推理；现场所谓训练，是把当前产品 OK/NG 样本转成 embedding，并建立 OK/NG 原型，不重新训练 CNN 权重。

## 22. PPT 推荐表达

可以在 PPT 中写成：

> 本系统使用官方预训练 CNN 权重作为通用视觉特征提取器。现场小样本注册阶段只进行前向推理，提取当前产品 ROI 的 embedding 特征，并建立 OK/NG 特征原型；不会重新训练或微调 CNN 权重。

也可以更口语化地汇报：

> 这里的“学习”不是重新训练深度学习模型，而是把当前产品的 OK/NG 样本注册成特征库。CNN 权重是固定的，负责提取通用视觉特征；真正跟产品相关的是后面保存的 OK/NG embedding 原型。

## 23. PyTorch 和 torchvision 分别是什么？

`PyTorch` 是现在常用的深度学习框架，可以理解成深度学习里的基础开发平台。它负责：

- Tensor 张量计算
- GPU / CPU 推理和训练
- 自动求导
- 神经网络模块 `nn.Module`
- 模型保存和加载
- 前向推理和反向传播

`torchvision` 是 PyTorch 官方配套的计算机视觉工具库。它专门服务图像任务，里面提供：

- 现成视觉模型，例如 ResNet、MobileNet、EfficientNet、VGG、Vision Transformer
- ImageNet 预训练权重
- 图像预处理工具，例如 resize、crop、normalize
- 常用视觉数据集接口
- 检测、分割等视觉模型组件

在本系统里，主要用的是：

```python
from torchvision import models

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)
```

含义是：

```text
从 torchvision 里拿一个官方已经定义好的 EfficientNet-B0 模型，
并加载官方 ImageNet 预训练权重。
```

如果用 C# / .NET 类比，可以这样理解：

```text
PyTorch     ≈ .NET 平台 + 基础运行能力 + 核心 API
torchvision ≈ 官方提供的计算机视觉 NuGet 扩展包
```

一句话：

> PyTorch 是深度学习基础框架，torchvision 是 PyTorch 官方给图像任务准备的模型库和工具库。

## 24. CNN 里的“层”和 PyTorch 里的“模块”有什么区别？

最关键的一句话：

```text
层 layer：更偏理论和计算操作
模块 module：更偏代码组织方式，可以包含一层，也可以包含很多层
```

在 CNN 理论里，常说：

```text
卷积层 Conv
池化层 Pooling
全连接层 Fully Connected / Linear
激活层 ReLU / SiLU
归一化层 BatchNorm
```

但在 PyTorch 代码里，这些东西通常都继承自 `nn.Module`：

```text
Conv2d        是 nn.Module，也常叫卷积层
BatchNorm2d   是 nn.Module，也常叫 BN 层
ReLU / SiLU   是 nn.Module，也常叫激活层
MBConv block  是 nn.Module，但它里面包含多个真实层
features[8]   是 nn.Module，里面也可能包含 Conv + BN + 激活
整个模型       也是 nn.Module
```

所以关系更像：

```text
nn.Module
├─ 简单层：Conv2d、BatchNorm、ReLU、Linear
├─ 复合块：MBConv、InvertedResidual
├─ 阶段：stage
└─ 整个模型：EfficientNet-B0
```

也就是说：

```text
层是模块的一种；
但模块不一定只是一层。
```

例如：

```text
features[8]
└─ Conv2d + BatchNorm2d + SiLU
```

外面看是一个 `features[8]` 模块，里面实际包含多个计算操作。

再比如一个 MBConv block 可能包含：

```text
1x1 Conv
3x3 Depthwise Conv
SE 注意力
1x1 Conv
BatchNorm
Activation
Residual connection
```

这个 block 在代码里可以是一个模块，但不能简单说它只是“一层”。

一句话：

> `features[8]` 是代码里的模块编号，不是 CNN 理论里的第 8 个实际计算层。

## 25. 为什么 PyTorch 要统一用 `nn.Module`？

PyTorch 把层、block、stage、整个模型都统一抽象成 `nn.Module`，是为了让所有网络组件都能用同一套机制管理。

`nn.Module` 可以统一处理：

```text
保存参数
加载参数
切换 train/eval 模式
移动到 CPU/GPU
参与 forward 前向计算
嵌套子模块
导出模型
```

所以不管是一个小层：

```text
Conv2d
```

还是一个复杂 block：

```text
MBConv
```

还是整个模型：

```text
EfficientNet-B0
```

都可以统一这样使用：

```python
module.to(device)
module.eval()
module(x)
module.state_dict()
```

如果没有 `nn.Module` 这个统一抽象，不同类型的层、block、模型就要各自写不同的参数管理、设备管理、保存加载逻辑，代码会很难组合。

所以：

```text
PyTorch 不是不用“层”的概念，
而是代码实现上把层、block、stage、整个模型都统一组织成 Module。
```

## 26. EfficientNet-B0 的 `features[8]` 和 CNN 层级到底是什么关系？

EfficientNet-B0 是一种 CNN backbone。它在 torchvision 里的整体结构可以理解成：

```text
EfficientNet-B0
│
├─ model.features      特征提取部分
│  ├─ features[0]      初始卷积模块
│  ├─ features[1]      MBConv stage 1
│  ├─ features[2]      MBConv stage 2
│  ├─ features[3]      MBConv stage 3
│  ├─ features[4]      MBConv stage 4
│  ├─ features[5]      MBConv stage 5
│  ├─ features[6]      MBConv stage 6
│  ├─ features[7]      MBConv stage 7
│  └─ features[8]      最后的 1x1 卷积特征模块
│
├─ model.avgpool       全局平均池化
└─ model.classifier    ImageNet 分类头 / head
```

这里的 `features[8]` 不是 CNN 的第 8 层，而是：

```text
model.features 这个列表里的第 8 号顶层模块。
```

本系统也不是只运行 `features[8]`。真实流程是：

```text
ROI 图像
  -> features[0]
  -> features[1]
  -> features[2]
  -> ...
  -> features[8]
  -> GAP
  -> L2 normalize
  -> embedding
```

所以更准确的说法是：

> 系统使用 EfficientNet-B0 的完整 `features` 特征提取器，也就是从 `features[0]` 顺序执行到 `features[8]`。`features[8]` 只是最后一个特征模块的输出位置，位于 `avgpool` 和 `classifier/head` 之前。

不要说：

```text
系统只用了第 8 层。
```

应该说：

```text
系统使用了 features[0]~features[8] 的完整特征提取部分，最后取 features[8] 的输出。
```

## 27. 如果别人问“CNN 到底用了多少层”，应该怎么回答？

这个问题必须先说明统计口径，因为现代 CNN 里“层数”没有唯一答案。

不同口径会得到不同数字：

```text
只数 Conv2d 卷积层：一个数字
数 Conv2d + Linear：另一个数字
把 BatchNorm、激活函数也算层：数字会更多
把 MBConv block 算 1 层：又是另一个数字
按 torchvision 顶层 features 模块算：EfficientNet-B0 是 9 个模块
```

在当前 torchvision 实现中，实际统计结果可以这样说：

```text
EfficientNet-B0 的 features 特征提取部分：
按顶层 features 模块统计：9 个模块，features[0]~features[8]
按 Conv2d 卷积模块统计：81 个 Conv2d

完整 EfficientNet-B0 分类模型：
Conv2d：81 个
Linear 分类层：1 个
Conv2d + Linear：82 个带参数计算层
```

但本系统不使用原始 ImageNet 分类头，所以对本系统更准确的表达是：

```text
系统使用 EfficientNet-B0 的完整 features 特征提取器。
按 torchvision 顶层模块看，是 features[0]~features[8] 共 9 个模块；
按 Conv2d 卷积层统计，features 部分包含 81 个 Conv2d；
系统不使用最后的 ImageNet classifier/head。
```

注意不要直接说：

```text
系统只用了 81 层。
```

因为 `81` 只代表“按 Conv2d 统计”的卷积模块数量，不代表所有 layer 的总数量。

更严谨的回答：

> 如果把 CNN layer 特指为卷积层，那么本系统使用的 EfficientNet-B0 features 部分包含 81 个 Conv2d 卷积模块；但如果把 BatchNorm、激活函数、SE、池化等也算作 layer，总数会更多。因此工程上更推荐说明使用的 backbone 和取特征的位置，而不是只说总层数。

## 28. 什么时候可以用“层数”描述模型，什么时候不建议只说层数？

不是说深度学习完全不用“层数”。简单、规整的网络可以用层数描述。

比如早期 CNN：

```text
VGG-16
VGG-19
```

这里的 `16 / 19` 大体表示带参数的层数，主要是卷积层和全连接层。

ResNet 也常用层数命名：

```text
ResNet-18
ResNet-34
ResNet-50
ResNet-101
ResNet-152
```

这些名字里的数字是模型设计者定义好的统计口径，通常用于区分同一系列不同深度的版本。

但现代 CNN 结构更复杂，例如 EfficientNet、MobileNetV3，它们包含：

```text
普通卷积 Conv
Depthwise Conv
Pointwise 1x1 Conv
BatchNorm
激活函数
SE 注意力
残差连接
MBConv / InvertedResidual block
stage
```

这时只问“多少层”会不清楚，因为必须先规定：

```text
BN 算不算层？
激活函数算不算层？
SE 里的 1x1 Conv 算不算层？
一个 MBConv 算 1 层，还是拆开算里面的多个卷积层？
残差连接算不算层？
```

所以现代 CNN 工程汇报更常说：

```text
使用哪个 backbone？
取哪个 feature 输出？
输出通道数是多少？
输出特征图尺寸是多少？
embedding 维度是多少？
是否使用 classifier/head？
是否更新 backbone 权重？
```

例如本系统更专业的说法是：

> 使用 EfficientNet-B0 作为 backbone，取 `model.features` 的最后输出，也就是 classifier/head 之前的最后一个卷积特征模块输出；再经过 GAP 和 L2 归一化得到 1280 维 embedding。CNN 权重固定，不使用原始 ImageNet 分类头。

一句话：

> 层数仍然可以说，但现代 CNN 结构复杂，只说“多少层”往往不够准确；更清楚的描述是“用哪个 backbone、取哪个 feature 输出”。

## 29. `features[8]` 为什么输出 `1280 x 7 x 7` 或 `1280 x 8 x 8`？

特征图的形状一般写作：

```text
C x H x W
```

PyTorch 输出里通常带 batch 维度，所以是：

```text
N x C x H x W
```

含义：

```text
N = batch，一次处理几张图
C = channel，特征通道数
H = height，特征图高度
W = width，特征图宽度
```

例如输入 `256 x 256` 时，实际打印 EfficientNet-B0 的 `model.features` 输出如下：

```text
input torch.Size([1, 3, 256, 256])
0 torch.Size([1, 32, 128, 128])
1 torch.Size([1, 16, 128, 128])
2 torch.Size([1, 24, 64, 64])
3 torch.Size([1, 40, 32, 32])
4 torch.Size([1, 80, 16, 16])
5 torch.Size([1, 112, 16, 16])
6 torch.Size([1, 192, 8, 8])
7 torch.Size([1, 320, 8, 8])
8 torch.Size([1, 1280, 8, 8])
```

最后一行：

```text
[1, 1280, 8, 8]
```

表示：

```text
1 张图
1280 个特征通道
每个通道是 8 x 8 的空间响应
```

为什么是 `8 x 8`？

因为 EfficientNet-B0 的特征提取部分会把输入空间尺寸大约下采样 32 倍：

```text
256 / 32 = 8
```

如果输入是 `224 x 224`：

```text
224 / 32 = 7
```

所以最后常见是：

```text
输入 224 x 224 -> features[8] 输出约 1280 x 7 x 7
输入 256 x 256 -> features[8] 输出约 1280 x 8 x 8
输入 320 x 320 -> features[8] 输出约 1280 x 10 x 10
```

`7 x 7` 不是固定永远如此，而是和输入尺寸有关。

什么时候 `/2`？

```text
stride = 2 -> 宽高大约 /2
stride = 1 -> 宽高基本不变
```

以 `256 x 256` 为例：

```text
输入          256 x 256
features[0]   128 x 128   stride=2
features[1]   128 x 128   stride=1
features[2]    64 x 64    stride=2
features[3]    32 x 32    stride=2
features[4]    16 x 16    stride=2
features[5]    16 x 16    stride=1
features[6]     8 x 8     stride=2
features[7]     8 x 8     stride=1
features[8]     8 x 8     stride=1
```

为什么 `8 x 8` 后面不继续变成 `4 x 4`？

因为后面的模块没有再安排 `stride=2` 的下采样。`features[8]` 是最后的 `1x1` 卷积特征模块，主要改变通道数，不再缩小空间尺寸。

后面由 GAP 负责把空间尺寸去掉：

```text
1280 x 8 x 8
  -> GAP
1280
```

## 30. 为什么通道数是 `32、16、24、40、80、112、192、320、1280`，不是一直递增？

输出形状：

```text
[N, C, H, W]
```

其中：

```text
H / W 是否变小：主要看 stride
C 变成多少：看这一层或这个模块设置了多少 out_channels
```

所以：

```text
256 -> 128 -> 64 -> 32 -> 16 -> 8
```

这是空间尺寸变化，由 `stride=2` 控制。

而：

```text
32 -> 16 -> 24 -> 40 -> 80 -> 112 -> 192 -> 320 -> 1280
```

这是通道数变化，由 EfficientNet-B0 的网络结构设计决定，不要求每一步都递增。

为什么 `32 -> 16` 会变小？

因为：

```text
features[0] 是 stem 初始卷积，输出 32 个通道
features[1] 是第一个 MBConv stage，输出被设计成 16 个通道
```

卷积可以重新组合通道：

```text
32 个输入通道 -> 卷积重新组合 -> 16 个输出通道
```

这不是简单丢掉 16 张图，而是把低级特征重新组合成更紧凑的特征表示。

后面大体上通道数会越来越多，因为空间尺寸越来越小：

```text
128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
```

每张特征图面积变小后，计算量下降，就可以增加通道数，让网络表达更复杂的纹理、结构和形状。

最后：

```text
320 -> 1280
```

是 `features[8]` 的最后 `1x1` 卷积，把深层特征扩展到 1280 个通道，方便后面做 GAP 得到 1280 维 embedding。

一句话：

> 宽高变化看 stride；通道数变化看 out_channels；这些通道数是 EfficientNet-B0 架构设计好的，不是按固定公式递增。

## 31. 卷积层的原理是什么？

卷积层的核心是：

```text
用一个小窗口在图像上滑动，提取局部特征。
```

这个小窗口叫：

```text
卷积核 / kernel / filter
```

例如一个 `3 x 3` 卷积核：

```text
图片局部区域          卷积核
1 2 3              a b c
4 5 6       ×      d e f
7 8 9              g h i
```

每滑到一个位置，就做一次加权求和：

```text
1*a + 2*b + 3*c
+ 4*d + 5*e + 6*f
+ 7*g + 8*h + 9*i
= 输出特征图上的一个点
```

这个卷积核在整张图上滑一遍，就得到一张新的特征图。

关键关系：

```text
1 个卷积核  -> 生成 1 个输出通道
32 个卷积核 -> 生成 32 个输出通道
1280 个卷积核 -> 生成 1280 个输出通道
```

所以：

```text
输入: [1, 3, 256, 256]
features[0] 输出: [1, 32, 128, 128]
```

可以理解成：

```text
原图有 3 个通道，也就是 RGB
features[0] 用 32 个卷积核去看图
所以输出 32 个特征通道
因为 stride=2，所以宽高 256 -> 128
```

浅层卷积核通常学到：

```text
边缘
颜色变化
简单纹理
角点
```

深层卷积组合浅层特征，通常表达：

```text
复杂纹理
局部结构
形状模式
零件外观差异
```

在本系统里，卷积核参数不是现场训练出来的，而是来自官方 ImageNet 预训练权重。现场只拿它做前向推理和特征提取。

## 32. 全连接层是什么？和分类头有什么关系？

全连接层也叫：

```text
Fully Connected layer
Linear layer
线性层
```

卷积层主要负责从图像局部提取特征；全连接层主要负责把已经提好的特征综合成最终分类分数。

例如 EfficientNet-B0 经过 `features + GAP` 后得到：

```text
1280 维向量
[f1, f2, f3, ..., f1280]
```

原始 ImageNet 分类头会用全连接层把 1280 维向量变成 1000 个类别分数：

```text
Linear(1280 -> 1000)
```

每一个类别分数都由全部 1280 个输入特征共同计算：

```text
第1类分数 = w1*f1 + w2*f2 + ... + w1280*f1280 + b
第2类分数 = w1*f1 + w2*f2 + ... + w1280*f1280 + b
...
第1000类分数 = w1*f1 + w2*f2 + ... + w1280*f1280 + b
```

所以叫“全连接”：

```text
每个输出类别，都连接所有输入特征。
```

torchvision 里这几个模型的分类头大致是：

```text
EfficientNet-B0:
Linear(1280 -> 1000)

MobileNetV3-small:
Linear(576 -> 1024)
Linear(1024 -> 1000)

MobileNetV3-large:
Linear(960 -> 1280)
Linear(1280 -> 1000)
```

这里：

```text
1280 / 576 / 960 是特征维度
1000 是 ImageNet-1K 分类类别数
```

本系统不用这些原始分类头，因为它们回答的是：

```text
这张图属于 ImageNet 的哪一个类别？
```

比如猫、狗、车、杯子等。

而本系统要回答的是：

```text
当前产品 ROI 更像 OK 还是 NG？
```

所以本系统流程是：

```text
ROI 图像
  -> features
  -> GAP
  -> L2 normalize
  -> embedding
  -> 和 OK/NG 样本做相似度比较
```

不是：

```text
ROI 图像
  -> ImageNet classifier
  -> 1000 类分类结果
```

## 33. ImageNet-1K 是什么意思？EfficientNet-B0 为什么选 B0？

`ImageNet-1K 分类模型` 的意思是：

```text
这个模型原来是在 ImageNet 数据集的 1000 个类别上训练出来的分类模型。
```

拆开看：

```text
ImageNet = 一个大规模图像分类数据集
1K = 1000 个类别
分类模型 = 输入一张图片，输出它属于每个类别的分数
```

这些类别是常见物体，例如：

```text
猫
狗
鸟
汽车
飞机
杯子
键盘
椅子
```

所以 ImageNet-1K 分类头最后输出 1000 个数字：

```text
[第1类分数, 第2类分数, ..., 第1000类分数]
```

EfficientNet-B0、MobileNetV3-small、MobileNetV3-large 在 torchvision 里加载默认 ImageNet 预训练分类权重时，最后都是 ImageNet-1K 的 1000 类输出。

但本系统不用这个 1000 类结果，只借用它前面的通用视觉特征提取能力。

EfficientNet 除了 B0，还有：

```text
efficientnet_b0
efficientnet_b1
efficientnet_b2
efficientnet_b3
efficientnet_b4
efficientnet_b5
efficientnet_b6
efficientnet_b7

efficientnet_v2_s
efficientnet_v2_m
efficientnet_v2_l
```

一般规律：

```text
B0 最小、最快、内存占用最低
B1~B7 越往后模型越大，理论精度可能更高，但速度更慢
V2 是后来的改进版本
```

本系统选 B0 的原因主要是工程权衡：

- ROI 检测通常要频繁运行，B0 更快
- B0 的 1280 维 embedding 已经能表达很多外观特征
- CPU 或普通工控机上更容易部署
- 当前系统不重新训练 CNN，只做特征提取和相似度比较，没有必要一开始就用很大的模型
- B0 在速度、效果、部署成本之间比较稳

一句话：

> B0 不是最强的 EfficientNet，但它是最轻量、最适合工程部署的 EfficientNet 基础版本。

## 34. 给初学者解释这套 CNN embedding 流程的推荐说法

如果面对完全不熟悉 CNN 的人，可以按下面顺序讲：

```text
1. CNN 是一类会从图像中提取视觉特征的网络。
2. EfficientNet-B0 是 CNN 的一种具体结构。
3. PyTorch / torchvision 把 EfficientNet-B0 组织成 features、avgpool、classifier 三大部分。
4. features 负责看图和提特征；classifier 是原来 ImageNet 的 1000 类分类头。
5. 本系统不用 classifier，因为我们不是要判断猫狗车，而是判断 OK/NG。
6. 系统会完整运行 features[0] 到 features[8]，最后取 features[8] 的输出。
7. features[8] 不是第 8 层，而是 features 列表里的最后一个顶层模块。
8. 最后特征图例如是 1280 x 7 x 7 或 1280 x 8 x 8。
9. GAP 把每个通道的空间响应求平均，得到 1280 个数。
10. L2 归一化后得到 1280 维 embedding。
11. 系统拿这个 embedding 和 OK/NG 样本特征比较相似度，输出 OK 或 NG。
```

一句汇报版：

> 本系统使用 torchvision 官方 ImageNet 预训练 CNN 作为固定 backbone。以 EfficientNet-B0 为例，系统会完整执行 `model.features[0]~model.features[8]`，取 classifier/head 之前最后一个特征模块的输出，而不是使用 ImageNet 的 1000 类分类结果。该特征图经过 GAP 和 L2 归一化后变成 1280 维 embedding，再与当前产品注册的 OK/NG embedding 原型计算相似度完成判断。
