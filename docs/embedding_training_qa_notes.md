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

如果一定要回答“第几层”，按 `torchvision` 的 EfficientNet-B0 结构编号，可以说：

> 对 EfficientNet-B0，本系统取的是 `model.features[8]` 的输出，也就是分类头之前最后一个卷积特征模块的输出。

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

## 7. MobileNetV3 具体是第几层？

MobileNetV3 也是同样逻辑：取 `model.features` 的最后输出。

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

汇报时可以说：

> 对 MobileNetV3-small，本系统取 `features[12]` 的输出；对 MobileNetV3-large，取 `features[16]` 的输出。它们都是分类头之前最后一个卷积特征模块的输出。576 和 960 表示特征通道数，不是网络层数。

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
