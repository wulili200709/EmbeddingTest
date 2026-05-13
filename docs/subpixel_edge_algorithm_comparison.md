# HALCON STDC007 与当前高阶亚像素找边算法说明

本文说明两部分内容：

1. `STDC_007_GapSizeWithReflexionArea.h` 中 HALCON 对间隙边缘和尺寸的处理流程。
2. 当前项目新增的 `find_line_subpix` 高阶亚像素找线算法的处理流程。

结论先说清楚：当前项目没有直接调用 HALCON，所以 `find_line_subpix` 不是 HALCON `EdgesSubPix("shen")` 的逐行复刻。它是按同一类思路实现的 OpenCV/Numpy 版本：方向导数响应、平滑、亚像素峰值定位、连续边缘段筛选、鲁棒直线拟合。原来的 `find_line` 保留不变。

## 一、坐标和术语

HALCON 里常用：

- `row`：图像行坐标，相当于 Python 里的 `y`。
- `column`：图像列坐标，相当于 Python 里的 `x`。
- `XLD`：HALCON 的亚像素轮廓对象，轮廓点不是整数像素坐标，而是浮点坐标。
- `HRegion`：区域，也就是搜索域或 ROI。
- `HImage.ReduceDomain(region)`：只在指定区域内处理图像。

当前 Python 项目里常用：

- `x, y`：OpenCV/Numpy 坐标，`x` 是列，`y` 是行。
- `mask`：ROI 掩码，非 ROI 区域不参与找边。
- `edge_points`：找出来的亚像素边缘点。
- `FittedLine(vx, vy, x0, y0)`：拟合直线，方向向量是 `(vx, vy)`，直线上一点是 `(x0, y0)`。

## 二、STDC007 中 HALCON 的边缘提取流程

源码主要位置：

- `STDC_007_GapSizeWithReflexionArea.h:583`：`EdgesSubPix("shen")`
- `STDC_007_GapSizeWithReflexionArea.h:590`：`SegmentContoursXld("lines")`
- `STDC_007_GapSizeWithReflexionArea.h:605`：`UnionCollinearContoursExtXld(...)`
- `STDC_007_GapSizeWithReflexionArea.h:899`、`:915`：`FitLineContourXld("tukey", ...)`
- `STDC_007_GapSizeWithReflexionArea.h:1189` 到 `:1272`：两条线之间的尺寸计算

### 1. 生成搜索区域

HALCON 先根据产品目标位置、间隙高度、图像边界等信息生成搜索区域。

典型代码逻辑：

```cpp
HRegion SearchArea = HRegion::GenRectangle2(...);
HImage hiPImg = poHImg->getImg()->ReduceDomain(SearchArea);
```

含义：

- 只在搜索区域内找边。
- 搜索区域外即使有强边缘，也不会进入后续计算。
- 这样可以减少误检，也能减少计算量。

当前项目对应的是：

- `_shape_from_labels(...)` 找到指定 ROI。
- `_crop_from_shape(...)` 从图像中裁剪 ROI。
- 同时生成 `mask`，只允许 ROI 内像素参与找边。

### 2. 根据亮度选择边缘参数

STDC007 会根据间隙反光亮度 `RefGapBrightness` 选择不同的边缘参数：

```cpp
Alpha = AlphaValueStep1 / Step2 / Step3 / Default;
Low   = LowValueStep1 / Step2 / Step3 / Default;
High  = HighValueStep1 / Step2 / Step3 / Default;
```

也可以强制使用固定参数：

```cpp
if (ConstValueGapSearch)
{
    Alpha = AlphaValueConst;
    Low = LowValueConst;
    High = HighValueConst;
}
```

含义：

- 亮度不同，边缘强度不同。
- 亮背景、暗背景、反光区域需要不同的阈值。
- `Alpha` 控制平滑强度。源码注释里写得很清楚：小值会带来更强平滑，也会损失更多图像细节。
- `Low` 和 `High` 是滞后阈值，类似高低双阈值。强边缘先被保留，弱边缘只有和强边缘连通时才保留。

当前项目对应的是：

- `edge_threshold`：当前只有一个边缘响应阈值。
- `blur_ksize`：可选高斯平滑。
- `polarity`：控制暗到亮、亮到暗或任意边缘。
- 目前没有根据图像亮度自动切换多组参数。

### 3. EdgesSubPix("shen") 提取亚像素 XLD 轮廓

STDC007 使用：

```cpp
hxa = himgDomFB.ReduceDomain(hrDomArea).EdgesSubPix(
    "shen",
    Alpha,
    Low,
    High
);
```

这一步是 HALCON 的核心。

它的输出不是普通二值边缘图，而是 `HXLDContArray`，也就是一组亚像素轮廓。每个轮廓由一串浮点坐标点组成。

从效果上看，这一步完成了：

1. 在 ROI 内计算边缘响应。
2. 用 `shen` 算子进行平滑和边缘定位。
3. 用高低阈值做滞后连接。
4. 输出连续的 XLD 亚像素轮廓。

和普通 Canny 的差别：

- Canny 常见输出是像素级二值边缘图。
- `EdgesSubPix` 直接输出亚像素轮廓。
- XLD 轮廓保留了连续性，后续可以直接按轮廓长度、角度、形状来筛选。

### 4. SegmentContoursXld 把弯曲轮廓拆成线段

STDC007 使用：

```cpp
hxa = hxa.SegmentContoursXld(
    "lines",
    5,
    2.0,
    3.0
);
```

含义：

- 输入是一组可能弯曲、断裂、复杂的 XLD 轮廓。
- 输出是更接近直线的线段轮廓。
- `"lines"` 表示按线段模型切分。
- `5` 是轮廓平滑范围。
- `2.0`、`3.0` 是轮廓点到近似直线的最大允许距离，分两个阶段控制切分。

为什么要做这一步：

- 一个边缘轮廓可能包含弯角、毛刺、局部缺口。
- 直接拟合整条轮廓，容易被弯曲部分带偏。
- 先拆成近似直线段，再筛选符合方向和长度的线段，结果更稳定。

当前项目对应的是：

- 不是完整 XLD 轮廓切分。
- 当前 `find_line_subpix` 每条扫描线只取一个峰值点，然后用连续性规则把明显跳变的点分段。
- `_filter_subpixel_edge_runs(...)` 会保留最长连续边缘段。

也就是说：HALCON 是“先得到轮廓，再做线段分割”；当前实现是“先按扫描线得到边缘点，再做简单连续段筛选”。

### 5. SelectContoursXld 按长度过滤

STDC007 多次用长度过滤：

```cpp
hxa = hxa.SelectContoursXld(
    "contour_length",
    MmToPixel(0.10),
    9999,
    0.5,
    0.5
);
```

后面还会再过滤一次：

```cpp
hxa = hxa.SelectContoursXld(
    "contour_length",
    MmToPixel(TargetGapHeight / 2.5),
    9999,
    0.5,
    0.5
);
```

含义：

- 太短的边缘线段很可能是噪声、反光毛刺或局部纹理。
- 第一次较低门槛用于去掉小碎边。
- 合并共线线段后，再用更高门槛保留真正的长边。

当前项目对应的是：

- `min_points`：边缘点数量少于该值则认为找线失败。
- `_filter_subpixel_edge_runs(...)`：如果最长连续段点数达到 `min_points`，就用最长段。

当前实现没有按实际毫米长度过滤，只按点数和连续性判断。

### 6. UnionCollinearContoursExtXld 合并共线轮廓

STDC007 使用：

```cpp
hxa = hxa.UnionCollinearContoursExtXld(
    MmToPixel(TargetGapHeight / 3),
    MaxDistanceCloseEdges,
    MaxDistanceCloseEdges,
    RAD(5),
    10.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    "attr_keep"
);
```

这一步用于把断裂但共线的线段合并。

关键判断条件包括：

- 端点沿参考直线方向的距离不能太大。
- 线段到参考回归线的距离不能太大。
- 两条线段方向差不能太大，这里是 `RAD(5)`，也就是约 5 度。
- 重叠范围不能太大。
- 合并成本不能超过阈值。

为什么重要：

- 金属边、反光边、黑白过渡边经常会断裂。
- 如果不合并，可能只拿到一截短边。
- 合并后得到的边更长，直线拟合更稳定。

当前项目对应的是：

- 没有做完整的共线轮廓合并。
- 当前只按扫描顺序分连续段，保留最长段。
- 如果边缘中间有大缺口，当前实现不会像 HALCON 一样智能地把两段共线边合并成一条。

### 7. 按形状、角度和方向筛选边缘

STDC007 继续做多种筛选。

#### 7.1 剔除弯曲线

```cpp
HTuple circ = hxa.CircularityXld();
hxa = hxa.SelectShapeXld("circularity", "and", 0, MaxCircularity);
```

含义：

- 弯曲程度太大的轮廓不要。
- 间隙边缘应当接近直线。

#### 7.2 按方向剔除错误边

垂直间隙时，保留接近垂直的边：

```cpp
HXLDContArray hxaPhiN = hxa.SelectShapeXld("rect2_phi", "and", -2.0, -1.0);
HXLDContArray hxaPhiP = hxa.SelectShapeXld("rect2_phi", "and", 1.0, 2.0);
hxa = hxaPhiN.Append(hxaPhiP);
```

水平间隙时，保留接近水平的边：

```cpp
hxa = hxa.SelectShapeXld("rect2_phi", "and", -1.0, 1.0);
```

含义：

- `rect2_phi` 是最小外接旋转矩形的角度。
- 间隙方向已知，所以可以剔除方向不对的边。

#### 7.3 剔除锯齿线

```cpp
hxa = hxa.SelectShapeXld("rect2_len2", "and", 0, 3.3);
```

含义：

- 轮廓宽度或短轴太大，说明可能不是干净边缘，而是锯齿、斑块或宽区域。
- 这样的边用于尺寸测量会不稳定。

当前项目对应的是：

- 主要依靠 ROI、扫描方向、极性、阈值、最长连续段和鲁棒拟合。
- 没有完整的 `circularity`、`rect2_phi`、`rect2_len2` 多条件筛选。
- 当前的方向控制在 `direction` 参数里，比如 `left_right`、`right_left`、`top_down`、`bottom_up`。

### 8. 选择真正的左右两条内侧边

STDC007 对垂直间隙会按 `DomCenterX` 把轮廓分到左右两侧：

```cpp
if (DomCenterX < column[i].I())
    hxaTempRight.Append(hxa[i]);
else
    hxaTempLeft.Append(hxa[i]);
```

然后：

- 右侧边里选最靠左的一条，也就是右侧的内边。
- 左侧边里选最靠右的一条，也就是左侧的内边。

这样做的目的：

- 每一侧可能有外边、内边、反光边、毛刺边。
- 真正要测的是间隙宽度，所以要选两条“内侧边”。

当前项目对应的是：

- 用户手动或自动生成两个 ROI。
- 左 ROI 用一个找线工具，右 ROI 用另一个找线工具。
- 每个 ROI 通过扫描方向决定取哪条边。
- 例如左边 ROI 设置 `left_right`，右边 ROI 设置 `right_left`，就能分别拿到两个内侧边。

当前实现把“选内侧边”的责任交给 ROI 和扫描方向，而不是在一个大区域里自动分类左右边。

### 9. FitLineContourXld("tukey") 鲁棒拟合直线

STDC007 对两条边分别拟合：

```cpp
RowBegin0 = hxa[0].FitLineContourXld(
    "tukey",
    -1,
    10,
    5,
    2.0,
    &ColBegin0,
    &RowEnd0,
    &ColEnd0,
    __nullptr,
    __nullptr,
    __nullptr
);
```

第二条边同理：

```cpp
RowBegin1 = hxa[1].FitLineContourXld("tukey", ...);
```

含义：

- 输入是亚像素 XLD 轮廓。
- 输出是拟合直线的两个端点：
  - `(ColBegin0, RowBegin0)`
  - `(ColEnd0, RowEnd0)`
- `"tukey"` 是鲁棒拟合方法，对离群点降低权重。
- 后面的参数控制裁剪、迭代、距离阈值等。

为什么用鲁棒拟合：

- 边缘上可能有毛刺、缺口、反光点。
- 普通最小二乘会被离群点拉偏。
- Tukey 权重会让离群点影响变小。

当前项目对应的是：

```python
cv2.fitLine(pts, cv2.DIST_WELSCH, 0, 0.01, 0.01)
```

并且多做了一步：

1. 先拟合一次。
2. 计算所有点到直线的距离。
3. 用 MAD 鲁棒统计剔除明显离群点。
4. 再拟合一次。

当前用的是 Welsch 鲁棒距离，不是 Tukey。两者都属于降低离群点影响的鲁棒拟合，但权重函数不同。

## 三、STDC007 中两条不平行线的尺寸计算方式

STDC007 不是直接算“两条无限直线的最短距离”。

它的尺寸计算流程是：

1. 先得到两条拟合线：
   - 线 0：`(ColBegin0, RowBegin0)` 到 `(ColEnd0, RowEnd0)`
   - 线 1：`(ColBegin1, RowBegin1)` 到 `(ColEnd1, RowEnd1)`
2. 判断哪条 XLD 轮廓更长。
3. 用更长的那条边作为参考边。
4. 沿参考边分成 `CountGapMultiSize` 个测量位置。
5. 在每个测量位置，构造一条垂直于参考边的测量线。
6. 用这条测量线分别和两条拟合边求交点。
7. 两个交点之间的欧氏距离就是该位置的间隙宽度。
8. 转成毫米后和上下限比较。

源码中关键逻辑：

```cpp
hom_mat2d_rotate(HomMat2D, DegToRad(90), y, xCen, &HomMat2D);
affine_trans_pixel(... RowBegin0, ColBegin0 ... &transY1, &transX1);
affine_trans_pixel(... RowEnd0, ColEnd0 ... &transY2, &transX2);

IntersectionLL(ColBegin0, RowBegin0, ColEnd0, RowEnd0,
               transX1, transY1, transX2, transY2,
               XGabelpos0, YGabelpos0);

IntersectionLL(ColBegin1, RowBegin1, ColEnd1, RowEnd1,
               transX1, transY1, transX2, transY2,
               XGabelpos1, YGabelpos1);

Value = PixelToMm(DistancePP(XGabelpos0, YGabelpos0, XGabelpos1, YGabelpos1));
```

这和“无限直线最短距离”的区别很重要：

- 如果两条线不平行，数学上的无限直线最短距离是 0，因为它们最终会相交。
- 实际卡尺测量不是求无限直线的最短距离。
- 实际卡尺测量是在某个测量位置，沿垂直于参考边的方向量两个面的间距。
- STDC007 用的就是这种“参考边法线方向上的交点距离”。

所以它更符合“卡尺量两个面宽度”的实际含义。

## 四、当前项目原 `find_line` 的算法流程

原算法仍然保留，代码入口是 `algorithms/measurement.py` 的 `find_edge_points(...)`，当 `edge_detector == "canny"` 时走旧逻辑。

流程如下。

### 1. ROI 裁剪和 mask

```python
crop, mask, origin = _crop_from_shape(image_bgr, shape)
```

含义：

- 根据 ROI 取出局部图。
- 生成同尺寸 mask。
- 只允许 ROI 内像素参与后续判断。

### 2. 灰度化和可选模糊

```python
gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
if config.blur_ksize >= 3:
    gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
```

### 3. Canny 得到像素级候选边缘

```python
edges = cv2.Canny(gray_u8, canny_low, canny_high, L2gradient=True)
```

这里 Canny 只用于生成候选点。真正的位置还会用灰度梯度做亚像素细化。

### 4. 计算方向导数响应

如果是左右找边：

```python
delta = gray[:, 1:] - gray[:, :-1]
```

如果是上下找边：

```python
delta = gray[1:, :] - gray[:-1, :]
```

然后根据方向和极性统一响应：

```python
response = _edge_response(delta, polarity, direction=direction)
```

例如：

- `dark_to_bright`：保留正梯度。
- `bright_to_dark`：保留负梯度。
- `any`：取绝对值。
- `right_left` 或 `bottom_up` 会先把梯度方向反过来。

### 5. 按扫描线找第一个满足条件的边

例如 `left_right`：

- 每隔 `scan_step` 行扫描一次。
- 每行从左到右找。
- 必须同时满足：
  - 当前像素在 ROI mask 内。
  - Canny 有边缘。
  - 相邻像素也在 ROI 内。
  - 灰度导数响应超过 `edge_threshold`。

找到第一个候选边后，这一行就停止。

这个设计的作用：

- 一个 ROI 内可能有多条边。
- 扫描方向决定取哪条边。
- 左 ROI 通常从左到右取第一个边；右 ROI 通常从右到左取第一个边。

### 6. 抛物线亚像素细化

旧算法不是只用 Canny 的整数像素点。

找到候选点后，会在局部灰度响应中重新找最大梯度点，并做三点抛物线插值：

```python
offset = 0.5 * (left - right) / (left - 2 * center + right)
```

最终位置：

```python
x_sub = best + 1 + offset
```

其中 `offset` 被限制在 `[-1, 1]`。

所以原 `find_line` 也有亚像素细化，只是它依赖 Canny 先给出候选边缘。

### 7. 鲁棒直线拟合

旧算法和新算法共用 `fit_line_filtered(...)`：

1. 用 `cv2.fitLine(... DIST_WELSCH ...)` 初次拟合。
2. 计算点到线距离。
3. 用中位数和 MAD 估计离群阈值。
4. 剔除明显离群点。
5. 再拟合一次。

## 五、当前新增 `find_line_subpix` 的算法流程

新增算法入口仍然是 `measure_find_line_from_array(...)`，但传入：

```json
"algorithm_code": "find_line_subpix"
```

默认参数中会设置：

```json
"edge_detector": "subpix_shen"
```

代码入口：

- `algorithms/measurement.py:432`：`_find_subpixel_edge_points(...)`
- `algorithms/measurement.py:481`：`find_edge_points(...)` 根据 `edge_detector` 分发

### 1. ROI 裁剪和 mask

和旧算法一样：

```python
crop, mask, origin = _crop_from_shape(image_bgr, shape)
```

这保证：

- 找边只在 ROI 内进行。
- ROI 外强边缘不会参与。

### 2. 灰度化和可选高斯平滑

```python
gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
if config.blur_ksize >= 3:
    gray = cv2.GaussianBlur(gray, (config.blur_ksize, config.blur_ksize), 0)
```

这一步对应 HALCON `EdgesSubPix("shen")` 中的平滑思想，但不是 HALCON 的同一个滤波器。

### 3. 计算方向导数

左右找边：

```python
delta = gray[:, 1:] - gray[:, :-1]
```

上下找边：

```python
delta = gray[1:, :] - gray[:-1, :]
```

这一步得到边缘强度的基础响应。灰度变化越明显，`delta` 越大。

### 4. 对导数响应做高阶平滑

新增算法不用 Canny。它直接对导数响应做平滑：

```python
kernel = [1, 4, 6, 4, 1] / 16
filtered_delta = filter2D(delta, kernel)
```

这个核是 5 点二项式平滑核。

作用：

- 抑制单点噪声。
- 让边缘响应峰更平滑。
- 便于后续做抛物线亚像素定位。

为什么这里叫“高阶亚像素”：

- 它不是先用 Canny 得到整数边，再局部修正。
- 它直接在平滑后的连续响应峰上定位。
- 后续坐标是浮点值，能得到小数像素位置。

需要注意：

- 这不是 HALCON `EdgesSubPix("shen")` 的完整内部实现。
- 它是当前项目里不依赖 HALCON 的近似实现。

### 5. 根据方向和极性生成响应

```python
response = _edge_response(filtered_delta, config.polarity, direction=config.direction)
```

规则：

- `left_right`：按原方向响应。
- `right_left`：响应取反后再判断。
- `top_down`：按原方向响应。
- `bottom_up`：响应取反后再判断。
- `dark_to_bright`：只接受暗到亮边。
- `bright_to_dark`：只接受亮到暗边。
- `any`：接受任意极性，用绝对值。

这一步对你之前说的“ROI 方向改变后极性也要跟着变”很关键。方向变化会改变导数符号，所以响应必须结合方向和极性一起计算。

### 6. 找局部峰值，而不是找 Canny 边缘点

新增算法的候选条件是：

```python
center >= edge_threshold
center >= left
center >= right
```

也就是：

- 响应值超过阈值。
- 当前点是局部最大值。
- 当前点和相邻点在 ROI mask 内。

它不再要求：

```python
edges[y, x] > 0
```

因为新算法没有 Canny 这一步。

### 7. 每条扫描线只取第一个边缘峰

例如 `left_right`：

- 每隔 `scan_step` 行扫描。
- 每一行从左往右找第一个局部峰。
- 找到后立即停止这一行。

例如 `right_left`：

- 每一行从右往左找第一个局部峰。

这样可以避免一个 ROI 内有多个边缘时取错。

### 8. 抛物线亚像素定位

找到局部峰值后，使用三点抛物线插值：

```python
offset = 0.5 * (left - right) / (left - 2 * center + right)
```

最终坐标：

```python
coord_sub = index + 1 + offset
```

含义：

- `index + 1` 是因为导数 `gray[x+1] - gray[x]` 的边缘位置在两个像素之间。
- `offset` 是峰值相对整数位置的亚像素偏移。
- 如果响应左右不对称，峰值位置会偏向响应更强的一侧。

得到的边缘点可能是：

```text
(35.846, 10.000)
(35.846, 12.000)
(35.846, 14.000)
```

而不是：

```text
(36, 10)
(36, 12)
(36, 14)
```

### 9. 连续边缘段筛选

新增算法用 `_filter_subpixel_edge_runs(...)` 做简化版轮廓筛选。

对于左右找边：

- 主轴是 `y`，也就是扫描行。
- 副轴是 `x`，也就是边缘位置。

如果相邻点满足下面任意条件，就认为断开：

```python
abs(primary_gap) > scan_step * 2.5
abs(secondary_jump) > max(8, scan_step * 8)
```

然后：

- 把点拆成多个连续段。
- 选择最长连续段。
- 如果最长段点数不少于 `min_points`，就只用最长段。
- 如果最长段不够，就退回使用全部点，让后续拟合决定。

这一步对应 HALCON 的：

- `SegmentContoursXld("lines")`
- `SelectContoursXld("contour_length")`

但它比 HALCON 简化很多。

### 10. 鲁棒直线拟合

新增算法和旧算法共用：

```python
fit_line_filtered(points, min_points=...)
```

过程：

1. 初次拟合：

   ```python
   cv2.fitLine(pts, cv2.DIST_WELSCH, 0, 0.01, 0.01)
   ```

2. 得到直线：

   ```text
   x = x0 + t * vx
   y = y0 + t * vy
   ```

3. 计算点到直线距离：

   ```python
   distance = abs(vy * (x - x0) - vx * (y - y0))
   ```

4. 用中位数和 MAD 估计离群：

   ```python
   robust_sigma = 1.4826 * median(abs(distance - median(distance)))
   threshold = max(2.0, median + 3.0 * robust_sigma)
   ```

5. 剔除距离过大的点。
6. 再拟合一次。

输出：

- `line.vx, line.vy`：直线方向。
- `line.x0, line.y0`：直线上的点。
- `line.residual`：平均残差。
- `line.point_count`：参与最终拟合的点数。

## 六、当前距离测量如何使用新找线结果

`find_line_subpix` 只负责找边和拟合直线。

距离测量仍然由距离工具完成：

- `line_distance`
- `line_distance_ref_normal`，界面显示为“卡尺距离测量”

如果左右两个找线工具改成：

```json
"algorithm_code": "find_line_subpix"
```

那么距离工具引用它们时，就会使用高阶亚像素找线结果。

示例：

```json
[
  {
    "item_id": "left",
    "display_name": "左",
    "camera_id": "cam1",
    "roi_label": "roi1",
    "algorithm_code": "find_line_subpix",
    "enabled": true,
    "params": {
      "line": {
        "direction": "left_right",
        "edge_detector": "subpix_shen",
        "edge_threshold": 10,
        "scan_step": 2,
        "min_points": 10
      }
    },
    "algorithm_type": "find_line_subpix"
  },
  {
    "item_id": "right",
    "display_name": "右",
    "camera_id": "cam1",
    "roi_label": "roi2",
    "algorithm_code": "find_line_subpix",
    "enabled": true,
    "params": {
      "line": {
        "direction": "right_left",
        "edge_detector": "subpix_shen",
        "edge_threshold": 10,
        "scan_step": 2,
        "min_points": 10
      }
    },
    "algorithm_type": "find_line_subpix"
  },
  {
    "item_id": "line_distance",
    "display_name": "卡尺距离测量",
    "camera_id": "cam1",
    "algorithm_code": "line_distance_ref_normal",
    "enabled": true,
    "params": {
      "line_a_item_id": "left",
      "line_b_item_id": "right",
      "limit_unit": "mm",
      "pixel_size_mm": 0.006912,
      "lower_limit": 6.650,
      "upper_limit": 6.750
    },
    "algorithm_type": "line_distance_ref_normal"
  }
]
```

## 七、HALCON 方式与当前实现的差异

| 项目 | HALCON STDC007 | 当前 `find_line_subpix` |
|---|---|---|
| 边缘算子 | `EdgesSubPix("shen")` | 平滑方向导数 + 局部峰值 |
| 输出 | XLD 亚像素轮廓 | 亚像素边缘点数组 |
| 阈值 | Low/High 双阈值滞后 | 单一 `edge_threshold` |
| 平滑参数 | `Alpha` | `blur_ksize` + 5 点平滑核 |
| 轮廓连续性 | HALCON XLD 原生连续轮廓 | 按扫描线点序做连续段筛选 |
| 线段切分 | `SegmentContoursXld("lines")` | 无完整线段切分，保留最长连续段 |
| 共线合并 | `UnionCollinearContoursExtXld` | 暂无完整共线合并 |
| 形状过滤 | 长度、圆度、角度、短轴等 | 点数、方向、连续性、鲁棒拟合 |
| 拟合 | `FitLineContourXld("tukey")` | `cv2.fitLine(DIST_WELSCH)` + MAD 二次过滤 |
| 距离 | 参考边法线方向交点距离，可多点测量 | 距离工具复用拟合线，`line_distance_ref_normal` 接近参考法线测量 |
| 是否依赖 HALCON | 是 | 否 |

## 八、为什么当前实现仍然比原 Canny 版本更接近 STDC007

原 `find_line` 的流程是：

```text
Canny 整数候选边缘 -> 灰度梯度局部细化 -> 拟合直线
```

新增 `find_line_subpix` 的流程是：

```text
方向导数 -> 平滑响应 -> 局部峰值 -> 亚像素点 -> 连续段筛选 -> 拟合直线
```

它更接近 STDC007 的原因：

1. 不依赖 Canny 的整数边缘图。
2. 直接在边缘响应峰上做亚像素定位。
3. 会保留一串亚像素边缘点用于拟合。
4. 会做连续边缘段筛选。
5. 拟合阶段使用鲁棒方法。

但它仍然不是 HALCON 完整复刻，差异主要在：

1. 没有 HALCON XLD 的完整轮廓拓扑。
2. 没有 `Low/High` 双阈值滞后连接。
3. 没有完整的共线轮廓合并。
4. 没有完整的形状属性筛选。
5. 拟合权重是 Welsch，不是 Tukey。

## 九、实际调参建议

### 0. 扫描线是什么

扫描线就是在 ROI 里面按固定方向一行一行或一列一列地找边。

如果要找竖直边，ROI 通常是一个竖长矩形，算法会按行扫描：

```text
ROI
┌──────────┐
│  →       │  第 1 条扫描线
│  →       │  第 2 条扫描线
│  →       │  第 3 条扫描线
│  →       │
│  →       │
└──────────┘
```

如果方向是 `left_right`，每一行从左往右找第一个边缘点。

如果方向是 `right_left`，每一行从右往左找第一个边缘点：

```text
ROI
┌──────────┐
│       ←  │
│       ←  │
│       ←  │
│       ←  │
└──────────┘
```

如果要找水平边，扫描线通常就是一列一列：

```text
ROI
┌──────────┐
│ ↓ ↓ ↓ ↓  │  top_down
│          │
│          │
└──────────┘
```

所以：

- 找竖直边：扫描线通常是横着的一行，沿 X 方向找边。
- 找水平边：扫描线通常是竖着的一列，沿 Y 方向找边。
- `scan_step=2` 表示每隔 2 行或 2 列取一条扫描线。
- 每条扫描线一般只取一个边缘点。
- 很多条扫描线得到很多边缘点，最后用这些点拟合一条直线。

左右两个竖直 ROI 的典型情况是：

- 左 ROI：逐行从左往右扫，找到左侧目标边。
- 右 ROI：逐行从右往左扫，找到右侧目标边。
- 两边各得到一串点。
- 两串点分别拟合成两条直线。
- 距离工具再根据两条直线计算尺寸。

### 1. ROI 要尽量只框住目标边

新算法每条扫描线取第一个峰值。ROI 内如果有多个强边缘，方向选错或 ROI 太宽都会导致取错边。

建议：

- 左边 ROI 框住左内侧边。
- 右边 ROI 框住右内侧边。
- ROI 尽量避开反光毛刺和背景强边。

### 2. 方向要按“从 ROI 外侧扫向目标边”设置

例如测两个内侧面：

- 左 ROI：通常 `left_right`
- 右 ROI：通常 `right_left`

如果方向反了，算法可能会先遇到 ROI 里的另一条边。

### 3. 极性要和图像亮暗变化一致

如果从扫描方向看是黑到白，设置：

```json
"polarity": "dark_to_bright"
```

如果从扫描方向看是白到黑，设置：

```json
"polarity": "bright_to_dark"
```

如果不确定，可以先用：

```json
"polarity": "any"
```

但 `any` 在有反光或多边缘时更容易取错。

### 4. `edge_threshold`

阈值太低：

- 容易吃到纹理、噪声、反光。

阈值太高：

- 边缘弱时找不到足够点。

建议从 `10` 开始，根据图像边缘强度调整。

### 5. `scan_step`

`scan_step` 越小：

- 点越多。
- 拟合更稳定。
- 计算稍慢。

`scan_step` 越大：

- 点更少。
- 对局部缺陷更敏感。

尺寸测量建议优先用 `1` 或 `2`。

### 6. `min_points`

点数太低：

- 短边、噪声也可能拟合成功。

点数太高：

- ROI 较短或边缘有断裂时容易失败。

建议：

- ROI 高度较大时用 `10` 到 `30`。
- ROI 很短时适当降低。

## 十、后续如果要更接近 HALCON，可以继续增强

当前版本已经能做到亚像素找边和鲁棒拟合。如果要更接近 STDC007，可以继续加这些功能：

1. 双阈值滞后连接，替代单阈值。
2. 生成真正的连续边缘链，而不是每条扫描线一个点。
3. 增加共线线段合并，类似 `UnionCollinearContoursExtXld`。
4. 增加形状过滤：
   - 长度
   - 角度
   - 弯曲度
   - 短轴宽度
5. 拟合方法增加 Tukey 权重。
6. `line_distance_ref_normal` 增加多点测量，类似 STDC007 的 `CountGapMultiSize`，最后可以取均值、中位数或逐点判定。

对当前“卡尺量两个面宽度”的需求来说，优先级最高的是：

1. ROI 和方向稳定。
2. 用 `find_line_subpix` 找两条边。
3. 用 `line_distance_ref_normal` 做参考法线方向距离。
4. 如果需要进一步抗局部毛刺，再加多点法线测量。

## 十一、为什么“方向导数 -> 平滑响应 -> 局部峰值 -> 亚像素点”更接近高阶亚像素

原 `find_line` 更像这个流程：

```text
图像灰度 -> Canny 判断哪个整数像素是边 -> 在这个整数像素附近微调
```

新增 `find_line_subpix` 更像这个流程：

```text
图像灰度 -> 计算连续灰度变化趋势 -> 找变化最强的位置 -> 直接用峰值曲线算小数像素位置
```

区别在于：新算法不是先把边缘压成整数像素点，而是在平滑后的边缘响应曲线上找峰值，再计算峰顶的小数位置。

### 1. 方向导数为什么能表示边缘

边缘本质上是灰度变化最快的位置。

例如某条扫描线上的灰度：

```text
灰度：  10  12  15  80  180  220
变化：    2   3  65 100   40
```

`变化` 就是相邻像素灰度差，也就是离散意义上的方向导数。

其中变化最大的地方，就是边缘附近。

### 2. 平滑响应为什么重要

真实图片里有噪声、纹理、反光。如果直接看导数，响应可能很抖：

```text
没平滑：  2  9  4  80  30  75  8
```

平滑以后，边缘响应会更像一个连续峰：

```text
平滑后：  3  8  22 55  62  40  15
```

这样做的好处：

- 单点噪声会被压低。
- 真正边缘的响应峰更稳定。
- 后续用三点抛物线估计峰顶会更合理。

这就是它更接近 HALCON `EdgesSubPix("shen")` 思路的原因：不是只判断某个像素是不是边，而是先得到平滑的边缘响应，再在响应上定位边缘。

### 3. 局部峰值为什么对应边缘中心

边缘不是一个无限薄的点。真实图像里，从黑到白通常会跨过几个像素：

```text
灰度：  10  20  45  90  150  205  230
```

它的变化量可能是：

```text
变化：   10  25  45  60   55   25
```

最大变化处就是边缘过渡最剧烈的位置，通常可视为边缘中心。

所以新算法找的是响应的局部最大值：

```text
center >= left
center >= right
center >= edge_threshold
```

### 4. 为什么能得到亚像素点

整数像素只能表示：

```text
边缘在 x=35 附近
```

但响应峰顶不一定正好落在 `35`，可能在 `35.17` 或 `34.82`。

如果知道峰值附近三个点的响应：

```text
像素位置： 34    35    36
响应值：   L     C     R
```

就可以用三点抛物线插值估计真正峰顶位置。

最终结果就是小数像素位置：

```text
x = 35 + offset
```

这个 `offset` 就是亚像素偏移。

### 5. 为什么这还不能完全等同 HALCON

虽然流程更接近高阶亚像素，但当前实现仍然不是 HALCON `EdgesSubPix("shen")` 的完整复刻。

HALCON 还有：

- XLD 连续轮廓结构。
- Low/High 双阈值滞后连接。
- 轮廓线段切分。
- 共线轮廓合并。
- 多种形状属性筛选。
- Tukey 鲁棒拟合。

当前实现是：

- 平滑导数响应。
- 局部峰值定位。
- 三点抛物线亚像素。
- 简化连续段筛选。
- Welsch 鲁棒拟合 + MAD 离群过滤。

所以准确说法是：当前 `find_line_subpix` 是“不依赖 HALCON 的近似高阶亚像素找边流程”，比原 Canny 候选点流程更接近 STDC007 的亚像素边缘思想。

## 十二、35.17 这种小数像素是怎么计算出来的

假设某条扫描线上，边缘响应最大点在整数像素 `35` 附近。

已知三个相邻位置的响应：

```text
像素位置： 34    35    36
响应值：   65   100    80
```

其中：

```text
left   = 65
center = 100
right  = 80
```

整数判断只能说峰值在 `35` 这个位置。

但因为右侧响应 `80` 比左侧响应 `65` 更高，说明真正峰顶会从 `35` 稍微往右偏。

代码里用：

```python
offset = 0.5 * (left - right) / (left - 2 * center + right)
```

代入：

```text
offset = 0.5 * (65 - 80) / (65 - 2*100 + 80)
       = 0.5 * (-15) / (65 - 200 + 80)
       = 0.5 * (-15) / (-55)
       = 0.136
```

所以峰值坐标是：

```text
x = 35 + 0.136
x = 35.136
```

如果响应组合略有不同，就可能得到 `35.17`：

```text
像素位置： 34    35    36
响应值：   62   100    83
```

计算：

```text
offset = 0.5 * (62 - 83) / (62 - 200 + 83)
       = 0.5 * (-21) / (-55)
       = 0.191
```

位置：

```text
x = 35.191
```

这类小数坐标就是亚像素定位结果。

## 十三、三点抛物线公式从哪里来

公式是：

```python
offset = 0.5 * (left - right) / (left - 2 * center + right)
```

它来自“三点抛物线插值求峰值位置”。

### 1. 假设峰值附近可以用抛物线近似

设三个相邻点的位置是：

```text
x = -1, 0, +1
```

对应响应值是：

```text
y(-1) = L
y(0)  = C
y(+1) = R
```

用一条抛物线表示：

```text
y = ax² + bx + c
```

### 2. 代入三个点

中间点：

```text
C = a*0² + b*0 + c
C = c
```

左边点：

```text
L = a*(-1)² + b*(-1) + c
L = a - b + c
```

右边点：

```text
R = a*(+1)² + b*(+1) + c
R = a + b + c
```

因为 `c = C`，所以：

```text
L = a - b + C
R = a + b + C
```

### 3. 解出 a 和 b

左右相加：

```text
L + R = 2a + 2C
a = (L - 2C + R) / 2
```

左右相减：

```text
R - L = 2b
b = (R - L) / 2
```

### 4. 抛物线峰值在哪里

抛物线：

```text
y = ax² + bx + c
```

导数是：

```text
y' = 2ax + b
```

峰值位置满足导数为 0：

```text
2ax + b = 0
x = -b / (2a)
```

代入 `a` 和 `b`：

```text
x = - ((R - L) / 2) / (2 * ((L - 2C + R) / 2))
```

化简：

```text
x = 0.5 * (L - R) / (L - 2C + R)
```

也就是：

```python
offset = 0.5 * (left - right) / (left - 2 * center + right)
```

### 5. offset 的意义

`offset` 表示峰值相对中间整数点的偏移：

```text
真实峰值位置 = center_pixel + offset
```

如果：

```text
left == right
```

则：

```text
offset = 0
```

说明峰值就在中间点。

如果：

```text
right > left
```

则 `offset > 0`，峰值偏右。

如果：

```text
left > right
```

则 `offset < 0`，峰值偏左。

代码里还会把 `offset` 限制在 `[-1, 1]`，防止异常响应导致峰值偏移过大。

对应代码位置：

```python
def _parabolic_peak_offset(left, center, right):
    denominator = left - 2 * center + right
    if abs(denominator) <= 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return max(-1.0, min(1.0, offset))
```

## 十四、把这些概念串起来

完整理解可以这样看：

```text
扫描线
  ↓
沿扫描方向取灰度序列
  ↓
计算相邻像素灰度差，也就是方向导数
  ↓
对导数做平滑，降低噪声
  ↓
找到响应最高的局部峰值
  ↓
用峰值左右三个点做抛物线插值
  ↓
得到 x=35.17 这种亚像素边缘点
  ↓
很多扫描线得到很多亚像素点
  ↓
筛掉不连续或离群的点
  ↓
鲁棒拟合直线
```

所以“高阶亚像素”的核心不是某一个公式，而是整条链路都尽量围绕灰度响应和浮点边缘位置来做，而不是只依赖整数像素边缘。
