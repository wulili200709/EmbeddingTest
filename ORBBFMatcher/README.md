# ORB/BRISK + BEBLID / SIFT Python Demo

这个目录是一个独立的 Python 小项目，当前包含三套方案：

- `template_maker.py`：交互式或坐标式制作模板
- `ORB` 或 `BRISK` 提取关键点
- `BEBLID` 生成二进制描述子
- `BFMatcher(Hamming)` 完成匹配
- `SIFT` 提取关键点和浮点描述子
- `BFMatcher(L2)` 完成匹配
- `LINE / LINEMOD` 风格的 2D `shape-based` 模板匹配

当有效匹配足够时，程序会继续用 `findHomography(..., RANSAC)` 在场景图上估计模板位置，并输出可视化结果。

## 1. 使用当前目录下的虚拟环境

如果当前目录还没有虚拟环境：

```powershell
py -3.12 -m venv .venv
```

安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. 制作模板

交互方式：

```powershell
.\.venv\Scripts\python.exe .\template_maker.py `
  --image .\your_source.png `
  --output-template .\output\my_template.png
```

窗口操作：

- `p`：多边形模式
- `r`：矩形模式
- 左键拖动：框矩形
- 左键单击：添加多边形点
- 右键或 `Enter`：闭合多边形
- `u`：撤销一个多边形点
- `c`：清空当前选择
- `s`：保存
- `q` / `Esc`：退出

非交互方式：

```powershell
.\.venv\Scripts\python.exe .\template_maker.py `
  --image ..\img\template.png `
  --output-template .\output\demo_polygon_template.png `
  --polygon 5,5,121,8,122,112,10,116
```

默认输出：

- `xxx.png`：模板图；如果是 PNG，会带透明 alpha
- `xxx_mask.png`：二值 mask
- `xxx_preview.png`：模板预览
- `xxx.json`：模板元数据

不规则模板可以做。实现方式是“裁剪后的模板图 + mask”，匹配时只在 mask 区域内提取特征。

查看帮助：

```powershell
.\.venv\Scripts\python.exe .\template_maker.py --help
```

## 3. LINE / LINEMOD shape-based 示例

这个脚本使用 OpenCV `cv2.linemod.getDefaultLINE()`，适合 RGB 图像上的 2D 轮廓/梯度模板匹配。

```powershell
.\.venv\Scripts\python.exe .\linemod_shape_match.py `
  --template ..\img\template.png `
  --scene ..\img\test.png `
  --output .\output\linemod_overlay.png `
  --template-preview-output .\output\linemod_templates.png `
  --threshold 90 `
  --angles 0,-10,10 `
  --scales 1.0
```

运行后会输出：

- `output/linemod_overlay.png`：场景图上的检测结果
- `output/linemod_templates.png`：生成出的模板变体和 LINE 特征预览

常用参数：

- `--mask`：可选的模板二值 mask；不传时脚本会按浅色背景自动生成
- `--angles`：模板旋转扩增角度列表，例如 `0,-15,15`
- `--scales`：模板尺度扩增，例如 `0.9,1.0,1.1`
- `--threshold`：LINE 匹配阈值，范围 `0~100`
- `--top-k`：NMS 前最多检查多少个原始候选
- `--max-detections`：NMS 后最多保留多少个结果
- `--nms-iou`：检测框去重阈值
- `--tile-size` / `--tile-overlap`：大图整幅匹配失败时的分块搜索参数

查看帮助：

```powershell
.\.venv\Scripts\python.exe .\linemod_shape_match.py --help
```

## 4. ORB / BRISK + BEBLID 运行示例

使用当前仓库里已有图片测试，默认检测器为 `orb`：

```powershell
.\.venv\Scripts\python.exe .\orb_bf_match.py `
  --template ..\img\template.png `
  --scene ..\img\test.png `
  --output .\output\orb_beblid_matches.png `
  --overlay-output .\output\orb_beblid_overlay.png
```

切换到 `brisk`：

```powershell
.\.venv\Scripts\python.exe .\orb_bf_match.py `
  --detector brisk `
  --template ..\img\template.png `
  --scene ..\img\test.png `
  --output .\output\brisk_beblid_matches.png `
  --overlay-output .\output\brisk_beblid_overlay.png
```

运行后会输出：

- `output/*_matches.png`：模板图、场景图和匹配连线的组合可视化
- `output/*_overlay.png`：仅场景图上的检测框

如果模板来自 `template_maker.py`，可以直接使用生成的透明 PNG：

```powershell
.\.venv\Scripts\python.exe .\orb_bf_match.py `
  --template .\output\demo_polygon_template.png `
  --scene ..\img\test.png `
  --output .\output\demo_polygon_matches.png `
  --overlay-output .\output\demo_polygon_overlay.png
```

如果模板不是透明 PNG，也可以显式传 mask：

```powershell
.\.venv\Scripts\python.exe .\orb_bf_match.py `
  --template .\output\demo_polygon_template.jpg `
  --template-mask .\output\demo_polygon_template_mask.png `
  --scene ..\img\test.png `
  --output .\output\demo_polygon_matches.png `
  --overlay-output .\output\demo_polygon_overlay.png
```

## 5. ORB / BRISK 参数说明

```powershell
.\.venv\Scripts\python.exe .\orb_bf_match.py --help
```

常用参数：

- `--detector orb|brisk`：选择关键点检测器
- `--template-mask`：显式指定模板 mask；不传时会先读模板 alpha，再尝试同名 `*_mask.png`
- `--beblid-bits 256|512`：选择 BEBLID 描述子长度
- `--beblid-scale-factor`：手动覆盖 BEBLID 采样窗口尺度
- `--ratio-threshold`：Lowe ratio test 阈值
- `--min-matches`：估计单应性所需的最少有效匹配数
- `--ransac-threshold`：RANSAC 重投影误差阈值

`orb` 模式常用参数：

- `--max-features`
- `--edge-threshold`
- `--fast-threshold`

`brisk` 模式常用参数：

- `--brisk-thresh`
- `--brisk-octaves`
- `--brisk-pattern-scale`

## 6. SIFT 运行示例

SIFT 脚本沿用和 `orb_bf_match.py` 一样的模板 mask、单应性估计和可视化输出方式，只是匹配器改成了 `BFMatcher(L2)`。

```powershell
.\.venv\Scripts\python.exe .\sift_bf_match.py `
  --template .\images\2\images2_template.png `
  --scene .\images\2\Image_20260206154258679.bmp `
  --output .\output\sift_matches.png `
  --overlay-output .\output\sift_overlay.png
```

如果模板不是透明 PNG，也可以显式传 mask：

```powershell
.\.venv\Scripts\python.exe .\sift_bf_match.py `
  --template .\images\2\images2_template.png `
  --template-mask .\images\2\images2_template_mask.png `
  --scene .\images\2\Image_20260206154258679.bmp `
  --output .\output\sift_matches.png `
  --overlay-output .\output\sift_overlay.png
```

查看帮助：

```powershell
.\.venv\Scripts\python.exe .\sift_bf_match.py --help
```

常用参数：

- `--template-mask`：显式指定模板 mask；不传时会先读模板 alpha，再尝试同名 `*_mask.png`
- `--max-features`：限制 SIFT 最大关键点数量
- `--n-octave-layers`：每个 octave 的层数
- `--contrast-threshold`：降低后会检测到更多关键点
- `--edge-threshold`：提高后会保留更多边缘附近特征
- `--sigma`：初始高斯平滑参数
- `--ratio-threshold`：Lowe ratio test 阈值
- `--min-matches`：估计单应性所需的最少有效匹配数
- `--ransac-threshold`：RANSAC 重投影误差阈值

## 7. SIFT 批量匹配与性能测试

下面这个脚本会对目录内所有场景图做 SIFT 匹配，并输出每张图的细分阶段耗时：

```powershell
.\.venv\Scripts\python.exe .\sift_bf_match_batch.py `
  --template .\images\2\images2_template.png `
  --input-dir .\images\2 `
  --output-dir .\output\sift_batch_images2
```

如果只关心纯匹配性能，不想把保存图片的磁盘耗时算进去：

```powershell
.\.venv\Scripts\python.exe .\sift_bf_match_batch.py `
  --template .\images\2\images2_template.png `
  --input-dir .\images\2 `
  --output-dir .\output\sift_batch_images2_noviz `
  --no-write-visuals
```

运行后会输出：

- `summary.csv`：每张图的详细阶段耗时、关键点数、匹配数、内点数
- `summary.json`：整体成功率、各阶段平均/中位/最小/最大耗时
- `*_matches.png` / `*_overlay.png`：逐图可视化结果；使用 `--no-write-visuals` 时不会生成

阶段耗时包括：

- `scene_load_ms`
- `resize_ms`
- `gray_ms`
- `feature_ms`
- `knn_match_ms`
- `ratio_filter_ms`
- `homography_ms`
- `overlay_draw_ms`
- `save_match_ms`
- `save_overlay_ms`
- `total_ms`

## 8. 说明

- 依赖使用 `opencv-contrib-python`，因为 `BEBLID` 位于 `cv2.xfeatures2d`
- `LINE / LINEMOD` 实现也依赖 `opencv-contrib-python` 中的 `cv2.linemod`
- 对于某些高分辨率场景图，脚本会自动回退到重叠分块匹配，再把结果映射回原图坐标
- BEBLID 默认尺度会随检测器自动选择：`ORB=1.0`，`BRISK=5.0`
- 匹配器使用的是 `cv2.BFMatcher(cv2.NORM_HAMMING)`
- `SIFT` 使用的是 `cv2.SIFT_create()` 和 `cv2.BFMatcher(cv2.NORM_L2)`
