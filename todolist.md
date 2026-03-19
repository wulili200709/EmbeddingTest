# EmbeddingTest 中的 line2dup 长期集成实施文档

## 目标
在不修改仓库根目录原始 `MatchTemplate2` 代码的前提下，把 `line2dup` 的模板创建、模型保存、匹配和 ROI 跟随能力接入 `EmbeddingTest/qr_gui_pyside6.py`。

最终链路：

`参考图创建 line2dup 模板 -> 保存到产品目录 -> 测试图先做模板匹配 -> 自动输出 roi -> roi 提 embedding -> 计算余弦相似度 -> OK/NG`

## 约束
- 根目录现有代码保持不变，方便对照测试。
- 所有新增和改动都只放在 `EmbeddingTest` 下。
- `qr_gui_pyside6.py` 继续作为统一的主界面。
- `line2dup` 作为新的 `loc_method = line2dup` 接入现有自动 ROI 流程。

## 已落地模块

### 1. Bootstrap 层
- `EmbeddingTest/line2dup_bootstrap.py`

作用：
- 让 `EmbeddingTest` 作为运行根目录时，仍能安全 import 父目录中的 `line2dup_like_matcher.py` 等原始模块。

### 2. 共享核心层
- `EmbeddingTest/line2dup_template_core.py`

作用：
- 从原 `line2dup_template_workbench.py` 中抽出无 UI 核心逻辑
- 保持模型格式与根目录实现兼容
- 负责模板构建、pose 变换、模型 source 信息处理

当前提供的核心能力：
- `RoiRect`
- `MaskRect`
- `parse_levels`
- `build_mask_from_rects`
- `pose_infos_from_ui_values`
- `normalize_extracted_levels_to_roi`
- `transform_levels_for_pose`
- `transform_image_and_mask_expanded`
- `expanded_pose_affine`
- `make_class_source_payload`
- `load_class_source_assets`
- `build_multi_backend_detector`
- `copy_detector_class`

### 3. Recipe 层
- `EmbeddingTest/line2dup_recipe.py`

作用：
- 把 `line2dup` 的产品级参数从 UI 状态里抽出来
- 保存/加载产品级定位 recipe

当前字段：
- `model_path`
- `reference_image`
- `class_id`
- `backend`
- `threshold`
- `nms_iou`
- `topk`
- `crop_stride`
- `use_scene_mask`
- `follow_mode`
- `output_label`
- `reference_label`

### 4. ROI 跟随层
- `EmbeddingTest/line2dup_roi_follow.py`

作用：
- 执行 `line2dup` 匹配
- 基于匹配结果把参考图上的 `roi` 跟随到当前图
- 输出 polygon 或 rect

当前支持的跟随模式：
- `match_bbox`
  - 直接把匹配框作为输出 ROI
- `affine_roi`
  - 读取参考图 `roi`
  - 根据模板 pose 和匹配结果，把参考 ROI 变换到当前图

### 5. 服务层
- `EmbeddingTest/line2dup_locator.py`

作用：
- 按产品目录管理 `line2dup_model.json` 和 `line2dup_recipe.json`
- 对外提供统一的自动生成 ROI 接口

当前核心接口：
- `product_paths(product_dir)`
- `load_recipe_for_product(product_dir)`
- `save_recipe_for_product(product_dir, recipe)`
- `autogen_roi_json_from_line2dup(tgt_img_path, ref_img_path, product_dir)`

### 6. PySide6 模板页
- `EmbeddingTest/line2dup_template_page_pyside6.py`

作用：
- 在 Qt 中实现一个轻量模板页
- 不嵌 Tk，不依赖根目录 GUI
- 用于参考图建模和 recipe 保存

当前支持：
- 打开参考图
- 画 `template_roi`
- 追加 `exclude_mask`
- 配置角度/尺度/特征参数
- 创建并保存 `line2dup_model.json`
- 保存 `line2dup_recipe.json`
- 快速测试模板匹配和 ROI 跟随

## 已改造主界面

### `EmbeddingTest/qr_gui_pyside6.py`
已完成这些接入：

1. 新增定位方式
- `SUPPORTED_LOC_MODES` 增加 `line2dup`

2. 新增产品级路径
- `self._product_dir`
- `self._line2dup_model_path`
- `self._line2dup_recipe_path`

3. 新增模板页入口
- “line2dup 模板页”按钮

4. 会话持久化增强
- `session.json` 现在额外保存：
  - `ref_image`
  - `loc_method`

5. 产品切换状态清理
- 切产品时会同步清空：
  - `line2dup_recipe`
  - `ref_image`
  - 参考图标签显示

6. 自动 ROI 流程新增 line2dup 分支
- 当 `loc_method == line2dup`
- 自动加载产品级 recipe
- 自动检查 `line2dup_model.json`
- 自动把参考图 `roi` 跟随到目标图
- 最后写回 labelme `roi`

## 产品目录结构
每个产品目录位于：

`EmbeddingTest/.qr_session/<product_name>/`

当前资产结构：
- `session.json`
- `shape_model.npz`
- `register_model_<backbone>.npz`
- `line2dup_model.json`
- `line2dup_recipe.json`

## 推荐使用流程

### A. 初次建立某个产品
1. 在 `qr_gui_pyside6.py` 中新建产品
2. 设置参考图
3. 在参考图上手动画好最终用于 embedding 的 `roi`
4. 点击“line2dup 模板页”
5. 在模板页中：
   - 画 `template_roi`
   - 按需要添加 `exclude_mask`
   - 设置角度/尺度范围
   - 保存模型
6. 回到主界面，把定位方式切到 `line2dup`
7. 用“批量生成 ROI”生成当前产品图片的 `roi`
8. 再做 `OK/NG` 注册训练

### B. 日常测试
1. 选择产品
2. 选择 `loc_method = line2dup`
3. 打开测试图
4. 执行测试
5. 程序会先自动补 ROI，再计算 embedding 和余弦相似度

## 当前设计取舍

### 保持不动的内容
- 根目录 `line2dup_template_workbench.py`
- 根目录 `line2dup_like_matcher.py`
- 根目录原生 `.pyd` backend

### 当前没有做的事
- 没有把 Tk workbench 完全迁进 Qt
- 没有在 `EmbeddingTest` 中复制或改写根目录 line2dup 算法实现
- 没有改动根目录模型格式

### 当前 Qt 模板页的定位
它不是原 Tk workbench 的 1:1 全量替代，而是面向 `qr_gui` 集成的轻量模板编辑器。

## 后续建议

### 下一步优先项
1. 在模板页中增加“加载已有参考图标注”的能力
2. 给模板页增加 scene 匹配预览列表
3. 增加 `fusion / sim3` 的可视化测试项
4. 给 `line2dup_recipe.json` 增加更多生产参数
5. 补一组最小回归样例

### 后续可选优化
1. 支持模板页里直接编辑最终 embedding `roi`
2. 支持多 class_id 模型
3. 支持 scene mask
4. 支持在主界面显示 line2dup 匹配框叠加层

## 验收标准
满足以下条件即可认为本阶段完成：
- `EmbeddingTest` 下可以独立运行 `qr_gui_pyside6.py`
- 不改根目录代码即可使用 `line2dup` 模型
- `loc_method = line2dup` 能自动生成 `roi`
- 自动生成的 `roi` 能继续用于 embedding 和余弦值判别
- 产品切换后，模型和 recipe 不串目录

## 一句话结论
本次实现不是把 Tk 界面嵌进 Qt，而是：



sim 是匹配相似度分数，来自 result.match.similarity；分数越高，说明当前图和模板越像。tid 是 template_id，也就是命中的那一张模板/姿态变体编号。一个模型里如果有多角度、多尺度模板，tid 就表示这次命中了哪一个。相关绘制在 line2dup_template_page_pyside6.py (line 111)。




在 `EmbeddingTest` 内新增一套无 UI 共享层、ROI 跟随层、产品级 recipe 层和 PySide6 模板页，然后把 `line2dup` 作为新的自动 ROI 后端接入现有 `qr_gui_pyside6.py`。




1.模板创建的时候可以画多个区域，比如第一个画完之后点击确认，再点击+ ——，可以继续画区域
那同样，掩膜也是，点击+ — 可以继续画掩膜

2.模板中的recipe是什么follow mode
3.特征点需要可编辑，以前的版本是可以右键删除特征点的

sim什么意思 tid什么意思  create reference roi find 这个页面里的图都需要滚动鼠标图像会缩小放大。希望在find页面图片测试完之后，上面的字体需要放大两倍，在点击图的画会跳至到这张图  并显示之前的检测结果 在点击clear或者关闭line2dup的窗体的时候清空find页面的图片

现在的行为是：

Find 页不再只画轴对齐矩形模板框了，匹配模板本体改成按 match_quad() 画四边形；参考 ROI 跟随结果也按多边形画。
代码在 line2dup_template_page_pyside6.py (line 146)、line2dup_template_page_pyside6.py (line 1596)。

Reference ROI 现在支持多框。
默认连续画就是 roi1、roi2、roi3……
左侧有 Reference Regions 列表，点哪一个，右侧就把它作为当前选中 ROI，颜色也会和其它 ROI 区分开。
代码在 line2dup_template_page_pyside6.py (line 422)、line2dup_template_page_pyside6.py (line 1207)。

多个 Reference ROI 现在不只是保存显示了，训练/测试时也会一起送进 backbone。
我这里用的是“**每个 ROI 各提一个 embedding，然后拼接成一个总向量**，再做 L2 normalize”。
代码在 qr_core.py (line 758)、qr_core.py (line 879)、qr_gui_pyside6.py (line 1258)。

polygon ROI 送给 embedding 时，不再直接退化成纯外接矩形裁剪。
现在如果 labelme 里是 polygon，会先按 polygon 做掩膜，再裁 bbox 区域送进 backbone。
代码在 qr_core.py (line 738)。

另外：

Create 页里和模板保存无关的 定位Recipe 组我先隐藏了，见 line2dup_template_page_pyside6.py (line 360)。
Find 页顶部和列表里都继续保留 time=...ms。
你现在要这样验证：

彻底关掉当前 line2dup 窗口和主界面，再重开。
进 Reference ROI 页，连续画两个框，看左侧是不是自动出现 roi1、roi2。
点左侧某个 ROI，看右侧是否切换到这个 ROI。
保存后去 Find 跑一张有角度的图，确认：
模板匹配框是四边形
跟随 ROI 是 polygon
回到主界面训练/测试，当前产品会把这些 roi1/roi2/... 一起参与 embedding。

意思是以前如果你画的是一个斜着的四边形 ROI，程序虽然知道它是 polygon，但真正送进 backbone 时，还是会偷懒变成这个 polygon 的外接矩形来裁图。

比如你画的是一个斜着的目标区域：

   /------/
  /      /
 /------/
旧做法实际送进去的是：

+--------+
| /----/ |
|/    /  |
|----/   |
+--------+
也就是把 polygon 外面的背景也一起裁进去了。
这样会带来两个问题：

多余背景进入 embedding，特征会被污染
旋转目标周围的黑边/背景也会参与相似度计算
我现在改成的是：

先取 polygon 的外接矩形 bbox
只裁这块 bbox
但在 bbox 内再做一个 polygon mask
polygon 里面保留原图
polygon 外面全部置黑
再把这张“掩膜后的 ROI 图”送进 backbone
所以现在真正送进网络的是：

+--------+
|  ####  |
| ###### |
|  ####  |
+--------+
这里 # 表示 polygon 内有效区域，外面虽然还在 bbox 里，但已经被 mask 掉了。

一句话：

以前：polygon -> 直接变 bbox 裁剪
现在：polygon -> bbox 裁剪 + polygon 掩膜
这样更接近你真实画的 ROI，而不是把周围背景也算进去。

但要注意，它还不是旋转拉正。
也就是说现在只是“保留 polygon 内区域、去掉外面背景”，并没有把斜着的目标转正成水平图块。

如果你要，我下一条可以直接给你画一张：

外接矩形裁剪
polygon mask裁剪
旋转拉正裁剪

之前 Create 页打开时，优先拿的是主界面当前那张图，所以如果你主界面停在某张 OK/NG/TEST 图上，模板点会被叠到这张图上，看起来就像“模板和图片不匹配”。
我已经改了：

加载已有模型 时，会优先切回这个模型保存时对应的参考图，再显示模板点和模板区域。
新保存的模型也会把参考图路径一起存进去，后面重开更稳。
代码在 line2dup_template_core.py (line 381) 和 line2dup_template_page_pyside6.py (line 717)。
所以现在正确体验应该是：

打开 line2dup 模板页
如果当前产品已有模型，它会自动加载已有模型
并自动切到那张模板参考图
你看到的模板点就会和原模板图对上

对，每个区域的特征向量就是单独算的。现在这套逻辑不是把两个 ROI 先合成一个大框，而是：

roi1 -> e1
roi2 -> e2
然后再组成总向量 e = [e1, e2]
这里的 [] 不是求平均，也不是相加，是按维度首尾拼接。

如果每个 ROI 输出都是 1280 维，那就是：

e1: [a1, a2, ..., a1280]
e2: [b1, b2, ..., b1280]

e = [a1, a2, ..., a1280, b1, b2, ..., b1280]
也就是从 1280 维变成 2560 维。实现就在 qr_core.py。

为什么“拼接”通常比“两个 ROI 分数取平均”更合适，核心有 3 个原因。

它保留了每个 ROI 的身份
拼接后，前半段永远是 roi1，后半段永远是 roi2
模型能区分“上面区域像不像”和“下面区域像不像”
如果你最后只把两个 ROI 的分数平均：

score = (score1 + score2) / 2
那就丢掉了“到底是哪一个 ROI 出问题”的信息。

它保留了更多信息
平均分是很强的压缩。
例如两张图：

图 A: roi1 很像 OK，roi2 很不像 OK
图 B: roi1 很不像 OK，roi2 很像 OK
这两种情况平均分可能一样，但业务上它们不是同一种样本。
拼接向量能把这两种情况区分开，平均分不行。

它更适合你现在这套 proto/topk 注册方式
你现在不是训练一个复杂分类头，而是做 embedding 相似度比较。
那最自然的做法就是给每张样本构造一个完整的联合特征向量，然后：
OK 样本库里存这个联合向量
NG 样本库里也存这个联合向量
测试时也生成同样结构的联合向量
这样 train/test 结构完全一致。

为什么还要再做一次 L2 normalize

因为每个 ROI 的 embedding 虽然通常已经各自归一化了，但拼起来以后，总向量长度变了。

比如：

e1 已经是单位向量，||e1|| = 1
e2 也是单位向量，||e2|| = 1
拼起来以后：

e = [e1, e2]
它的长度大约会变成：

||e|| = sqrt(1^2 + 1^2) = sqrt(2)
这时候如果不重新归一化：

多 ROI 样本和单 ROI 样本的向量长度不可比
后面做余弦相似度时，数值尺度会不统一
所以要再做一次：

e_final = e / ||e||
这样最终还是单位向量，便于：

做余弦相似度
让不同样本的相似度可比
避免“ROI 数量越多，向量越长”这种无意义影响
一句话总结：

是先分别算 roi1、roi2 的特征向量
再拼接成一个联合特征
最后整体做一次 L2 normalize
这么做比“最后两个 ROI 分数取平均”更稳，因为它保留了每个区域的独立信息和顺序信息
如果你要，我下一条可以继续讲：

“拼接” 和 “两个 ROI 分别算分再加权融合” 这两种方案，各自适合什么场景。

e = [e1, e2]  为何要拼接再 L2 normalize？不单独每个roi的特征向量？


其实现在就是先单独算每个 ROI 的特征向量，不是没单独算。

流程是：

roi1 -> e1
roi2 -> e2
再把它们合成一个最终样本向量 E
之所以不把 e1、e2 永远分开存、分开比，主要是因为你现在这套 proto/topk 注册框架，最后需要的是每张图一个固定向量，这样才方便：

存 OK/NG bank
算 prototype
统一做余弦相似度
拼接 + L2 normalize 的好处是：

保留每个 ROI 的独立信息
前一段永远是 roi1，后一段永远是 roi2。

仍然只需要一个最终向量
对现有 proto/topk 逻辑最简单，不用改成“多路分数再融合”的另一套框架。

做完归一化后，比较是稳定的
不会因为 ROI 个数多、向量更长，就让相似度失真。

最关键的一点是：

如果每个 ROI 的向量本身都已经做过 L2 normalize，
那 拼接后再 L2 normalize，它的数学效果其实接近于：

把各 ROI 的余弦相似度做等权平均

比如两个 ROI：

样本 A: E = [e1, e2] / sqrt(2)
样本 B: F = [f1, f2] / sqrt(2)
那么：

cos(E, F) = (e1·f1 + e2·f2) / 2
也就是：

roi1 相似度
加上 roi2 相似度
再平均
所以它不是乱拼。
它相当于把“多 ROI 相似度融合”压缩成了一个单向量比较问题。

如果完全不拼接、永远分开处理，也不是不行，但你就得另外设计一套融合规则，比如：

score = 0.5 * score1 + 0.5 * score2
或者取最小值
或者某个 ROI 权重大一点
那样系统会更复杂。

一句话：

每个 ROI 的特征本来就是单独算的
拼接是为了把多 ROI 变成一个统一样本表示
再 L2 normalize，是为了让这个联合向量还能稳定地做余弦相似度比较
如果你要，我可以下一条直接给你画一个“两 ROI -> concat -> normalize -> cosine”的公式图。


# 最标准的使用顺序

在主界面选产品，并设置参考图。
打开 line2dup 模板页。
在 Create 页建模板。
在 Reference ROI 页画 roi1 / roi2 / ...，并保存。
回到主界面，定位方式选 line2dup。
点 批量生成ROI。
所有 OK/NG/TEST 图就会自动补齐这些 ROI。
再去训练/注册。
所以“提前把 ROI 全补好”真正指的是：
先把参考 ROI 定义好，再让程序批量自动生成到所有图片上。

# find的参数会带到 主界面测试TEST上
主要看 line2dup 模板页 里的 Find 参数
里面的：
backend
threshold
nms_iou
follow_mode
会同步保存到当前产品的 line2dup_recipe.json
主界面点“批量生成ROI”或“测试TEST”时，实际就是读这个 recipe 去跑匹配

# 多个模板测试取分数最高的
主界面 TEST 用的是当前产品模型里分数最高的那个模板命中结果

# tid 和 class_id 的关系
class_id 和 tid 的关系可以理解成：

class_id：这是哪一类产品/模板组
tid：这是这个类下面的第几张模板变体
也就是：

class_id 是“大类名”
tid 是“这个类里的模板编号”
比如你现在产品叫 Block，那可能就是：

class_id = Block
而这个 Block 下面，不一定只有一张模板。
如果建模板时做了：

多角度
多尺度
多姿态变体
那就会形成很多模板，例如：

tid=0
tid=1
tid=2
tid=3
...
它们都属于同一个：

class_id = Block
所以像你图上这种：

class=Metal tid=2
意思就是：

命中的类别是 Metal
在 Metal 这一类下面，最终匹配最好的是第 2 号模板变体
为什么会有很多 tid
因为 line2dup 建模时，通常不是只存一张模板原图，而是会根据你设置的：

angle_start / angle_end / angle_step
scale_start / scale_end / scale_step
生成很多模板姿态。
这些姿态都挂在同一个 class_id 下面，但各自有不同的 template_id，也就是你看到的 tid。

所以关系就是：

class_id
  ├─ tid 0
  ├─ tid 1
  ├─ tid 2
  └─ tid 3
在匹配时：

先按 class_id 限定搜哪一类
再在这类下面所有 tid 模板里找
选分数最高的那个 tid
一句话总结：

class_id = 模板类别/产品名
tid = 这个类别下的具体模板变体编号
如果你要，我下一条可以继续给你举一个你现在这套 Block 模型里：

为什么会出现 tid=0,1,2...
它们分别对应哪些角度/尺度变化

# topK & proto
proto 和 topk 的区别，本质上是：

proto：拿你所有 OK 样本平均成一个中心，NG 样本也平均成一个中心，再看测试图更靠近哪个中心
topk：不先压成一个中心，而是直接拿测试图去和样本库里的所有图比，取最像的前 k 个来算分
在代码里就是 qr_core.py (line 915) 这一段：

proto
sim_ok = e @ ok_proto
sim_ng = e @ ng_proto
topk
sim_ok = score_topk(e, ok_bank, k=k_ok)
sim_ng = score_topk(e, ng_bank, k=k_ng)
proto 的特点：

快
简单
样本多、类内变化不大时比较稳
但缺点是：

一旦 OK 内部差异很大，平均以后会把细节抹掉
小样本时很容易被某几张样本带偏
如果姿态、光照变化大，均值原型不一定能代表真实分布
topk 的特点：

更像“最近邻”
对小样本更友好
对类内多样性更容忍
因为它不是问：

“你像不像这个平均人”
而是问：

“你像不像样本库里最像你的那几个人”
所以像你这种场景：

ROI 小
样本少
角度变化大
裂缝差异又很局部
通常 topk 往往比 proto 更靠谱。

举个直觉例子。

假设 OK 有三张：

正放
轻微旋转
亮度偏暗
NG 也有三张：

裂缝很明显
裂缝较轻
旋转状态下裂缝
如果你用 proto：

这些样本都会被平均
最后得到一个“模糊的中心”
测试图如果正好接近其中某个具体样本，但不接近平均中心，就可能判错
如果你用 topk=3：

它会找最像的几张样本
测试图只要和某些真实样本很接近，就更容易被判对
一句话理解：

proto：和“类中心”比
topk：和“最像的几个真实样本”比
实战建议：

样本少、变化大：优先试 topk
样本多、分布稳定：proto 可以更简单
你现在这个裂缝任务：我更建议先试 topk
如果你要，我下一条可以继续讲：

topk 里的 k=1、3、5 分别适合什么情况。