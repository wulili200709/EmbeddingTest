XFeat 是一个卷积神经网络，对输入图像进行多尺度特征图提取，在每个像素位置预测检测置信度（detectability score）和一个 64 维描述子。最终只保留置信度最高的前 top_k=4096 个位置作为关键点。这个数值反映了特征检测器能从图中挖掘出多少个可区分的局部结构（角点、纹理边缘等）。

matches：2255
LighterGlue（Transformer 架构）接受两组关键点和描述子作为输入，通过 Self-Attention + Cross-Attention 对两组点建立全局上下文感知，最终输出一个双射匹配矩阵（每个模板点最多配一个搜索点，反之亦然）。matches=2255 表示 LighterGlue 在置信度阈值（min_confidence=0.1）过滤后，认为有 2255 对点之间存在语义上的对应关系。这是纯描述子层面的软性匹配，尚未做几何一致性约束。

inliers：978 / inlier_ratio：0.43
将 2255 对匹配点送入 RANSAC（随机采样一致性），以 4 点法估计单应矩阵 H（findHomography，maxIters=5000，confidence=0.995）。RANSAC 每次随机采 4 对点，估一个 H，然后统计在该 H 下重投影误差 < 4 像素（ransacReprojThreshold=4.0）的点对数量，迭代后选最优 H。

inliers = 978：满足最优 H 的几何约束的点对数，即几何内点
outliers = 2255 - 978 = 1277：描述子匹配成功但几何不一致，视为误匹配（outlier）
inlier_ratio = 978 / 2255 ≈ 0.43：内点率，反映单应矩阵的质量和估计可信度


# RANSAC（Random Sample Consensus，随机采样一致性）
核心思想：在一堆数据里，有些是"好点"（inlier），有些是"噪声"（outlier）。RANSAC 的策略是：不试图用所有数据拟合，而是反复随机抽一小撮数据，找出能让最多点满足的那个模型。

以单应矩阵估计为例
已知 2255 对匹配点，但其中约 57% 是误匹配（outlier）。RANSAC 的做法：

重复 N 次（最多 5000 次）：
  1. 随机抽 4 对匹配点
  2. 用这 4 对点计算一个候选单应矩阵 H
  3. 把全部 2255 对点代入 H，
     计算"把模板点用 H 映射后与搜索点的距离"（重投影误差）
  4. 距离 < 4 像素 → 内点（inlier），否则 → 外点（outlier）
  5. 记录本次内点数量
最终选内点数量最多的那个 H 作为结果
为什么要随机抽样？
因为如果直接用所有点做最小二乘拟合，少量误匹配会严重拉偏结果。
随机抽 4 对点，概率上有很大可能抽到全是正确匹配的点，这样算出的 H 才是正确的。

直观比喻
你有 100 个人在猜一条直线的位置，其中 60 个人说真话，40 个人在乱说。
RANSAC 的做法是：随机抽 2 个人画一条线，看有多少其他人（近似）同意这条线，
重复多次，选"支持者最多"的那条线。

概念	含义
inlier	符合当前模型（误差 < 阈值）的点
outlier	不符合模型的点（误匹配、噪声）
inlier_ratio	inlier ÷ 总匹配数，越高说明误匹配越少
重投影误差	把点用 H 变换后与真实位置的像素距离


.\.venv\Scripts\python.exe -m iv4matchmethod annotate-template `
  --image C:\Users\goney\Desktop\EmbeddingTest\1\test\Image_20260206135743336.bmp `
  --output C:\Users\goney\Desktop\EmbeddingTest\1\test\template_annotation1.json



  .\.venv\Scripts\python.exe -m iv4matchmethod match-xfeat `
  --template-annotation C:\Users\goney\Desktop\test\2\test\template_annotation.json `
  --search-image C:\Users\goney\Desktop\test\2\test\Image_20260304155029880.bmp `
  --output-dir C:\Users\goney\Desktop\test\2\test\xfeat_result



.\.venv\Scripts\Activate.ps1
  .\.venv\Scripts\python.exe -m iv4matchmethod match-xfeat `
  --template-annotation C:\Users\goney\Desktop\test\2\test\template_annotation.json `
  --search-image C:\Users\goney\Desktop\test\2\test\Image_20260304155111521.bmp `
  --output-dir C:\Users\goney\Desktop\test\2\test\xfeat_result `
  --max-dim 1024 `
  --top-k 4096 `
  --min-cossim 0.82



  cd c:\Users\goney\Desktop\MatchTemplate\iv4matchmethod

.\.venv\Scripts\python.exe -m iv4matchmethod match-xfeat `
  --template-annotation "C:\Users\goney\Desktop\test\2\test\template_annotation.json" `
  --search-image "C:\Users\goney\Desktop\test\2\test\test\Image_20260304155141610.bmp" `
  --output-dir "c:\Users\goney\Desktop\test\2\test\tmp_lightglue_match_2" `
  --matcher lightglue `
  --max-dim 1024 `
  --template-top-k 1024 `
  --search-top-k 2048 `
  --min-confidence 0.1


.\.venv\Scripts\python.exe -m iv4matchmethod match-xfeat `
  --template-annotation "C:\Users\goney\Desktop\EmbeddingTest\1\test\template_annotation1.json" `
  --search-image "C:\Users\goney\Desktop\EmbeddingTest\1\test\Image_20260206135738065.bmp" `
  --output-dir "c:\Users\goney\Desktop\test\2\test\tmp_lightglue_match_3" `
  --matcher lightglue `
  --max-dim 1024 `
  --template-top-k 1024 `
  --search-top-k 2048 `
  --min-confidence 0.1
