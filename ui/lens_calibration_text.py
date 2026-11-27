"""Bilingual lens UI copy and display translations for saved and live diagnostics."""
from __future__ import annotations

import re
from string import Formatter

from ui.i18n import LANG_EN, language_code


def lens_label(chinese: str, english: str) -> str:
    return english if language_code() == LANG_EN else chinese


# Calibration files retain their original diagnostic text. Translate it only for display.
_ENGLISH = {
    "不是本 Demo 的标定参数文件，或文件版本不受支持。": "Unsupported calibration file or version.",
    "标定参数文件无效：{0}": "Invalid calibration file: {0}",
    "{0}必须包含宽和高两个整数。": "{0} must contain integer width and height.",
    "{0}必须为大于等于 2 的整数。": "{0} must contain integers of at least 2.",
    "图像尺寸": "Image dimensions",
    "相机内参必须为有限数值组成的 3×3 矩阵。": "The camera matrix must be a finite 3 × 3 matrix.",
    "相机焦距参数 fx、fy 必须大于 0。": "Focal lengths fx and fy must be positive.",
    "相机内参不是受支持的标准针孔模型。": "Unsupported pinhole camera matrix.",
    "畸变系数必须为 4 或 5 个有限数值（k1、k2、p1、p2、可选 k3）。": "Expected four or five finite distortion coefficients (k1, k2, p1, p2, optional k3).",
    "畸变系数必须是一维向量。": "Distortion coefficients must be a one-dimensional vector.",
    "重投影误差必须为非负有限数值。": "Reprojection error must be finite and nonnegative.",
    "标定板类型不受支持。": "Unsupported board type.",
    "标定板点数": "Board dimensions",
    "标定板间距必须为正数。": "Board spacing must be positive.",
    "标定图片报告格式无效。": "Invalid calibration image report.",
    "标定提示格式无效。": "Invalid calibration warning format.",
    "请选择棋盘格或对称圆点阵。": "Select a checkerboard or symmetric circle grid.",
    "每行和每列至少需要 3 个角点或圆点。": "Each row and column needs at least three points.",
    "标定板间距必须大于 0。": "Board spacing must be greater than zero.",
    "请至少导入 {0} 张不同位置和倾角的原始标定板图片。": "Select at least {0} original board images at different positions and angles.",
    "无法读取图片": "Cannot read image",
    "图片尺寸 {0} 与 {1} 不一致": "Image size {0} differs from {1}",
    "{0}：分辨率不一致。请使用同一相机、同一分辨率的原始图片。": "{0}: resolution mismatch. Use originals from the same camera at the same resolution.",
    "图片像素内容重复，未计入有效张数": "Duplicate image excluded",
    "特征检测失败：{0}": "Feature detection failed: {0}",
    "未找到完整图案，请检查行列数、清晰度和遮挡": "Incomplete pattern. Check the board dimensions, focus and visibility.",
    "检测到的标定点无效": "Invalid calibration points",
    "已检测到完整图案": "Complete pattern detected",
    "有效且不重复的图片只有 {0} 张，需要至少 {1} 张。\n请确认棋盘格填写的是内角点数，圆点阵填写的是圆心数，并补充不同位置和倾角的照片。": "Only {0} distinct usable images. At least {1} are required. Check the inner-corner or circle-center counts and add views at other angles and positions.",
    "正在求解相机内参、畸变系数和重投影误差…": "Solving camera intrinsics and distortion…",
    "标定求解失败：{0}": "Calibration solver failed: {0}",
    "重投影误差仅反映这批标定图的拟合情况；请用独立标定板图片检查边缘直线和尺寸。": "Reprojection error describes the calibration set. Verify edges and dimensions on an independent image.",
    "标定点对视野边缘的覆盖不足，长针位两端的矫正效果需要补充边缘照片后再验证。": "Limited edge coverage. Add views that cover the ends of the inspection area.",
    "标定板的位置变化较小，请增加左右、上下位置及不同倾角的照片。": "Board positions vary too little. Add horizontal, vertical and tilted views.",
    "重投影误差超过 1 px，请检查图案参数、板面平整度、图像清晰度和采集姿态。": "Reprojection error exceeds 1 px. Check board dimensions, flatness, focus and pose.",
    "手动试调参数：仅供观察畸变补偿效果，尚未经标定板求解，不可据此确认尺寸精度。": "Manual parameters are for visual preview only. Dimensional accuracy has not been calibrated.",
    "视野保留参数必须位于 0 到 1 之间。": "Field of view must be between 0 and 1.",
    "参数无法生成有效矫正图像，请重新标定或减小手动畸变系数。": "Cannot generate a corrected image. Recalibrate or reduce the manual coefficients.",
    "畸变参数生成了无效像素映射，请重新标定。": "Invalid pixel map. Recalibrate the lens.",
    "请输入灰度或彩色图像。": "Provide a grayscale or color image.",
    "图片尺寸 {0}×{1} 与标定尺寸 {2}×{3} 不一致；请使用同一分辨率的原始图片。": "Image size {0} × {1} differs from the calibration size {2} × {3}. Use the original resolution.",
    "图像数据类型不受支持。": "Unsupported image data type.",
    "图像通道数不受支持。": "Unsupported image channel count.",
    "棋盘内角点数": "Inner-corner counts",
    "单张模式至少需要 20 个内角点，且每行、每列至少 4 点，例如 5×4。": "Single-image calibration needs at least 20 inner corners and at least four in each row and column.",
    "可见棋盘网格范围": "Visible grid dimensions",
    "稀疏棋盘至少需要 20 个实际检测点及其逐点对应的二维网格索引。": "A sparse grid needs at least 20 observed points with matching 2D indices.",
    "棋盘网格索引必须为有限整数。": "Grid indices must be finite integers.",
    "棋盘网格索引必须是整数，不能使用小数格位。": "Grid indices must be whole numbers.",
    "棋盘网格索引重复，多个像素点不能对应同一个格点。": "Multiple image points share one grid index.",
    "稀疏网格索引必须从 0 开始，且最大索引应与可见网格行列范围一致。": "Sparse indices must start at zero and span the visible rows and columns.",
    "稀疏棋盘至少需要四个不同列和四个不同行的实际角点。": "The sparse grid needs observations in at least four rows and four columns.",
    "棋盘网格索引共线，无法确定二维平面。": "Grid indices lie on a line. The plane is undefined.",
    "棋盘相邻格距确认状态必须为布尔值。": "The grid-spacing confirmation flag must be boolean.",
    "缺少足够的双轴相邻格点，不能确认每一步对应所填写的物理格距。": "Too few neighbors on both axes to confirm the physical spacing.",
    "棋盘平面映射退化，无法确定有效的矫正尺度。": "Degenerate board-plane mapping. The correction scale is undefined.",
    "拟合的径向模型在输出画布内发生折叠或接近折叠；请使用覆盖更大、更加清晰的原始棋盘图。": "The radial map folds within the output image. Use a sharper board image covering more of the frame.",
    "棋盘平面在输出画布内接近无穷远或发生翻折；请减小拍摄倾角，并让棋盘覆盖更大视野。": "The board plane folds or approaches infinity. Reduce the tilt and increase board coverage.",
    "棋盘网格范围": "Board grid dimensions",
    "不是受支持的单张棋盘标定参数文件。": "Unsupported single-image calibration profile.",
    "稀疏参数缺少实际角点的网格索引。": "The sparse profile is missing corner indices.",
    "旧版参数不能包含稀疏网格索引，请使用 v2 格式。": "Sparse grid indices require a v2 profile.",
    "单张标定参数无效：{0}": "Invalid single-image calibration: {0}",
    "单张矫正模式不受支持。": "Unsupported single-image correction mode.",
    "畸变中心必须是两个有限坐标。": "The distortion center needs two finite coordinates.",
    "此单张模型只支持固定在图像中心的畸变中心。": "This model requires the distortion center to be the image center.",
    "径向系数必须包含有限的 k1、k2 两个数值。": "Both radial coefficients k1 and k2 must be finite.",
    "平面变换必须是有限的 3×3 矩阵。": "The homography must be a finite 3 × 3 matrix.",
    "平面变换奇异或未正确归一化。": "Singular or incorrectly normalized homography.",
    "棋盘范围内的平面映射存在翻折或无穷远点。": "The board mapping folds or reaches infinity.",
    "标定误差和棋盘间距无效。": "Invalid calibration error or board spacing.",
    "棋盘角点数量或坐标无效。": "Invalid corner count or coordinates.",
    "棋盘角点重复或退化为一条线。": "Board corners are duplicated or collinear.",
    "棋盘角点超出图像范围。": "Board corners lie outside the image.",
    "图片报告格式无效。": "Invalid image report format.",
    "棋盘覆盖信息格式无效。": "Invalid board coverage data.",
    "棋盘覆盖信息 {0} 必须为有限数值。": "Coverage field {0} must be finite.",
    "棋盘覆盖信息 {0} 含有无效数值。": "Coverage field {0} contains invalid values.",
    "棋盘覆盖信息 {0} 必须为单个数值。": "Coverage field {0} must be a scalar.",
    "棋盘覆盖比例必须位于 0 到 1 之间。": "Coverage fraction must be between 0 and 1.",
    "棋盘覆盖边界必须含有四个坐标。": "Coverage bounds need four coordinates.",
    "矫正模式或棋盘间距无效。": "Invalid correction mode or board spacing.",
    "角点必须与网格索引逐点对应（完整棋盘为列数×行数），且坐标必须为有限数值。": "Each grid index needs a matching corner with finite coordinates.",
    "棋盘角点重复或共线，无法标定。": "Duplicate or collinear corners prevent calibration.",
    "棋盘覆盖范围过小，单张图无法可靠约束边缘畸变；请让棋盘覆盖更大视野后重新拍摄。": "Board coverage is too small. Retake the image with the checkerboard filling more of the frame.",
    "正在拟合径向畸变和平面姿态：初值 {0}/5…": "Fitting radial distortion and plane pose, seed {0}/5…",
    "单张参数拟合未收敛，请检查棋盘是否清晰、完整且平整。": "The fit did not converge. Check board focus, visibility and flatness.",
    "单张拟合误差过大（RMS {0} px）；请检查行列数、板面平整度和角点检测。": "Fit error is too high (RMS {0} px). Check grid dimensions, board flatness and corner detection.",
    "拟合参数触及模型边界，单张图片不足以约束当前畸变；请更换覆盖更完整的棋盘照片。": "The fit reached its bounds. Use an image with greater board coverage.",
    "单张受约束拟合：畸变中心固定在图像中心，只估计 k1/k2；不代表完整相机内参或切向畸变标定。": "Single-view fitting estimates k1 and k2 about the image center. Full intrinsics and tangential distortion are outside this model.",
    "RMS 是本张棋盘的拟合误差，请用另一张原图验证边缘效果。固定画布可能出现黑边或内容超出边界。": "RMS describes this board image. Check edges on an independent image. The fixed canvas may clip content or show black borders.",
    "棋盘没有覆盖整个视野；棋盘外区域属于外推，尤其四角的矫正和尺寸精度尚未验证。": "Areas beyond the observed board are extrapolated. Corner accuracy is unverified.",
    "拟合 RMS 超过 0.5 px，请放大检查角点和板面平整度。": "Fit RMS exceeds 0.5 px. Inspect corner positions and board flatness.",
    "本张棋盘对两个径向系数的独立约束较弱；即使 RMS 较低，边缘外推也可能不准，请扩大棋盘覆盖范围验证。": "These observations weakly constrain the two radial coefficients. Increase coverage to verify edge accuracy.",
    "平面拉正和标称 mm/px 仅适用于本次棋盘所在平面；不同高度的针脚不能直接沿用该比例。": "The perspective correction and mm/px scale apply only to the calibrated plane.",
    "使用实际检测到的可见格点拟合；编码区或遮挡区的缺失角点未补造。": "The fit uses observed corners and skips missing or obscured points.",
    "尚未确认索引相邻一步对应一个真实方格；当前仅作几何预览，不输出标称 mm/px。": "Physical grid spacing is unconfirmed. Only geometric preview is available.",
    "单张棋盘拟合": "Single-image checkerboard fit",
    "发现矫正记录但无法读取，请选用相机原始图片。": "Unreadable correction record. Select an original camera image.",
    "这张图片已有矫正记录，请使用相机原始棋盘图进行标定。": "This image is already corrected. Select the original checkerboard image.",
    "正在读取原始棋盘并检测内角点…": "Reading the board and detecting inner corners…",
    "无法读取棋盘图片。": "Cannot read the checkerboard image.",
    "自动网格检测返回了无效的诊断信息。": "Invalid grid detection diagnostics.",
    "未找到完整棋盘，请检查内角点行列数、清晰度和遮挡；带编码块或被裁切的大棋盘可改用自动可见网格。": "Incomplete checkerboard. Check inner-corner counts, focus and occlusion, or enable automatic grid detection.",
    "已检测到 {0} 个角点，正在联合拟合径向畸变和平面姿态…": "Detected {0} corners. Fitting distortion and board pose…",
    "已识别 {0} 个可见格点（网格范围 {1}×{2}），但标定未通过：{3}": "Detected {0} corners in a {1} × {2} grid. Calibration failed: {3}",
    "自动可见格点拟合（缺失角点未补造）": "Fit to visible grid points",
    "单张拟合完成：RMS {0} px": "Fit complete. RMS {0} px",
    "单张模式使用固定画布，视野参数只能为 1；不会自动裁剪或缩放。": "Single-image correction requires a fixed canvas with field of view set to 1.",
    "无法生成有限的矫正映射。": "Cannot generate a finite correction map.",
    "请输入灰度或彩色原图。": "Provide an original grayscale or color image.",
    "图片尺寸与单张标定尺寸 {0}×{1} 不一致。": "Image dimensions differ from the calibration size {0} × {1}.",
    "自动格点检测需要单通道灰度原图。": "Automatic grid detection requires a single-channel grayscale image.",
    "灰度图太小或数据类型不受支持。": "The grayscale image is too small or has an unsupported data type.",
    "灰度图包含非有限数值。": "The grayscale image contains nonfinite values.",
    "图像没有足够明暗变化，无法检测棋盘。": "Insufficient tonal variation to detect a checkerboard.",
    "图像对比度不足，未发现可靠棋盘。": "Image contrast is too low to detect a reliable checkerboard.",
    "棋盘格数量或图像尺寸不足，无法确认横纵周期。": "Too few squares or too small an image to confirm both grid periods.",
    "未发现可靠的双向棋盘周期；空图、条纹和随机纹理不能用于棋盘标定。": "No reliable checkerboard pattern in both directions. Use a checkerboard image.",
    "棋盘格在检测图中不足约 7 像素，无法可靠区分真实角点；请提高格子成像尺寸。": "Grid squares occupy fewer than about 7 pixels. Increase their image size.",
    "横纵格距差异过大；自动检测仅支持接近正视、格轴接近水平或垂直的方格棋盘。": "Horizontal and vertical spacing differ too much. Use a nearly front-facing, axis-aligned checkerboard.",
    "未检测到棋盘交点。": "No checkerboard intersections detected.",
    "角点不能组成接近水平/垂直的单格邻接关系；请检查棋盘角度和清晰度。": "Corners do not form single-cell horizontal and vertical neighbors. Check board angle and focus.",
    "能确认的一格棋盘边太少；编码、模糊或裁切使实际格距不可靠。": "Too few confirmed single-cell edges. Check blur, cropping and coded regions.",
    "未确认相近的横纵单格步长；请使用接近正视的方格棋盘。": "Horizontal and vertical steps differ. Use a nearly front-facing square checkerboard.",
    "正在确认横纵棋盘周期和真实鞍点…": "Checking grid periods and saddle points…",
    "可靠棋盘鞍点不足 20 个；请检查格子大小、曝光和图案类型。": "Fewer than 20 reliable saddle points. Check square size, exposure and board type.",
    "找到 {0} 个候选鞍点，正在验证单格邻接与黑白交替…": "Found {0} candidate points. Checking adjacency and alternating squares…",
    "候选点不能组成至少 20 点的单一二维棋盘网格。": "The candidates do not form a single 2D grid of at least 20 points.",
    "棋盘邻接环路出现索引冲突，可能存在错点或跳格；已拒绝自动标定。": "Conflicting grid indices. Check for misdetected or skipped corners.",
    "棋盘格点索引重复，无法可靠排除编码区域的干扰。": "Duplicate grid indices. Coded regions may interfere with detection.",
    "发现多个较大的棋盘连通区域，无法确认属于同一张标定板；请减少干扰或遮挡。": "Several large grids detected. Reduce interference or occlusion.",
    "缺少同时受横纵邻接支持的真实棋盘角点。": "Too few corners with neighbors on both axes.",
    "局部筛选后的格点索引不一致，已拒绝自动标定。": "Inconsistent grid indices after filtering.",
    "双向邻接筛选后出现多个较大网格，无法可靠确认同一棋盘。": "Several large grids remain after neighbor filtering.",
    "回到原始分辨率后部分角点移动过大，局部结构不可靠；请提高图像清晰度。": "Corners shifted too far at full resolution. Improve image focus.",
    "大量局部角点与相邻格距不一致，请检查编码干扰、图像清晰度或板面折痕。": "Corner spacing is inconsistent. Check focus, coded regions and board creases.",
    "局部格距和角点结构不一致，无法确认可靠的二维棋盘。": "Local spacing and corner structure do not form a reliable 2D grid.",
    "局部结构检查后的棋盘索引不一致。": "Inconsistent indices after local structure checks.",
    "局部结构检查后棋盘分成多个较大区域，请检查遮挡或板面折痕。": "The grid splits into several large regions. Check for occlusion or board creases.",
    "自动标定至少需要四行、四列实际观测到的棋盘交点。": "Automatic calibration needs observed intersections in at least four rows and four columns.",
    "棋盘网格过于稀疏，无法可靠区分缺点与跨格连接。": "The grid is too sparse to distinguish missing points from skipped cells.",
    "已确认 {0} 个真实角点，格网跨度 {1}×{2}，未补缺点。": "Confirmed {0} observed corners spanning {1} × {2}.",
    "无效的物理相机角色：{0}": "Invalid physical camera role: {0}",
    "JSON 顶层必须是对象": "The JSON root must be an object",
    "无法读取镜头矫正文件 {0}：{1}": "Cannot read lens correction file {0}: {1}",
    "镜头矫正设置格式或版本无效。": "Invalid lens correction settings format or version.",
    "镜头矫正开关必须为布尔值。": "The correction flag must be boolean.",
    "视野参数必须在 0 到 1 之间": "Field of view must be between 0 and 1",
    "参数版本标识无效": "Invalid profile version ID",
    "标定分辨率无效": "Invalid calibration resolution",
    "尚未显式导入标定参数，不能启用矫正": "Import a calibration profile before enabling correction",
    "镜头矫正设置无效：{0}": "Invalid lens settings: {0}",
    "单张标定只支持固定画布，alpha 必须为 1": "Single-image calibration requires a fixed canvas with alpha set to 1",
    "标定参数无效：{0}": "Invalid calibration profile: {0}",
    "无法生成镜头矫正映射：{0}": "Cannot build the lens correction map: {0}",
    "相机矫正配置必须是映射。": "Camera correction settings must be a mapping.",
    "相机配置重复或矫正开关无效。": "Duplicate camera configuration or invalid correction flag.",
    "已导入的标定版本内容被修改；请重新显式导入。": "The imported profile has changed. Import it again.",
    "请先为该相机显式导入标定参数。": "Import a calibration profile for this camera first.",
    "标定版本导入信息格式无效。": "Invalid profile import metadata.",
    "标定版本缺少导入信息，无法确认历史输出视野。": "Missing import metadata. The original output field of view is unknown.",
    "标定版本与当前设置不符，禁止自动切换参数。": "The profile version differs from the current settings.",
    "标定版本导入信息与几何参数不一致。": "The profile import metadata differs from its geometry.",
    "已选择的标定版本不存在：{0}": "Cannot find the selected calibration profile: {0}",
    "设置中的分辨率与标定版本不一致。": "The configured resolution differs from the calibration profile.",
    "相机序列号不符：参数绑定 {0}，当前为 {1}。": "Camera serial mismatch. Expected {0}, received {1}.",
    "该图缺少已矫正版本记录，不能作为当前矫正空间的参考图。": "The image has no correction version record for this coordinate system.",
    "已矫正图片缺少完整标定参数，禁止再次矫正。": "The corrected image lacks its full calibration profile.",
    "图片已使用不同版本的标定参数矫正，请选择原图或匹配版本。": "This image uses another profile. Select the original or a matching profile.",
    "已矫正图片属于另一台相机，不能混用。": "This corrected image belongs to another camera.",
    "图像分辨率与标定分辨率 {0} 不一致，不能缩放套用。": "Image resolution differs from the calibrated resolution {0}.",
    "镜头矫正失败：{0}": "Lens correction failed: {0}",
    "无法读取图像：{0}": "Cannot read image: {0}",
    "当前为原图模式，不能混用已矫正参考图。": "Select an original reference image in raw-image mode.",
    "参考图或其矫正版本记录不存在。": "Missing reference image or correction record.",
    "已矫正图片的原图记录存在循环。": "Circular source-image reference in the correction record.",
    "已矫正图片的原图记录层级过深。": "Too many nested source-image references.",
    "导出帧缺少有效的矫正版本。": "The exported frame lacks a valid correction version.",
    "无法读取原图：{0}": "Cannot read the original image: {0}",
    "校正图片缓存与原图指纹不符。": "The cached correction does not match the original image.",
    "PNG 缓存只支持 uint8/uint16 原图；浮点图请使用数组入口，不能隐式转换。": "PNG caching supports uint8/uint16 originals. Use the array interface for floating-point images.",
    "无法编码矫正图片。": "Cannot encode the corrected image.",
    "此参数由模拟棋盘图求得，不能用于真实相机。": "Synthetic checkerboard profile. Camera application is disabled.",
}


def _message_pattern(template):
    parts = []
    for literal, field, _, _ in Formatter().parse(template):
        parts.append(re.escape(literal))
        if field is not None:
            parts.append("(.+?)")
    return re.compile("".join(parts), re.DOTALL)


_TEMPLATES = [(_message_pattern(source), target) for source, target in _ENGLISH.items() if "{0}" in source]


def diagnostic_text(value) -> str:
    text = str(value)
    if language_code() != LANG_EN or text.isascii():
        return text
    if text in _ENGLISH:
        return _ENGLISH[text]
    for pattern, target in _TEMPLATES:
        match = pattern.fullmatch(text)
        if match:
            return target.format(*(diagnostic_text(part) for part in match.groups()))
    # Multi-view progress prefixes the diagnostic with the original filename.
    filename, separator, message = text.partition("：")
    if separator and message:
        translated = diagnostic_text(message)
        if translated != message:
            return f"{filename}: {translated}"
    return text
