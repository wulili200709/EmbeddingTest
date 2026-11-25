# 原生模块构建说明

本工程有两条原生编译流程，另有独立的 PyInstaller 打包流程。目录与 Git 规则整理不会运行这些构建，也不会改写已有算法或二进制。

## 构建前提

- Windows x64、Python 3.12、匹配的 MSVC C++ 构建工具。
- Python 构建依赖：`setuptools`、`wheel`、`pybind11`、`numpy`。
- Shape 后端需要包含头文件、`.lib` 和运行 DLL 的 OpenCV 开发包；仅安装 Python `opencv` 包不足以提供这些编译输入。
- 当前四个 `shape_*.pyd` 和 `NccMatchNative.dll` 的 PE 导入表依赖 `opencv_world4130.dll`。不要仅为目录整洁更换版本或删除该 DLL。

构建脚本通过 `SHAPE_OPENCV_BUILD` 指定 OpenCV 的 `build` 目录，通过 `SHAPE_OPENCV_WORLD_LIB` 指定库名（默认 `opencv_world4130`）。`LINE2DUP_OPENCV_BUILD` 和 `LINE2DUP_OPENCV_WORLD_LIB` 仅作为历史兼容变量。脚本中 `C:\Users\ADMIN\tools\opencv\build` 是旧机器的默认路径，其他机器应显式设置环境变量。

## Shape 模板匹配后端

输入：

- `native/line2dup_native.cpp`
- `vendor/_third_party_shape_based_matching/`
- `vendor/_third_party_shape_based_matching_fusion_fix_memo/`
- `vendor/_third_party_shape_based_matching_sim3/`
- `vendor/eigen/`

`setup.py` 会创建 Original、Fusion、Fusion V2、SIM3 四个扩展，生成 `shape_original.cp312-win_amd64.pyd`、`shape_fusion.cp312-win_amd64.pyd`、`shape_fusionv2.cp312-win_amd64.pyd` 和 `shape_sim3.cp312-win_amd64.pyd`，并将它们及 OpenCV DLL 复制回应用根目录。Fusion 与 SIM3 的编译兼容补丁写入 `build/_patched_backends/`，该目录属于生成物。

在 `EmbeddingTest` 应用根目录运行：

```powershell
py -3.12 .\_dev_only\setup.py build_ext --inplace
```

**当前不能宣称此命令可完整成功：SIM3 vendor 目录为空。** 原历史记录只有 Gitlink `ed61d828da4fd4cca5f8e8e07e6b856d72137a92`，已核查的本地副本和 Git 对象无法提供源码。必须先取得可验证的原始源码，至少包括 `line2Dup.cpp`、`line2Dup.h`、`MIPP/`、`cuda_icp/icp.cpp` 和 `cuda_icp/scene/edge_scene/edge_scene.cpp`。不要跳过、替换或改写算法来掩盖缺失依赖。

## Embedding 核心

输入为 `native/embedding_core_native.cpp`，输出为 `algorithms/embedding_core.cp312-win_amd64.pyd`，由 `algorithms.embedding` 导入 `compute_prototypes`、`score_topk` 和 `predict_one`。

在应用根目录运行：

```powershell
py -3.12 .\_dev_only\build_embedding_core_pyd.py build_ext --inplace
```

该流程不使用 SIM3 源码。源码和构建脚本现在属于 Git 收录白名单，但是否已提交应通过 `git status`、`git ls-files` 确认；白名单不等于已经进入提交历史。

## 打包与路径约定

`build_py312.ps1` / `build_lite_py312.ps1` 调用 PyInstaller，并收集现有运行二进制；它们不是上述 C++ 模块的重新编译入口。

正常部署不需要 `_dev_only`，但需要当前应用代码、运行二进制及所启用功能的硬件依赖。开发仓库必须保存可重建的源码、版本来源和构建说明。二进制已能运行，不能证明源码重建也已通过。

当前构建脚本用 `SCRIPT_ROOT.parent` 定位应用根目录；Shape 加载器按根模块名导入扩展；NCC 加载器、相机 SDK 路径和 `.spec` 中也有明确目录约定。未来移动源码或二进制时，应同步检查这些路径并分别验证源码运行、测试、原生编译和打包。此次整理不移动文件、不编译、不打包。
