# 开发资料与源码边界

`_dev_only` 不随正常运行版发布，但并不等于“可删除的测试文件”。这里同时保存正式算法的原生源码、构建输入、回归测试和历史资料。目录整理不改变算法、运行路径或已有二进制。

## 应纳入版本控制

| 路径 | 用途 |
| --- | --- |
| `native/line2dup_native.cpp` | 四个 Shape 原生后端共用的 Python/C++ 桥接源码 |
| `native/embedding_core_native.cpp` | `algorithms.embedding_core` 原生实现 |
| `setup.py` | 构建 `shape_original`、`shape_fusion`、`shape_fusionv2`、`shape_sim3` |
| `build_embedding_core_pyd.py` | 构建 `algorithms/embedding_core.cp312-win_amd64.pyd` |
| `vendor/_third_party_shape_based_matching/` | Original 后端源码及所需头文件 |
| `vendor/_third_party_shape_based_matching_fusion_fix_memo/` | Fusion / Fusion V2 后端源码及所需头文件 |
| `vendor/_third_party_shape_based_matching_sim3/` | SIM3 后端源码；当前缺失状态见下文 |
| `vendor/eigen/` | 编译需要的 Eigen 头文件 |
| `tests/**/*.py`、`manual_capture/**/*.py` | 原有测试、诊断和采集脚本入口；保留当前位置 |
| `README.md`、`native_build.md` | 分类说明与构建条件 |

第三方源码应保留许可证和版本来源。恢复目录时不能只留下一个缺少 `.gitmodules` 的 Git 子模块指针；需要能从仓库取得完整源码，或具有明确地址与固定版本的完整依赖管理方式。

## 本地资料、生成物和归档

- `legacy/`、`root_files/`、旧 `.pyd.bak`：历史实验和备份，不参与当前运行或原生编译。已有 Git 跟踪记录不会被忽略规则自动删除；是否移出工作树应另行审查。
- `images/`、截图、手工采集图、日志：本地开发资料，不随源码自动收录。
- `build/`、vendor 内的 `build/`、`build_bench/`、CMake 缓存、`.obj`、`.pdb` 等：可重建的中间产物，不收录。
- 未列入白名单的旧 `tools/`、`docs/` 等目录继续保持现有归档边界；已经跟踪的文件仍受 Git 管理。新增必要工具应放入应用根目录的 `tools/`，或经审查添加精确例外。
- 新增测试用小型合成夹具时，应单独审查并增加精确白名单，不要通过强制添加整个目录把现场图片、模型和日志一起收录。

根目录 `experiments/` 用于私有实验输入和输出，当前只开放其 `README.md`；`artifacts/` 存放本地生成物。不要将产品图片、现场配方或模型当作通用源码夹具提交。

## Git 规则与运行文件

`.gitignore` 忽略 `_dev_only` 的内容，而不是排除父目录本身，再逐层开放源码与脚本。这使新增关键源码可以被 Git 正常发现。修改忽略规则不会自动提交文件，也不会取消已经跟踪的文件。

当前运行仍使用应用根目录的 `shape_*.pyd`、`NccMatchNative.dll`、`opencv_world4130.dll`，以及 `algorithms/embedding_core.cp312-win_amd64.pyd`。`shape/`、`ncc/`、`third_party/MvImport/`、`third_party/nkio/` 是正式运行模块或依赖，不属于本目录的实验归档。此次规则整理不移动或重新编译这些文件。

`config/` 下的现场密码、相机、运行与界面设置，以及对应 `.bak` 应保持本机配置边界；`config/defaults/` 和 `config/examples/` 仍可收录。忽略规则对已跟踪配置不追溯生效；取消跟踪应在确认默认初始化和打包行为后单独进行，不能删除现场文件来代替。

## 已知源码缺口（2026-11-25 核查）

SIM3 vendor 当前为空，不能声称已可从本仓库完整重编四个 Shape 后端。历史提交 `53931b8` 曾以 Gitlink（mode `160000`）记录 `_third_party_shape_based_matching_sim3`，指向 `ed61d828da4fd4cca5f8e8e07e6b856d72137a92`；该记录没有包含源码文件。

已检查本机当前目录、父目录同名目录、备份和两个副本中的五个候选目录，均为空；检查到的本地 Git 库也没有上述目标对象。需从原始仓库、可验证备份或原构建机取得对应源码后恢复。不能用其他算法实现替代，也不能把现有 `.pyd` 反推为已验证源码。

详见 [原生构建说明](native_build.md)。正常运行和使用已有二进制打包，与从源码重新编译是不同的验证目标。
