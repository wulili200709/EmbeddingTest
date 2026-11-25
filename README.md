# LC System / EmbeddingTest

工业视觉检测程序。正式源码入口为 `Main.py`，轻量版入口为 `MainLite.py`；使用 Python 3.12 与本机匹配的原生模块和硬件 SDK。

## 目录约定

| 目录 | 内容 | Git 边界 |
| --- | --- | --- |
| `algorithms/`、`application/`、`common/`、`devices/`、`domain/`、`infrastructure/`、`services/`、`ui/` | 正式源码，保留现有导入路径 | 收录 |
| `shape/`、`ncc/`、`third_party/`、`res/` | 定位子系统、运行依赖与资源 | 收录运行所需文件 |
| `_dev_only/` | 原生编译源码、开发测试和历史资料 | 按白名单收录，见其 README |
| `tests/`、`tools/`、`docs/`、`releases/` | 自动化测试、工具、文档、版本记录 | 收录经审查的源码与说明 |
| `experiments/` | 可继续编辑的测试产品和标定项目 | 私有数据默认忽略，保留目录说明 |
| `artifacts/reports/` | 生成的报告、图片和诊断结果 | 忽略 |
| `artifacts/build/` | 构建中间文件、元数据及迁移记录 | 忽略 |
| `artifacts/packages/` | 新发布包，分 `full/`、`lite/` | 忽略 |
| `.qr_session/`、`records/`、`logs/` | 正式产品、模型、账户与运行记录 | 本机数据，保留原路径 |
| `config/` | 默认配置和本机设置 | 收录源码、defaults、examples；忽略本机设置与备份 |
| `.cache/ort_backbones/`、根目录 DLL/PYD | 当前运行需要的模型资产和二进制 | 本轮保持既有收录与位置，不能按普通缓存删除 |

目录整理不改变任何匹配、校准、检测算法，不修改现场配方或原生二进制。旧发布目录 `dist*` 可能包含独有产品数据，保留原地；新构建使用新目录。

## 使用

```powershell
py -3.12 Main.py
py -3.12 MainLite.py
```

独立测试仍可双击 `run_roi_group_demo.bat` 和 `run_lens_distortion_demo.bat`。定位组测试默认打开 `experiments/localization_groups/product_1125`；首次在新机器创建副本需要指定或选择源产品。镜头 Demo 的新参数默认保存到 `experiments/lens_calibration/profiles`，导出结果放到 `artifacts/reports/lens_distortion_demo`。

已有测试副本仍可能引用项目外的原始图片；迁移会保持其路径含义，不会将现场图片自动收录进 Git。

## 旧路径兼容

本机迁移脚本 `tools/migrate_workspace_layout.ps1` 默认只列计划，传 `-Apply` 才执行。迁移采用同磁盘原子目录改名，目的目录有文件时拒绝覆盖；文件被占用时保留原目录并报错。旧目录保留 Windows 目录联接（Junction），确保原有脚本、已保存的路径和快捷方式继续工作：

- `reports` → `artifacts/reports`
- 旧定位组测试产品 → `experiments/localization_groups/product_1125`
- 镜头 Demo 的 `real_image`、`single_view` → `experiments/lens_calibration/` 下同名项目
- `dist-build`、`dist-build-lite`、`.build-metadata` → `artifacts/build/legacy/` 下的原构建记录

联接是兼容入口，不是第二份数据。全新检出使用新的默认路径，不需要复制这些本机联接。迁移前应正常关闭占用目标目录的 Demo；主程序的正式数据路径不在迁移范围内。

## 开发验证与构建

```powershell
py -3.12 -m unittest discover -s tests -p 'test_localization*.py' -v
py -3.12 -m unittest discover -s tests -p 'test_lens*.py' -v
py -3.12 -m unittest discover -s tests -p 'test_demo_workspace_paths.py' -v
py -3.12 -m unittest discover -s tests -p 'test_build_artifact_paths.py' -v
```

打包步骤见 [打包说明](docs/PACKAGING.md)。原生模块重新编译见 [_dev_only/native_build.md](_dev_only/native_build.md)。当前 SIM3 原始依赖源码存在既有缺口；现有二进制仍可运行，不能据此声称能从当前仓库完整重新编译四个后端。

修改 `.gitignore` 不会删除本机文件，也不会自动移除 Git 历史中的旧文件。不要把整个工作目录强制加入 Git；本机账户、现场配置和测试图片保持在本机数据边界内。
