# LC System 打包发布

项目使用 PyInstaller 单目录模式。运行电脑不需要安装 Python 或执行 `pip install`，但必须复制完整发布文件夹，不能只复制 EXE。

## 正式版

build_release.bat


最简单的方式是在项目目录运行：

```bat
build_release.bat 3.1.0
```

批处理入口会自动绕过本机 PowerShell 脚本执行策略。也可以直接使用：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build_py312.ps1 -Version 3.1.0
```

输出：

```text
artifacts\packages\full\LC System V3.1.0\
artifacts\packages\full\LC_System_V3.1.0.zip
```

## 轻量版

```bat
build_lite_release.bat 3.1.0
```

输出：

```text
artifacts\packages\lite\LC System Lite V3.1.0\
artifacts\packages\lite\LC_System_Lite_V3.1.0.zip
```

Full 和 Lite 默认分别输出到 `artifacts\packages\full` 和 `artifacts\packages\lite`。
发布包入口固定为 `LC System Lite.exe`，包内还会生成 `README_LITE.txt` 作为版本标识。

Lite 使用 HALCON 小图或普通外部图片进行离线训练、测试时，目标电脑不需要
安装 Python、项目依赖或相机驱动；必须解压并复制完整发布目录，不能只复制 EXE。
如果需要连接实体相机、采集卡或 IO 硬件，仍然需要安装对应厂商的系统级驱动。

如果省略 `-Version`，构建脚本读取项目根目录的 `VERSION`。如只需要发布文件夹、不需要 ZIP，可增加 `-SkipArchive`。

脚本会自动查找本机 Python 3.12。如果 Python 安装在自定义位置，可指定：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build_py312.ps1 -Version 3.1.0 -PythonExe "D:\Python312\python.exe"
```

Lite 指定自定义 Python 路径时使用：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build_lite_py312.ps1 -Version 3.1.0 -PythonExe "D:\Python312\python.exe"
```

需要把产物输出到其他目录时，可增加 `-OutputPath "D:\LC-Releases"`。该参数直接指定发布文件夹和 ZIP 的父目录，不会再追加 `full` 或 `lite`；相对路径以项目根目录为基准，与启动命令所在目录无关。

构建中间文件集中存放于：

| 用途 | Full | Lite |
| --- | --- | --- |
| PyInstaller 工作文件 | `artifacts\build\full` | `artifacts\build\lite` |
| 临时版本信息和构建清单 | `artifacts\build\metadata\full` | `artifacts\build\metadata\lite` |

Full 打包使用的干净账户数据库也生成在 `artifacts\build\metadata\full\audit.db`，再复制到包内 `EmbeddingTest\records`。项目运行数据 `.qr_session`、`config`、`records` 的位置不变。

旧的 `dist*` 发布目录保留原位，新构建不会自动移动或覆盖它们。根目录原来的 `dist-build`、`dist-build-lite` 和 `.build-metadata` 不再被上述构建入口使用；清理旧中间文件前应确认没有其他旧脚本正在使用。同版本重复发布会另加时间后缀，既有发布历史和自动版本记录继续保留在 `releases`。

版本号会同时写入：

- 软件界面版本；
- Windows EXE 文件版本和产品版本；
- 发布文件夹与 ZIP 文件名；
- 包内 `EmbeddingTest\VERSION`；
- 包内 `EmbeddingTest\build_manifest.json`。

打包电脑需要 Python 3.12、PyInstaller 及 `requirements.txt` 中的依赖。运行电脑无需这些 Python 环境；相机或 IO 硬件所需的系统级驱动仍需按硬件要求安装。
