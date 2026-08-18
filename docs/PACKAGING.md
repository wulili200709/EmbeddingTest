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
dist\LC System V3.1.0\
dist\LC_System_V3.1.0.zip
```

## 轻量版

```bat
build_lite_release.bat 3.1.0
```

输出：

```text
dist-lite\LC System Lite V3.1.0\
dist-lite\LC_System_Lite_V3.1.0.zip
```

Lite 默认使用独立的 `dist-lite` 目录，不会和 Main 正式版的 `dist` 目录混放。
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

需要把产物输出到其他目录时，可增加 `-OutputPath "D:\LC-Releases"`。

版本号会同时写入：

- 软件界面版本；
- Windows EXE 文件版本和产品版本；
- 发布文件夹与 ZIP 文件名；
- 包内 `EmbeddingTest\VERSION`；
- 包内 `EmbeddingTest\build_manifest.json`。

打包电脑需要 Python 3.12、PyInstaller 及 `requirements.txt` 中的依赖。运行电脑无需这些 Python 环境；相机或 IO 硬件所需的系统级驱动仍需按硬件要求安装。
