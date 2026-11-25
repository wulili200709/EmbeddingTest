@echo off
setlocal
cd /d "%~dp0"
py -3.12 tools\lens_distortion_demo.py --single-example %*
if errorlevel 1 (
  echo.
  echo Demo failed. Python 3.12, opencv-python, numpy, scipy and PySide6 are required.
  pause
)
endlocal
