@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "VERSION_ARG=%~1"

if "%VERSION_ARG%"=="" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%build_lite_py312.ps1"
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%build_lite_py312.ps1" -Version "%VERSION_ARG%"
)

if errorlevel 1 (
    echo.
    echo LC System Lite build failed.
    exit /b 1
)

echo.
echo LC System Lite build completed.
endlocal
