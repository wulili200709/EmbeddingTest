@echo off
setlocal enabledelayedexpansion

title WinRing0 Defender Exclusion Tool
color 0A

:: Check administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo Error: Please run this script as Administrator!
    echo.
    echo How to run:
    echo 1. Right-click this file
    echo 2. Select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo.
echo ===============================================
echo    WinRing0 Defender Exclusion Tool
echo ===============================================
echo.

:: Define full file paths here
set "FILE_LIST="
set "FILE_LIST=!FILE_LIST! "C:\NODKA\NKDIOLC_SDK\Bin\WinRing0.dll""
set "FILE_LIST=!FILE_LIST! "C:\NODKA\NKDIOLC_SDK\Bin\WinRing0.sys""
set "FILE_LIST=!FILE_LIST! "C:\NODKA\NKDIOLC_SDK\Bin\WinRing0x64.dll""
set "FILE_LIST=!FILE_LIST! "C:\NODKA\NKDIOLC_SDK\Bin\WinRing0x64.sys""
set "FILE_LIST=!FILE_LIST! "C:\NODKA\NKDIOLC_SDK\Bin\NKIOLIBx86.dll""
set "FILE_LIST=!FILE_LIST! "C:\NODKA\NKDIOLC_SDK\Bin\NKLCLIBx86.dll""
set "FILE_LIST=!FILE_LIST! "C:\NODKA\NKDIOLC_SDK\Bin\NKIOLIBx64.dll""
set "FILE_LIST=!FILE_LIST! "C:\NODKA\NKDIOLC_SDK\Bin\NKLCLIBx64.dll""
set "FILE_LIST=!FILE_LIST! "C:\path\""
set "FILE_LIST=!FILE_LIST! "C:\path\""
set "FILE_LIST=!FILE_LIST! "C:\path\""
set "FILE_LIST=!FILE_LIST! "C:\path\""

:: Add your custom paths here
:: set "FILE_LIST=!FILE_LIST! "YOUR_CUSTOM_PATH_HERE""

:: Main program - no need to modify below
echo Adding WinRing0 files to Defender exclusions...
echo.

set success_count=0
set fail_count=0

:: Add all predefined file paths
for %%f in (%FILE_LIST%) do (
    if exist %%f (
        echo Found: %%f
        powershell -Command "Add-MpPreference -ExclusionPath '%%f'" >nul 2>&1
        if !errorlevel! equ 0 (
            echo [SUCCESS] Added to exclusions
            set /a success_count+=1
        ) else (
            echo [FAILED] Could not add
            set /a fail_count+=1
        )
    ) else (
        echo [MISSING] File not found: %%f
    )
    echo.
)

echo ===============================================
echo Operation completed!
echo Success: %success_count%
echo Failed: %fail_count%
echo ===============================================
echo.

pause