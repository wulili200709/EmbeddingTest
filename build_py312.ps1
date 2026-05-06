$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$distPath = Join-Path $scriptRoot "dist"
$workPath = Join-Path $scriptRoot "dist-build"
$specPath = Join-Path $scriptRoot "LC_System.spec"
$winRingSysCandidates = @(
    (Join-Path $scriptRoot "..\NKDIOLC_SDK\Lib\x64\WinRing0x64.sys"),
    (Join-Path $scriptRoot "..\NKDIOLC_SDK\Bin\WinRing0x64.sys"),
    (Join-Path $scriptRoot "..\NKDIOLC_SDK\Sample\CPP\NK_IO_LC_TEST_Console\SDKLib\Lib\x64\WinRing0x64.sys"),
    (Join-Path $scriptRoot "..\NKDIOLC_SDK\Sample\Qt\NK_IO_LC_TEST_Qt\SDKLib\Lib\x64\WinRing0x64.sys")
)
$winRingSys = $winRingSysCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if (-not $winRingSys) {
    throw "WinRing0x64.sys was not found in the NKDIOLC_SDK paths required for packaging."
}

Write-Host "Using Python 3.12 to build LC System..."
& py -3.12 -m PyInstaller --noconfirm --clean --distpath $distPath --workpath $workPath $specPath

$exeDir = Join-Path $distPath "LC System"
if (-not (Test-Path -LiteralPath $exeDir)) {
    throw "Build output directory not found: $exeDir"
}
Copy-Item -LiteralPath $winRingSys -Destination (Join-Path $exeDir "WinRing0x64.sys") -Force
Write-Host "Copied WinRing0x64.sys to $exeDir"
