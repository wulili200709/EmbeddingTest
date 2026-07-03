$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$distPath = Join-Path $scriptRoot "dist"
$workPath = Join-Path $scriptRoot "dist-build-lite"
$specPath = Join-Path $scriptRoot "LC_System_Lite.spec"
$exeDir = Join-Path $distPath "LC System Lite"

if (Test-Path -LiteralPath $exeDir) {
    Remove-Item -LiteralPath $exeDir -Recurse -Force
}

Write-Host "Using Python 3.12 to build LC System Lite..."
& py -3.12 -m PyInstaller --noconfirm --clean --distpath $distPath --workpath $workPath $specPath

$exePath = Join-Path $exeDir "LC System Lite.exe"
if (-not (Test-Path -LiteralPath $exePath)) {
    throw "Build output executable not found: $exePath"
}

$sessionDir = Join-Path $exeDir "EmbeddingTest\.qr_session"
if (Test-Path -LiteralPath $sessionDir) {
    Remove-Item -LiteralPath $sessionDir -Recurse -Force
}
New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null

Write-Host "Built LC System Lite: $exePath"
