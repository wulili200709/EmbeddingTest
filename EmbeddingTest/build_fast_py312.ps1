$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$distPath = Join-Path $scriptRoot "dist-fast"
$workPath = Join-Path $scriptRoot "dist-build-fast"
$specPath = Join-Path $scriptRoot "LC_System.spec"
$previousFastBuild = $env:LC_FAST_BUILD

Write-Host "Using Python 3.12 to build LC System FAST..."
$env:LC_FAST_BUILD = "1"

try {
    if (Test-Path $distPath) {
        Remove-Item -LiteralPath $distPath -Recurse -Force
    }
    $collectToc = Join-Path $workPath "LC_System\COLLECT-00.toc"
    if (Test-Path $collectToc) {
        Remove-Item -LiteralPath $collectToc -Force
    }

    & py -3.12 -m PyInstaller --noconfirm --distpath $distPath --workpath $workPath $specPath
}
finally {
    if ($null -eq $previousFastBuild) {
        Remove-Item Env:LC_FAST_BUILD -ErrorAction SilentlyContinue
    }
    else {
        $env:LC_FAST_BUILD = $previousFastBuild
    }
}
