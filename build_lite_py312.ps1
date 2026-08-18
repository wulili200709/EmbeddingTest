param(
    [string]$Version = "",
    [string]$PythonExe = "",
    [string]$OutputPath = "",
    [switch]$SkipArchive
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$releaseTools = Join-Path $scriptRoot "tools\package_release.ps1"
. $releaseTools

$PythonExe = Resolve-LCPython312 -RequestedPythonExe $PythonExe
$Version = Resolve-LCBuildVersion -ScriptRoot $scriptRoot -RequestedVersion $Version
$distPath = if ($OutputPath.Trim()) {
    if ([System.IO.Path]::IsPathRooted($OutputPath)) {
        [System.IO.Path]::GetFullPath($OutputPath)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $scriptRoot $OutputPath))
    }
} else {
    Join-Path $scriptRoot "dist-lite"
}
$workPath = Join-Path $scriptRoot "dist-build-lite"
$specPath = Join-Path $scriptRoot "LC_System_Lite.spec"
$exeDir = Join-Path $distPath "LC System Lite"
$metadataPath = Join-Path $scriptRoot ".build-metadata\lite"
$metadata = New-LCBuildMetadata `
    -ScriptRoot $scriptRoot `
    -MetadataPath $metadataPath `
    -PythonExe $PythonExe `
    -Version $Version `
    -ProductName "LC System Lite" `
    -ExeName "LC System Lite.exe"

if (Test-Path -LiteralPath $exeDir) {
    Remove-Item -LiteralPath $exeDir -Recurse -Force
}

Write-Host "Using Python 3.12 to build LC System Lite V$Version..."
$previousVersionFile = $env:LC_SYSTEM_VERSION_FILE
$previousVersionInfo = $env:LC_SYSTEM_VERSION_INFO
$previousBuildManifest = $env:LC_SYSTEM_BUILD_MANIFEST
$env:LC_SYSTEM_VERSION_FILE = $metadata.VersionFile
$env:LC_SYSTEM_VERSION_INFO = $metadata.ResourceFile
$env:LC_SYSTEM_BUILD_MANIFEST = $metadata.ManifestFile
try {
    & $PythonExe -m PyInstaller --noconfirm --clean --distpath $distPath --workpath $workPath $specPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }
}
finally {
    $env:LC_SYSTEM_VERSION_FILE = $previousVersionFile
    $env:LC_SYSTEM_VERSION_INFO = $previousVersionInfo
    $env:LC_SYSTEM_BUILD_MANIFEST = $previousBuildManifest
}

$exePath = Join-Path $exeDir "LC System Lite.exe"
if (-not (Test-Path -LiteralPath $exePath)) {
    throw "Build output executable not found: $exePath"
}
$mainExePath = Join-Path $exeDir "LC System.exe"
if (Test-Path -LiteralPath $mainExePath) {
    throw "Lite package unexpectedly contains the Main executable: $mainExePath"
}

$sessionDir = Join-Path $exeDir "EmbeddingTest\.qr_session"
if (Test-Path -LiteralPath $sessionDir) {
    Remove-Item -LiteralPath $sessionDir -Recurse -Force
}
New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null

$liteReadmePath = Join-Path $exeDir "README_LITE.txt"
$liteReadme = @"
LC System Lite V$Version

Executable: LC System Lite.exe
Edition: Lite. This is not the LC System Main edition.

This is a portable folder release. The target computer does not need Python,
PySide6, ONNX Runtime, or the other Python packages installed. Extract the
complete ZIP before running the executable; do not copy the EXE by itself.

Offline training and testing with HALCON crops or external images does not need
a camera driver. Physical cameras, capture cards, and IO devices still require
their vendor-provided system drivers.
"@
$liteReadme | Set-Content -LiteralPath $liteReadmePath -Encoding UTF8

$release = Publish-LCReleaseFolder `
    -DistPath $distPath `
    -BuildFolderName "LC System Lite" `
    -ReleaseFolderName "LC System Lite V$Version" `
    -ArchiveFileName "LC_System_Lite_V$Version.zip" `
    -SkipArchive

if (-not $SkipArchive) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archivePath = Join-Path $distPath "LC_System_Lite_V$Version.zip"
    $temporaryArchivePath = Join-Path $distPath "LC_System_Lite_V$Version.building.zip"
    if (Test-Path -LiteralPath $temporaryArchivePath) {
        Remove-Item -LiteralPath $temporaryArchivePath -Force
    }
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $release.ReleasePath,
        $temporaryArchivePath,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $true
    )
    $archiveInfo = Get-Item -LiteralPath $temporaryArchivePath
    if ($archiveInfo.Length -le 0) {
        throw "Lite release archive is empty: $temporaryArchivePath"
    }
    if (Test-Path -LiteralPath $archivePath) {
        Remove-Item -LiteralPath $archivePath -Force
    }
    Move-Item -LiteralPath $temporaryArchivePath -Destination $archivePath
    $release.ArchivePath = $archivePath
}

Write-Host "Built LC System Lite V${Version}: $($release.ReleasePath)"
if ($release.ArchivePath) {
    Write-Host "Created release archive: $($release.ArchivePath)"
}
