function Resolve-LCBuildPaths {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptRoot,
        [Parameter(Mandatory = $true)][ValidateSet("full", "lite")][string]$Edition,
        [string]$OutputPath = ""
    )

    $projectRoot = [System.IO.Path]::GetFullPath($ScriptRoot)
    $distPath = if ($OutputPath.Trim()) {
        if ([System.IO.Path]::IsPathRooted($OutputPath)) {
            [System.IO.Path]::GetFullPath($OutputPath)
        } else {
            [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputPath))
        }
    } else {
        Join-Path $projectRoot "artifacts\packages\$Edition"
    }
    $buildFolderName = if ($Edition -eq "lite") { "LC System Lite" } else { "LC System" }
    $buildPath = [System.IO.Path]::GetFullPath((Join-Path $distPath $buildFolderName))
    if ([System.IO.Path]::GetDirectoryName($buildPath).TrimEnd('\') -ne $distPath.TrimEnd('\')) {
        throw "Unsafe build output directory: $buildPath"
    }

    return [PSCustomObject]@{
        DistPath = $distPath
        WorkPath = Join-Path $projectRoot "artifacts\build\$Edition"
        MetadataPath = Join-Path $projectRoot "artifacts\build\metadata\$Edition"
        BuildPath = $buildPath
    }
}


function Resolve-LCPython312 {
    param([string]$RequestedPythonExe = "")

    $candidates = @()
    if ($RequestedPythonExe.Trim()) {
        $candidates += $RequestedPythonExe.Trim()
    }
    if ($env:LOCALAPPDATA) {
        $candidates += (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe")
    }
    if ($env:ProgramFiles) {
        $candidates += (Join-Path $env:ProgramFiles "Python312\python.exe")
    }
    $pathPython = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pathPython) {
        $candidates += $pathPython.Source
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            continue
        }
        & $candidate -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
        if ($LASTEXITCODE -eq 0) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    throw "Python 3.12 was not found. Install it or pass -PythonExe C:\path\to\python.exe."
}


function Resolve-LCBuildVersion {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptRoot,
        [string]$RequestedVersion = ""
    )

    $resolvedVersion = $RequestedVersion.Trim()
    if (-not $resolvedVersion) {
        $versionPath = Join-Path $ScriptRoot "VERSION"
        if (-not (Test-Path -LiteralPath $versionPath)) {
            throw "VERSION file not found: $versionPath"
        }
        $resolvedVersion = (Get-Content -LiteralPath $versionPath -Raw).Trim()
    }
    if ($resolvedVersion -notmatch '^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(\.(0|[1-9]\d*))?([-+][0-9A-Za-z.-]+)?$') {
        throw "Invalid version '$resolvedVersion'. Use a value such as 3.1.0 or 3.1.0.4."
    }
    return $resolvedVersion
}


function New-LCBuildMetadata {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptRoot,
        [Parameter(Mandatory = $true)][string]$MetadataPath,
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$Version,
        [Parameter(Mandatory = $true)][string]$ProductName,
        [Parameter(Mandatory = $true)][string]$ExeName
    )

    New-Item -ItemType Directory -Path $MetadataPath -Force | Out-Null
    $versionFile = Join-Path $MetadataPath "VERSION"
    $resourceFile = Join-Path $MetadataPath "version_info.txt"
    $manifestFile = Join-Path $MetadataPath "build_manifest.json"
    $generator = Join-Path $ScriptRoot "tools\generate_version_info.py"

    $generatorOutput = & $PythonExe $generator `
        --version $Version `
        --product-name $ProductName `
        --exe-name $ExeName `
        --project-root $ScriptRoot `
        --version-file $versionFile `
        --resource-file $resourceFile `
        --manifest-file $manifestFile
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate package version metadata."
    }
    foreach ($outputLine in $generatorOutput) {
        Write-Host $outputLine
    }

    return [PSCustomObject]@{
        VersionFile = $versionFile
        ResourceFile = $resourceFile
        ManifestFile = $manifestFile
    }
}


function Publish-LCReleaseFolder {
    param(
        [Parameter(Mandatory = $true)][string]$DistPath,
        [Parameter(Mandatory = $true)][string]$BuildFolderName,
        [Parameter(Mandatory = $true)][string]$ReleaseFolderName,
        [Parameter(Mandatory = $true)][string]$ArchiveFileName,
        [string]$ScriptRoot = "",
        [string]$PythonExe = "",
        [string]$ExeName = "",
        [switch]$SkipArchive
    )

    $distFullPath = [System.IO.Path]::GetFullPath($DistPath).TrimEnd('\')
    $sourcePath = Join-Path $distFullPath $BuildFolderName
    $releasePath = Join-Path $distFullPath $ReleaseFolderName
    $archivePath = Join-Path $distFullPath $ArchiveFileName

    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Build output directory not found: $sourcePath"
    }
    if ([System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($sourcePath)).TrimEnd('\') -ne $distFullPath) {
        throw "Unsafe build directory: $sourcePath"
    }
    if ([System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($releasePath)).TrimEnd('\') -ne $distFullPath) {
        throw "Unsafe release directory: $releasePath"
    }
    if ([System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($archivePath)).TrimEnd('\') -ne $distFullPath) {
        throw "Unsafe release archive: $archivePath"
    }

    if ((Test-Path -LiteralPath $releasePath) -or (Test-Path -LiteralPath $archivePath)) {
        $rebuildSuffix = Get-Date -Format "yyyyMMdd_HHmmss_fffffff"
        $releasePath = Join-Path $distFullPath "$ReleaseFolderName-$rebuildSuffix"
        $archiveStem = [System.IO.Path]::GetFileNameWithoutExtension($ArchiveFileName)
        $archivePath = Join-Path $distFullPath "$archiveStem-$rebuildSuffix.zip"
    }
    Move-Item -LiteralPath $sourcePath -Destination $releasePath

    if ($ScriptRoot -and $PythonExe -and $ExeName) {
        $recorder = Join-Path $ScriptRoot "tools\record_release.py"
        & $PythonExe $recorder --project-root $ScriptRoot --release-dir $releasePath --exe-name $ExeName | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) {
            throw "Release executable was built but its automatic version record failed."
        }
    }

    if (-not $SkipArchive) {
        # ZipFile streams large ML releases and includes hidden model files.
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $temporaryArchivePath = "$archivePath.building"
        [System.IO.Compression.ZipFile]::CreateFromDirectory(
            $releasePath, $temporaryArchivePath,
            [System.IO.Compression.CompressionLevel]::Optimal, $true
        )
        Move-Item -LiteralPath $temporaryArchivePath -Destination $archivePath
    }

    return [PSCustomObject]@{
        ReleasePath = $releasePath
        ArchivePath = if ($SkipArchive) { "" } else { $archivePath }
    }
}
