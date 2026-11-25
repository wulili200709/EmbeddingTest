param(
    [string]$WorkspaceRoot = (Split-Path -Parent $PSScriptRoot),
    [ValidateSet('All', 'Experiments', 'Build')]
    [string]$Section = 'All',
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $WorkspaceRoot).Path).TrimEnd('\')
if (-not (Test-Path -LiteralPath (Join-Path $workspace 'Main.py') -PathType Leaf)) {
    throw "Not an EmbeddingTest workspace: $workspace"
}

function Assert-WorkspacePathContained {
    param(
        [string]$PathToCheck,
        [System.Collections.Generic.HashSet[string]]$ActiveLinks
    )

    $checkedPath = [System.IO.Path]::GetFullPath($PathToCheck).TrimEnd('\')
    if ($checkedPath -ne $workspace -and
        -not $checkedPath.StartsWith($workspace + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the workspace: $checkedPath"
    }

    # Include the path itself and recursively check link targets. A target
    # inside the workspace can itself traverse another link to an outside path.
    $parent = $checkedPath
    while ($parent -and $parent -ne $workspace) {
        if (Test-Path -LiteralPath $parent) {
            $entry = Get-Item -LiteralPath $parent -Force
            if ($entry.LinkType) {
                if (-not $ActiveLinks.Add($parent)) {
                    throw "Circular workspace link: $parent"
                }
                try {
                    $rawTarget = [string](@($entry.Target)[0])
                    if (-not $rawTarget.Trim()) {
                        throw "Cannot resolve workspace link: $parent"
                    }
                    $target = if ([System.IO.Path]::IsPathRooted($rawTarget)) {
                        $rawTarget
                    } else {
                        Join-Path (Split-Path -Parent $parent) $rawTarget
                    }
                    Assert-WorkspacePathContained -PathToCheck $target -ActiveLinks $ActiveLinks
                }
                finally {
                    $null = $ActiveLinks.Remove($parent)
                }
            }
        }
        $parent = Split-Path -Parent $parent
    }
}

function Get-WorkspacePath([string]$RelativePath) {
    $fullPath = [System.IO.Path]::GetFullPath((Join-Path $workspace $RelativePath))
    $activeLinks = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    Assert-WorkspacePathContained -PathToCheck $fullPath -ActiveLinks $activeLinks
    return $fullPath
}

$mappings = @(
    @('reports', 'artifacts\reports', 'Reports'),
    @('reports\localization_groups_demo\product_1125', 'experiments\localization_groups\product_1125', 'Experiments'),
    @('reports\lens_distortion_demo\real_image', 'experiments\lens_calibration\real_image', 'Experiments'),
    @('reports\lens_distortion_demo\single_view', 'experiments\lens_calibration\single_view', 'Experiments'),
    @('dist-build', 'artifacts\build\legacy\full-work', 'Build'),
    @('dist-build-lite', 'artifacts\build\legacy\lite-work', 'Build'),
    @('.build-metadata', 'artifacts\build\legacy\metadata', 'Build')
)

foreach ($mapping in $mappings) {
    if ($Section -ne 'All' -and $mapping[2] -ne $Section) { continue }
    $source = Get-WorkspacePath $mapping[0]
    $destination = Get-WorkspacePath $mapping[1]
    if (-not (Test-Path -LiteralPath $source)) {
        Write-Output "Absent: $($mapping[0])"
        continue
    }
    $sourceEntry = Get-Item -LiteralPath $source -Force
    if ($sourceEntry.LinkType) {
        $target = [System.IO.Path]::GetFullPath([string](@($sourceEntry.Target)[0])).TrimEnd('\')
        if ($target -ne $destination.TrimEnd('\')) {
            throw "Unexpected existing link: $source -> $target"
        }
        Write-Output "Already migrated: $($mapping[0])"
        continue
    }
    if (-not $sourceEntry.PSIsContainer) {
        throw "Expected a directory: $source"
    }
    if (Test-Path -LiteralPath $destination) {
        $destinationEntry = Get-Item -LiteralPath $destination -Force
        if (-not $destinationEntry.PSIsContainer -or $destinationEntry.LinkType) {
            throw "Destination contains data; refusing to merge or replace it: $destination"
        }
        $children = @(Get-ChildItem -LiteralPath $destination -Force -Recurse)
        if (@($children | Where-Object { -not $_.PSIsContainer -or $_.LinkType }).Count) {
            throw "Destination contains data; refusing to merge or replace it: $destination"
        }
        if ($Apply) {
            # A Demo may already have created empty output directories. Remove
            # empty directories only; never recursively delete their contents.
            foreach ($entry in ($children | Sort-Object { $_.FullName.Length } -Descending)) {
                $checked = Get-WorkspacePath ($entry.FullName.Substring($workspace.Length + 1))
                [System.IO.Directory]::Delete($checked, $false)
            }
            [System.IO.Directory]::Delete($destination, $false)
        }
    }
    if (-not $Apply) {
        Write-Output "Plan: $($mapping[0]) -> $($mapping[1]) (keep old path as a junction)"
        continue
    }
    $null = New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force
    # Same-volume directory rename is atomic: a locked file must not leave a
    # partially moved tree (PowerShell Move-Item can otherwise recurse).
    [System.IO.Directory]::Move($source, $destination)
    try {
        $null = New-Item -ItemType Junction -Path $source -Target $destination
    }
    catch {
        # Restore the original pathname if creating its compatibility link fails.
        if (-not (Test-Path -LiteralPath $source)) {
            $checkedDestination = Get-WorkspacePath $mapping[1]
            $checkedSource = Get-WorkspacePath $mapping[0]
            [System.IO.Directory]::Move($checkedDestination, $checkedSource)
        }
        throw
    }
    Write-Output "Migrated: $($mapping[0]) -> $($mapping[1])"
}

if ($Apply) {
    foreach ($relative in @('artifacts\reports', 'artifacts\build', 'artifacts\packages',
                           'experiments\localization_groups', 'experiments\lens_calibration\profiles')) {
        $null = New-Item -ItemType Directory -Path (Get-WorkspacePath $relative) -Force
    }
}
