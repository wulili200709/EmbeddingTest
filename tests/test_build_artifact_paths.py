from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe")


def ps_quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


# This command only creates tiny fixture files. It never starts Python,
# PyInstaller, the application, or vendor drivers.
FAKE_PYTHON = r"""
$ErrorActionPreference = 'Stop'
$arguments = @($args | ForEach-Object { [string]$_ })
@{
    arguments = $arguments
    version_file = $env:LC_SYSTEM_VERSION_FILE
    version_info = $env:LC_SYSTEM_VERSION_INFO
    manifest_file = $env:LC_SYSTEM_BUILD_MANIFEST
} | ConvertTo-Json -Compress | Add-Content -LiteralPath $env:LC_PATH_TEST_CALLS -Encoding UTF8

function Get-ArgumentValue([string]$Name) {
    $index = [Array]::IndexOf($arguments, $Name)
    if ($index -lt 0 -or $index + 1 -ge $arguments.Count) { throw "Missing argument: $Name" }
    return $arguments[$index + 1]
}

if ($arguments[0] -eq '-c') { exit 0 }
if ([System.IO.Path]::GetFileName($arguments[0]) -eq 'generate_version_info.py') {
    $version = Get-ArgumentValue '--version'
    [System.IO.File]::WriteAllText((Get-ArgumentValue '--version-file'), "$version`n")
    [System.IO.File]::WriteAllText((Get-ArgumentValue '--resource-file'), 'fixture resource')
    [System.IO.File]::WriteAllText((Get-ArgumentValue '--manifest-file'), (@{version=$version} | ConvertTo-Json))
    exit 0
}
if ([System.IO.Path]::GetFileName($arguments[0]) -eq 'generate_seed_audit_db.py') {
    [System.IO.File]::WriteAllText($arguments[1], 'fixture seed')
    exit 0
}
if ($arguments[0] -eq '-m' -and $arguments[1] -eq 'PyInstaller') {
    $spec = [System.IO.Path]::GetFileName($arguments[-1])
    $name = if ($spec -eq 'LC_System_Lite.spec') { 'LC System Lite' } elseif ($spec -eq 'LC_System.spec') { 'LC System' } else { throw "Unexpected spec: $spec" }
    $stage = Join-Path (Get-ArgumentValue '--distpath') $name
    $metadata = Join-Path $stage 'EmbeddingTest'
    $session = Join-Path $metadata '.qr_session'
    New-Item -ItemType Directory -Path $session -Force | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $stage "$name.exe"), 'fixture executable')
    [System.IO.File]::WriteAllText((Join-Path $session 'fixture-product.txt'), 'fixture product')
    Copy-Item -LiteralPath $env:LC_SYSTEM_VERSION_FILE -Destination $metadata
    Copy-Item -LiteralPath $env:LC_SYSTEM_BUILD_MANIFEST -Destination $metadata
    exit 0
}
if ([System.IO.Path]::GetFileName($arguments[0]) -eq 'record_release.py') { exit 0 }
throw "Unexpected fake Python arguments: $arguments"
"""


@unittest.skipUnless(POWERSHELL, "Windows PowerShell required")
class BuildArtifactPathTests(unittest.TestCase):
    def run_powershell(self, script: Path, *, cwd: Path) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            cwd=cwd, capture_output=True, timeout=45,
        )
        self.assertEqual(
            result.returncode, 0,
            result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace"),
        )
        return result

    def test_powershell_sources_parse_without_executing(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "parse.ps1"
            files = [ROOT / name for name in (
                "build_py312.ps1", "build_lite_py312.ps1", "tools/package_release.ps1",
            )]
            script.write_text(
                "$ErrorActionPreference = 'Stop'\n"
                + "\n".join(
                    "$tokens = $null; $parseErrors = $null; "
                    f"[System.Management.Automation.Language.Parser]::ParseFile({ps_quote(path)}, "
                    "[ref]$tokens, [ref]$parseErrors) | Out-Null; "
                    "if ($parseErrors.Count) { throw ($parseErrors | Out-String) }"
                    for path in files
                ), encoding="utf-8-sig",
            )
            self.run_powershell(script, cwd=Path(directory))

    def test_path_resolution_is_pure_and_preserves_output_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project with spaces"
            script = Path(directory) / "paths.ps1"
            output = Path(directory) / "paths.json"
            absolute = Path(directory) / "external releases"
            script.write_text(
                "$ErrorActionPreference = 'Stop'\n"
                f". {ps_quote(ROOT / 'tools/package_release.ps1')}\n"
                "$results = @(\n"
                + "\n".join(
                    f"    Resolve-LCBuildPaths -ScriptRoot {ps_quote(root)} -Edition {ps_quote(edition)}"
                    + (f" -OutputPath {ps_quote(override)}" if override else "")
                    for edition in ("full", "lite")
                    for override in ("", "custom releases", str(absolute))
                )
                + f"\n)\n$results | ConvertTo-Json | Set-Content -LiteralPath {ps_quote(output)} -Encoding UTF8\n",
                encoding="utf-8-sig",
            )
            self.run_powershell(script, cwd=Path(directory))
            records = json.loads(output.read_text(encoding="utf-8-sig"))
            self.assertEqual(len(records), 6)
            for index, edition in enumerate(("full", "lite")):
                for offset, expected in enumerate((root / "artifacts/packages" / edition, root / "custom releases", absolute)):
                    record = records[index * 3 + offset]
                    self.assertEqual(Path(record["DistPath"]), expected)
                    self.assertEqual(Path(record["WorkPath"]), root / "artifacts/build" / edition)
                    self.assertEqual(Path(record["MetadataPath"]), root / "artifacts/build/metadata" / edition)
            self.assertFalse(root.exists(), "Resolving paths must not write directories")
            self.assertFalse(absolute.exists())

    def exercise_build(self, edition: str, override_kind: str = "default"):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "project with spaces"
            (root / "tools").mkdir(parents=True)
            for name in ("build_py312.ps1", "build_lite_py312.ps1", "tools/package_release.ps1"):
                shutil.copy2(ROOT / name, root / name)
            (root / "VERSION").write_text("9.8.7\n", encoding="utf-8")
            driver = temporary / "NKDIOLC_SDK/Lib/x64/WinRing0x64.sys"
            driver.parent.mkdir(parents=True)
            driver.write_bytes(b"fixture driver")
            fake_python = root / "fake-python.ps1"
            fake_python.write_text(FAKE_PYTHON, encoding="utf-8-sig")
            calls_path = temporary / "calls.jsonl"
            output_arg = None
            if override_kind == "relative":
                output_arg = "custom packages"
                dist = root / output_arg
            elif override_kind == "absolute":
                dist = temporary / "external packages"
                output_arg = str(dist)
            else:
                dist = root / "artifacts/packages" / edition
            version = "8.7.6" if output_arg else "9.8.7"
            build_script = root / ("build_lite_py312.ps1" if edition == "lite" else "build_py312.ps1")
            invoke = (
                f"& {ps_quote(build_script)} -PythonExe {ps_quote(fake_python)} -SkipArchive"
                + (f" -OutputPath {ps_quote(output_arg)} -Version {ps_quote(version)}" if output_arg else "")
            )
            wrapper = temporary / "exercise.ps1"
            wrapper.write_text(
                "$ErrorActionPreference = 'Stop'\n"
                f"$env:LC_PATH_TEST_CALLS = {ps_quote(calls_path)}\n"
                "$env:LC_SYSTEM_VERSION_FILE = 'before-version'\n"
                "$env:LC_SYSTEM_VERSION_INFO = 'before-resource'\n"
                "$env:LC_SYSTEM_BUILD_MANIFEST = 'before-manifest'\n"
                + invoke + "\n"
                + "if ($env:LC_SYSTEM_VERSION_FILE -ne 'before-version' -or "
                "$env:LC_SYSTEM_VERSION_INFO -ne 'before-resource' -or "
                "$env:LC_SYSTEM_BUILD_MANIFEST -ne 'before-manifest') { throw 'Build leaked metadata environment' }\n",
                encoding="utf-8-sig",
            )
            # CWD differs from the script root to exercise relative -OutputPath semantics.
            self.run_powershell(wrapper, cwd=temporary)
            calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8-sig").splitlines()]
            builds = [call for call in calls if call["arguments"][:2] == ["-m", "PyInstaller"]]
            self.assertEqual(len(builds), 1)
            build = builds[0]
            arguments = build["arguments"]
            self.assertEqual(Path(arguments[arguments.index("--distpath") + 1]), dist)
            self.assertEqual(Path(arguments[arguments.index("--workpath") + 1]), root / "artifacts/build" / edition)
            metadata = root / "artifacts/build/metadata" / edition
            self.assertEqual(Path(build["version_file"]), metadata / "VERSION")
            self.assertEqual(Path(build["version_info"]), metadata / "version_info.txt")
            self.assertEqual(Path(build["manifest_file"]), metadata / "build_manifest.json")
            self.assertEqual((metadata / "VERSION").read_text().strip(), version)
            name = "LC System Lite" if edition == "lite" else "LC System"
            release = dist / f"{name} V{version}"
            self.assertTrue((release / f"{name}.exe").is_file())
            self.assertFalse((dist / name).exists(), "Publishing should move the staging folder")
            self.assertFalse(list(dist.glob("*.zip")), "-SkipArchive must retain its behavior")
            recorder_calls = [call for call in calls if Path(call["arguments"][0]).name == "record_release.py"]
            self.assertEqual(len(recorder_calls), 1, "Release history invocation must be preserved")
            if edition == "full":
                self.assertTrue((metadata / "audit.db").is_file())
                self.assertTrue((release / "WinRing0x64.sys").is_file())
                self.assertFalse((root / "records").exists(), "Build seed must not enter runtime records")
            else:
                self.assertTrue((release / "EmbeddingTest/.qr_session").is_dir())
                self.assertFalse((release / "EmbeddingTest/.qr_session/fixture-product.txt").exists())
            for obsolete in ("dist", "dist-lite", "dist-build", "dist-build-lite", ".build-metadata"):
                self.assertFalse((root / obsolete).exists(), obsolete)

    def test_full_default_pipeline_with_stub_python(self):
        self.exercise_build("full")

    def test_lite_default_pipeline_with_stub_python(self):
        self.exercise_build("lite")

    def test_relative_output_pipeline_with_stub_python(self):
        self.exercise_build("full", "relative")

    def test_absolute_output_pipeline_with_stub_python(self):
        self.exercise_build("lite", "absolute")


class BuildSpecPathTests(unittest.TestCase):
    def test_specs_parse_and_full_seed_uses_build_metadata(self):
        for name in ("LC_System.spec", "LC_System_Lite.spec"):
            ast.parse((ROOT / name).read_text(encoding="utf-8-sig"), filename=name)
        tree = ast.parse((ROOT / "LC_System.spec").read_text(encoding="utf-8-sig"))
        scope = {"APP_ROOT": ROOT}
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in {"BUILD_METADATA_ROOT", "seed_audit_db"}:
                scope[target.id] = eval(compile(ast.Expression(statement.value), "spec-path", "eval"), {"__builtins__": {}}, scope)
        self.assertEqual(scope["seed_audit_db"], ROOT / "artifacts/build/metadata/full/audit.db")


if __name__ == "__main__":
    unittest.main()
