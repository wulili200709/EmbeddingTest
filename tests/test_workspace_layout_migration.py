from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "tools/migrate_workspace_layout.ps1"
POWERSHELL = shutil.which("powershell.exe")
BUILD_MAPPINGS = (
    ("dist-build", "artifacts/build/legacy/full-work"),
    ("dist-build-lite", "artifacts/build/legacy/lite-work"),
    (".build-metadata", "artifacts/build/legacy/metadata"),
)
EXPERIMENT_MAPPINGS = (
    ("reports/localization_groups_demo/product_1125", "experiments/localization_groups/product_1125"),
    ("reports/lens_distortion_demo/real_image", "experiments/lens_calibration/real_image"),
    ("reports/lens_distortion_demo/single_view", "experiments/lens_calibration/single_view"),
)


def ps_quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(POWERSHELL and os.name == "nt", "Windows junction support required")
class WorkspaceLayoutMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="workspace-layout-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.workspace = self.base / "EmbeddingTest fixture"
        self.workspace.mkdir()
        (self.workspace / "Main.py").write_text("# migration marker only\n", encoding="utf-8")

    def run_script(self, body: str) -> subprocess.CompletedProcess:
        script = self.base / "invoke.ps1"
        script.write_text("$ErrorActionPreference = 'Stop'\n" + body, encoding="utf-8-sig")
        return subprocess.run(
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            cwd=self.base, capture_output=True, timeout=45,
        )

    def invoke(self, section="All", *, apply=True, prefix="", success=True):
        result = self.run_script(
            prefix + "\n"
            + f"& {ps_quote(MIGRATION)} -WorkspaceRoot {ps_quote(self.workspace)} -Section {ps_quote(section)}"
            + (" -Apply" if apply else "") + "\n"
        )
        output = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
        if success:
            self.assertEqual(result.returncode, 0, output)
        else:
            self.assertNotEqual(result.returncode, 0, output)
        return result

    def populate(self, relative: str, value: bytes = b"original payload"):
        folder = self.workspace / relative
        (folder / "nested").mkdir(parents=True)
        (folder / "nested/payload.bin").write_bytes(value)
        (folder / ".hidden-payload").write_bytes(b"hidden payload")
        return folder

    def assert_compatible(self, old: str, new: str, value: bytes = b"original payload"):
        old_path, new_path = self.workspace / old, self.workspace / new
        self.assertTrue(os.path.samefile(old_path, new_path), f"{old} should reach {new}")
        self.assertEqual((old_path / "nested/payload.bin").read_bytes(), value)
        self.assertEqual((new_path / ".hidden-payload").read_bytes(), b"hidden payload")
        (old_path / "written-through-old-path.txt").write_text("junction write", encoding="utf-8")
        self.assertEqual((new_path / "written-through-old-path.txt").read_text(), "junction write")

    def make_junctions(self, pairs):
        result = self.run_script("\n".join(
            f"New-Item -ItemType Junction -Path {ps_quote(link)} -Target {ps_quote(target)} | Out-Null"
            for link, target in pairs
        ))
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_plan_does_not_move_or_create_directories(self):
        for old, _ in BUILD_MAPPINGS + EXPERIMENT_MAPPINGS:
            self.populate(old)
        self.invoke(apply=False)
        self.assertFalse((self.workspace / "artifacts").exists())
        self.assertFalse((self.workspace / "experiments").exists())
        for old, _ in BUILD_MAPPINGS + EXPERIMENT_MAPPINGS:
            self.assertEqual((self.workspace / old / "nested/payload.bin").read_bytes(), b"original payload")

    def test_all_migration_keeps_old_paths_and_is_idempotent(self):
        for old, _ in BUILD_MAPPINGS + EXPERIMENT_MAPPINGS:
            self.populate(old)
        (self.workspace / "reports/unrelated.txt").write_bytes(b"report remains")
        self.invoke()
        self.assertTrue(os.path.samefile(self.workspace / "reports", self.workspace / "artifacts/reports"))
        for old, new in BUILD_MAPPINGS + EXPERIMENT_MAPPINGS:
            self.assert_compatible(old, new)
        self.invoke()
        self.assertEqual((self.workspace / "artifacts/reports/unrelated.txt").read_bytes(), b"report remains")
        for old, new in BUILD_MAPPINGS + EXPERIMENT_MAPPINGS:
            self.assert_compatible(old, new)

    def test_build_section_does_not_move_experiments_or_reports(self):
        for old, _ in BUILD_MAPPINGS + EXPERIMENT_MAPPINGS:
            self.populate(old)
        self.invoke("Build")
        for old, new in BUILD_MAPPINGS:
            self.assert_compatible(old, new)
        for old, new in EXPERIMENT_MAPPINGS:
            self.assertTrue((self.workspace / old / "nested/payload.bin").is_file())
            self.assertFalse((self.workspace / new).exists())
        self.assertFalse(os.path.samefile(self.workspace / "reports", self.workspace / "artifacts/reports"))

    def test_experiments_section_does_not_move_build_or_reports_root(self):
        for old, _ in BUILD_MAPPINGS + EXPERIMENT_MAPPINGS:
            self.populate(old)
        self.invoke("Experiments")
        for old, new in EXPERIMENT_MAPPINGS:
            self.assert_compatible(old, new)
        for old, new in BUILD_MAPPINGS:
            self.assertTrue((self.workspace / old / "nested/payload.bin").is_file())
            self.assertFalse((self.workspace / new).exists())
        self.assertFalse(os.path.samefile(self.workspace / "reports", self.workspace / "artifacts/reports"))

    def test_populated_destination_is_not_merged_or_replaced(self):
        source = self.populate("dist-build")
        destination = self.populate("artifacts/build/legacy/full-work", b"existing destination")
        self.invoke("Build", success=False)
        self.assertEqual((source / "nested/payload.bin").read_bytes(), b"original payload")
        self.assertEqual((destination / "nested/payload.bin").read_bytes(), b"existing destination")
        self.assertFalse(os.path.samefile(source, destination))

    def test_empty_destination_tree_is_replaced_without_merging(self):
        self.populate("dist-build")
        destination = self.workspace / "artifacts/build/legacy/full-work"
        (destination / "empty/deeper").mkdir(parents=True)
        self.invoke("Build")
        self.assert_compatible("dist-build", "artifacts/build/legacy/full-work")
        self.assertFalse((destination / "empty").exists())

    def test_external_parent_junction_is_rejected_before_writing(self):
        source = self.populate("dist-build")
        outside = self.base / "outside workspace"
        outside.mkdir()
        self.make_junctions([(self.workspace / "artifacts", outside)])
        self.invoke("Build", success=False)
        self.assertEqual((source / "nested/payload.bin").read_bytes(), b"original payload")
        self.assertEqual(list(outside.iterdir()), [])

    def test_chained_parent_junction_cannot_escape_workspace(self):
        source = self.populate("dist-build")
        outside = self.base / "outside workspace"
        outside.mkdir()
        bridge = self.workspace / "bridge"
        self.make_junctions([(bridge, outside), (self.workspace / "artifacts", bridge)])
        self.invoke("Build", success=False)
        self.assertEqual((source / "nested/payload.bin").read_bytes(), b"original payload")
        self.assertEqual(list(outside.iterdir()), [])

    def test_unexpected_existing_source_link_is_not_replaced(self):
        other = self.populate("other-build")
        self.make_junctions([(self.workspace / "dist-build", other)])
        self.invoke("Build", success=False)
        self.assertEqual((other / "nested/payload.bin").read_bytes(), b"original payload")
        self.assertTrue(os.path.samefile(self.workspace / "dist-build", other))
        self.assertFalse((self.workspace / "artifacts").exists())

    def test_failed_junction_creation_restores_entire_source_directory(self):
        source = self.populate("dist-build")
        prefix = r"""
function New-Item {
    param([string]$ItemType, [string]$Path, [string]$Target, [switch]$Force)
    if ($ItemType -eq 'Junction') { throw 'fixture junction creation failure' }
    Microsoft.PowerShell.Management\New-Item @PSBoundParameters
}
"""
        self.invoke("Build", prefix=prefix, success=False)
        self.assertEqual((source / "nested/payload.bin").read_bytes(), b"original payload")
        self.assertEqual((source / ".hidden-payload").read_bytes(), b"hidden payload")
        self.assertFalse((self.workspace / "artifacts/build/legacy/full-work").exists())

    def test_directory_rename_with_locked_child_never_leaves_partial_tree(self):
        source = self.populate("dist-build")
        locked = source / "nested/payload.bin"
        result = self.run_script(
            f"$locked = [System.IO.File]::Open({ps_quote(locked)}, 'Open', 'Read', 'Read')\n"
            "try {\n"
            + f"& {ps_quote(MIGRATION)} -WorkspaceRoot {ps_quote(self.workspace)} -Section Build -Apply\n"
            + "} finally { $locked.Dispose() }\n"
        )
        destination = self.workspace / "artifacts/build/legacy/full-work"
        if result.returncode == 0:
            self.assert_compatible("dist-build", "artifacts/build/legacy/full-work")
        else:
            self.assertFalse(destination.exists())
            self.assertEqual((source / "nested/payload.bin").read_bytes(), b"original payload")
            self.assertEqual((source / ".hidden-payload").read_bytes(), b"hidden payload")

    def test_workspace_marker_and_section_are_required(self):
        self.populate("dist-build")
        self.invoke("Unknown", success=False)
        (self.workspace / "Main.py").unlink()
        self.invoke("Build", success=False)
        self.assertFalse((self.workspace / "artifacts").exists())


class MigrationImplementationTests(unittest.TestCase):
    def test_rename_uses_atomic_directory_move_instead_of_recursive_move_item(self):
        source = MIGRATION.read_text(encoding="utf-8-sig")
        self.assertIn("[System.IO.Directory]::Move($source, $destination)", source)
        self.assertNotRegex(source, re.compile(r"^\s*Move-Item\b", re.MULTILINE | re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
