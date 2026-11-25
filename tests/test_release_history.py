from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

from tools.generate_version_info import parse_version, version_resource_text
from tools.record_release import record_release


ROOT = Path(__file__).resolve().parents[1]


class ReleaseHistoryTests(unittest.TestCase):
    def make_package(self, root, name='package', build_id='20261122T100000000001Z'):
        release = root / name
        metadata = release / 'EmbeddingTest'
        metadata.mkdir(parents=True)
        (release / 'LC System.exe').write_bytes(b'packaged executable')
        (metadata / 'VERSION').write_text('3.0.7\n', encoding='utf-8')
        (metadata / 'build_manifest.json').write_text(json.dumps({
            'version': '3.0.7', 'build_id': build_id, 'product_name': 'LC System',
            'built_at_utc': '2026-11-22T10:00:00+00:00',
            'source': {'commit': 'a' * 40, 'dirty': True},
            'release_notes': {'changes': ['自动保存版本记录']},
        }), encoding='utf-8')
        return release

    def test_history_survives_same_version_rebuild_and_records_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.make_package(root)
            record = record_release(root, first, 'LC System.exe')
            self.assertEqual(record['executable']['sha256'], hashlib.sha256(b'packaged executable').hexdigest())
            self.assertIn('自动保存版本记录', (first / 'VERSION_HISTORY.txt').read_text(encoding='utf-8-sig'))
            self.assertEqual(record_release(root, first, 'LC System.exe'), record)
            second = self.make_package(root, 'package-rebuild', '20261122T100000000002Z')
            record_release(root, second, 'LC System.exe')
            history = json.loads((root / 'releases/history.json').read_text(encoding='utf-8'))
            self.assertEqual(len(history), 2)
            self.assertEqual(history[1], record)
            self.assertEqual(len(list((root / 'releases/builds/3.0.7').glob('*.json'))), 2)

    def test_invalid_or_mismatched_packages_do_not_create_success_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = self.make_package(root)
            for exe in ('absent.exe', '../outside.exe'):
                with self.assertRaises(ValueError):
                    record_release(root, release, exe)
            (release / 'EmbeddingTest/VERSION').write_text('3.0.6', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'do not match'):
                record_release(root, release, 'LC System.exe')
            self.assertFalse((root / 'releases').exists())

    def test_exe_version_resource_uses_requested_numeric_version(self):
        version, numeric = parse_version('3.0.7')
        self.assertEqual(numeric, (3, 0, 7, 0))
        resource = version_resource_text(version, numeric, product_name='LC System', exe_name='LC System.exe')
        self.assertIn('filevers=(3, 0, 7, 0)', resource)
        self.assertIn("StringStruct('ProductVersion', '3.0.7')", resource)

    @unittest.skipUnless(shutil.which('powershell.exe'), 'Windows PowerShell required')
    def test_build_pipeline_records_rebuilds_and_includes_history_in_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tools').mkdir()
            for name in ('generate_version_info.py', 'record_release.py'):
                shutil.copy2(ROOT / 'tools' / name, root / 'tools' / name)
            (root / 'VERSION').write_text('3.0.7', encoding='utf-8')
            def ps(path):
                return "'" + str(path).replace("'", "''") + "'"
            script = root / 'publish-test.ps1'
            script.write_text(f"""$ErrorActionPreference = 'Stop'
. {ps(ROOT / 'tools/package_release.ps1')}
$testRoot = {ps(root)}
$python = {ps(sys.executable)}
$dist = Join-Path $testRoot 'dist'
foreach ($run in 1..2) {{
    $stage = Join-Path $dist 'LC System'
    $metadataDir = Join-Path $stage 'EmbeddingTest'
    $metadata = New-LCBuildMetadata -ScriptRoot $testRoot -MetadataPath $metadataDir -PythonExe $python -Version '3.0.7' -ProductName 'LC System' -ExeName 'LC System.exe'
    [System.IO.File]::WriteAllText((Join-Path $stage 'LC System.exe'), "executable $run")
    $release = Publish-LCReleaseFolder -DistPath $dist -BuildFolderName 'LC System' -ReleaseFolderName 'LC System V3.0.7' -ArchiveFileName 'LC_System_V3.0.7.zip' -ScriptRoot $testRoot -PythonExe $python -ExeName 'LC System.exe'
}}
""", encoding='utf-8-sig')
            result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout.decode(errors='replace') + result.stderr.decode(errors='replace'))
            archives = list((root / 'dist').glob('*.zip'))
            self.assertEqual(len(archives), 2)
            self.assertEqual((root / 'dist/LC System V3.0.7/LC System.exe').read_text(), 'executable 1')
            for archive in archives:
                with zipfile.ZipFile(archive) as zipped:
                    self.assertTrue(any(name.endswith('/VERSION_HISTORY.txt') for name in zipped.namelist()))
                    self.assertTrue(any(name.endswith('/release_record.json') for name in zipped.namelist()))
            self.assertEqual(len(json.loads((root / 'releases/history.json').read_text(encoding='utf-8'))), 2)


if __name__ == '__main__':
    unittest.main()
