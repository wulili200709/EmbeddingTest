"""Record every successful portable executable build and include its history."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_version_info import parse_version


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record_release(project_root: Path, release_dir: Path, exe_name: str) -> dict:
    project_root, release_dir = project_root.resolve(), release_dir.resolve()
    exe_path = (release_dir / exe_name).resolve()
    if exe_path.parent != release_dir or not exe_path.is_file() or exe_path.stat().st_size == 0:
        raise ValueError(f'Missing or invalid release executable: {exe_path}')
    manifest_path = release_dir / 'EmbeddingTest' / 'build_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    version, _ = parse_version(manifest['version'])
    version_path = manifest_path.with_name('VERSION')
    if version_path.read_text(encoding='utf-8').strip() != version:
        raise ValueError('Packaged VERSION and build manifest do not match')
    build_id = manifest['build_id']
    if not build_id or any(c not in '0123456789TZ' for c in build_id):
        raise ValueError('Invalid build ID')
    edition = 'lite' if 'lite' in manifest['product_name'].lower() else 'full'
    record_id = f'{build_id}_{edition}'
    record = {
        'schema_version': 1, 'record_id': record_id, 'status': 'built', **manifest,
        'recorded_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'release_folder': str(release_dir),
        'executable': {'name': exe_name, 'size_bytes': exe_path.stat().st_size,
                       'sha256': sha256(exe_path)},
    }
    history_dir = project_root / 'releases' / 'builds'
    record_path = history_dir / version / f'{record_id}.json'
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding='utf-8'))
        if previous['executable'] != record['executable'] or previous['release_folder'] != record['release_folder']:
            raise ValueError('Build ID already belongs to a different executable; create new build metadata')
        record = previous
    else:
        write_json(record_path, record)
    history = [json.loads(path.read_text(encoding='utf-8'))
               for path in sorted(history_dir.glob('*/*.json'))]
    history.sort(key=lambda row: row['record_id'], reverse=True)
    write_json(project_root / 'releases' / 'history.json', history)
    write_json(release_dir / 'release_record.json', record)
    lines = ['LC System - 自动版本记录', '']
    for row in history:
        lines.extend([f"{row['product_name']} V{row['version']}  {row['recorded_at']}",
                      f"构建编号: {row['record_id']}",
                      f"代码提交: {row.get('source', {}).get('commit', 'unavailable')}"])
        notes = row.get('release_notes', {}).get('changes', [])
        if notes:
            lines.extend(f'  - {note}' for note in notes)
        else:
            lines.extend(f"  - {item.get('subject', '')}"
                         for item in row.get('source', {}).get('changes_since_previous_release', []))
        changes = row.get('source', {}).get('working_tree_changes', [])
        if changes:
            lines.append('构建时未提交的文件改动:')
            lines.extend(f'  {change}' for change in changes)
        lines.extend(f'说明: {note}' for note in row.get('release_notes', {}).get('validation_notes', []))
        lines.extend([f"EXE SHA-256: {row['executable']['sha256']}", ''])
    text = '\n'.join(lines)
    (project_root / 'releases' / 'VERSION_HISTORY.txt').write_text(text, encoding='utf-8-sig')
    (release_dir / 'VERSION_HISTORY.txt').write_text(text, encoding='utf-8-sig')
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--exe-name', required=True)
    args = parser.parse_args()
    record = record_release(args.project_root, args.release_dir, args.exe_name)
    print(f"Recorded {record['product_name']} V{record['version']}: {record['record_id']}")


if __name__ == '__main__':
    main()
