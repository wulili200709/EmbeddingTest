import json
import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets

from infrastructure.audit_store import AuditStore
from ui.shell.audit_dialogs import SoftwareVersionDialog


ROOT = Path(__file__).resolve().parents[1]


class SoftwareVersionUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_project_notes_have_a_unique_id_and_consistent_format(self):
        for path in (ROOT / "releases/notes").glob("*.json"):
            notes = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notes["version"], path.stem)
            seen = set()
            for change in notes["changes"]:
                with self.subTest(version=path.stem, change=change):
                    self.assertTrue(change["id"].strip())
                    self.assertNotIn(change["id"], seen)
                    seen.add(change["id"])
                    self.assertIn(change["type"], {"新增", "优化", "修复"})
                    self.assertTrue(change["summary"].strip())
                    self.assertTrue(change["details"].strip())
                    datetime.strptime(change["time"], "%Y-%m-%d %H:%M:%S")

    def test_existing_database_and_read_only_dialog_receive_each_change_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notes_dir = root / "releases/notes"
            notes_dir.mkdir(parents=True)
            path = notes_dir / "3.0.7.json"
            notes = {"version": path.stem, "changes": []}

            def add_change(key):
                notes["changes"].append({"id": key, "time": "2026-11-27 14:00:00", "type": "修复",
                                         "summary": key, "details": f"已修复{key}。"})
                path.write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")

            db = root / "audit.db"
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute("""CREATE TABLE software_versions(id INTEGER PRIMARY KEY, created_at TEXT,
                             version TEXT, summary TEXT, details TEXT, operator TEXT, software_version TEXT)""")
                conn.execute("INSERT INTO software_versions VALUES(1, '2026-07-09', 'V2.2.0', '原记录', '', 'admin', '')")
            add_change("触发按钮")
            store = AuditStore(db)
            with ThreadPoolExecutor(max_workers=2) as executor:
                counts = list(executor.map(lambda _: store.sync_software_updates(notes_dir), range(2)))
            self.assertEqual(sorted(counts), [0, 1])
            add_change("语言切换")
            with patch("infrastructure.audit_store.packaged_embedding_test_root", return_value=root):
                dialog = SoftwareVersionDialog(None, store, can_edit=False, current_user="operator", software_version="3.0.7")
                dialog._reload()
                self.assertEqual(dialog.table.rowCount(), 3)
                row = [dialog.table.item(0, column).text() for column in range(6)]
                self.assertEqual(row, ["2026-11-27 14:00:00", "V3.0.7", "【修复】语言切换",
                                       "已修复语言切换。", "系统", "3.0.7"])
                self.assertEqual(dialog.table.item(2, 2).text(), "原记录")
                dialog.close()
            self.assertEqual(AuditStore(db).sync_software_updates(notes_dir), 0)
            store.add_software_version(version="V3.0.7", summary="手工补充", operator="admin")
            original = next(row for row in store.software_versions() if row["summary"] == "【修复】触发按钮")
            notes["changes"][0].update(summary="相机触发按钮", details="修订后的说明。")
            path.write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(store.sync_software_updates(notes_dir), 1)
            self.assertEqual(store.sync_software_updates(notes_dir), 0)
            self.assertIn(dict(original, summary="【修复】相机触发按钮", details="修订后的说明。"), store.software_versions())
            self.assertEqual(len(store.software_versions()), 4)

    def test_upgrade_assigns_only_post_release_notes_to_3_0_8(self):
        notes_dir = ROOT / "releases/notes"
        cutoff = "2026-11-22 17:22:00"
        published = json.loads((notes_dir / "3.0.7.json").read_text(encoding="utf-8"))["changes"]
        current = json.loads((notes_dir / "3.0.8.json").read_text(encoding="utf-8"))["changes"]
        self.assertTrue(all(change["time"] <= cutoff for change in published))
        self.assertTrue(all(change["time"] > cutoff for change in current))
        moved = next(change for change in current if change["id"] == "shape-localization-groups")
        with tempfile.TemporaryDirectory() as directory, patch.object(AuditStore, "_ensure_defaults"):
            db = Path(directory) / "audit.db"
            store = AuditStore(db)
            with closing(store.connect()) as conn, conn:
                conn.execute("PRAGMA user_version=0")
                conn.executemany(
                    """INSERT INTO software_versions(id, created_at, version, summary, operator, software_version, source_key)
                       VALUES (?, ?, ?, '原记录', 'operator', ?, ?)""",
                    [(1, cutoff, "V3.0.7", "3.0.7", None),
                     (2, "2026-11-22 17:22:01", "3.0.7", "dev", None),
                     (3, moved["time"], "V3.0.7", "3.0.7", "3.0.7/" + moved["id"]),
                     (4, moved["time"], "V3.0.8", "3.0.8", "3.0.8/" + moved["id"])],
                )
            store = AuditStore(db)
            store.sync_software_updates(notes_dir)
            self.assertEqual(AuditStore(db).sync_software_updates(notes_dir), 0)
            with closing(store.connect()) as conn:
                result = conn.execute(
                    "SELECT id, created_at, version, operator, software_version, source_key FROM software_versions WHERE id<=4 ORDER BY id"
                ).fetchall()
                self.assertEqual([tuple(row) for row in result], [
                    (1, cutoff, "V3.0.7", "operator", "3.0.7", None),
                    (2, "2026-11-22 17:22:01", "3.0.8", "operator", "dev", None),
                    (3, moved["time"], "V3.0.8", "operator", "3.0.8", "3.0.8/" + moved["id"]),
                ])

    def test_original_manual_records_are_reformatted_in_place(self):
        notes_dir = ROOT / "releases/notes"
        with tempfile.TemporaryDirectory() as directory, patch.object(AuditStore, "_ensure_defaults"):
            store = AuditStore(Path(directory) / "audit.db")
            originals = []
            with closing(store.connect()) as conn, conn:
                for version in ("2.2.0", "2.2.1"):
                    changes = json.loads((notes_dir / f"{version}.json").read_text(encoding="utf-8"))["changes"]
                    entry = next(change for change in changes if "previous_record" in change)
                    old = entry["previous_record"]
                    cursor = conn.execute(
                        """INSERT INTO software_versions(created_at, version, summary, details, operator, software_version)
                           VALUES (?, ?, ?, ?, 'admin', 'dev')""",
                        (entry["time"], f"V{version}", old["summary"], old["details"]),
                    )
                    originals.append((cursor.lastrowid, version, entry))
                # A station's edited copy must remain separate from the shipped notes.
                conn.execute("""INSERT INTO software_versions(created_at, version, summary, details)
                             SELECT created_at, version, summary, '本地备注' FROM software_versions WHERE id=?""",
                             (originals[0][0],))
            store.sync_software_updates(notes_dir)
            self.assertEqual(store.sync_software_updates(notes_dir), 0)
            with closing(store.connect()) as conn:
                for row_id, version, entry in originals:
                    row = conn.execute("SELECT * FROM software_versions WHERE id=?", (row_id,)).fetchone()
                    self.assertEqual(row["source_key"], f"{version}/{entry['id']}")
                    self.assertEqual(row["summary"], f"【{entry['type']}】{entry['summary']}")
                    self.assertEqual(row["details"], entry["details"])
                    self.assertEqual((row["created_at"], row["operator"], row["software_version"]),
                                     (entry["time"], "admin", "dev"))
                self.assertEqual(conn.execute(
                    "SELECT details FROM software_versions WHERE source_key IS NULL"
                ).fetchall()[0][0], "本地备注")
