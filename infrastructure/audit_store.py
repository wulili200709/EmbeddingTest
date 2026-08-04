from __future__ import annotations

import csv
import hashlib
import hmac
import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

from common.app_paths import writable_embedding_test_root


AUDIT_DB_FILENAME = "audit.db"
DEFAULT_OPERATOR_USER = "operator"
DEFAULT_SUPER_ADMIN_USER = "admin"
DEFAULT_SUPER_ADMIN_PASSWORD = "admin654321"
DEFAULT_ADMIN_PASSWORD = "123456"
DEFAULT_BOOTSTRAP_ADMIN_USERS = (
    "YQCXN7",
    "DE7TPA",
    "BJA3BS",
)
PASSWORD_ITERATIONS = 200_000


@dataclass(frozen=True)
class PermissionDef:
    key: str
    module: str
    name: str


PERMISSIONS: tuple[PermissionDef, ...] = (
    PermissionDef("product.select", "产品", "选择产品"),
    PermissionDef("product.create", "产品", "新增产品"),
    PermissionDef("product.delete", "产品", "删除产品"),
    PermissionDef("runtime.run", "运行", "运行检测"),
    PermissionDef("runtime.connect_camera", "运行", "连接/断开相机"),
    PermissionDef("runtime.release_ng", "运行", "NG放行"),
    PermissionDef("runtime_records.view", "运行记录", "查看运行记录"),
    PermissionDef("runtime_records.export", "运行记录", "导出运行记录"),
    PermissionDef("io.debug", "硬件", "IO调试"),
    PermissionDef("camera.edit_params", "相机", "修改相机参数"),
    PermissionDef("camera.open_mvs", "相机", "打开 MVS（查看序列号）"),
    PermissionDef("template.edit_roi", "模板", "修改模板/ROI"),
    PermissionDef("template.edit_params", "模板", "修改模板参数"),
    PermissionDef("sample.manage", "样本", "维护训练/测试样本"),
    PermissionDef("inspection.edit_items", "检测项", "修改检测项"),
    PermissionDef("inspection.edit_limits", "检测项", "修改上下限"),
    PermissionDef("model.train", "模型", "重新训练"),
    PermissionDef("settings.tower_light", "设置", "三色灯设置"),
    PermissionDef("settings.record_path", "设置", "保存路径设置"),
    PermissionDef("settings.passwords", "设置", "密码设置"),
    PermissionDef("audit.view", "履历", "查看履历"),
    PermissionDef("audit.export", "履历", "导出履历"),
    PermissionDef("user.manage", "用户", "用户/权限管理"),
    PermissionDef("software.version_log", "软件", "软件版本履历维护"),
)

PERMISSION_LABELS = {item.key: item.name for item in PERMISSIONS}
PERMISSION_MODULES = {item.key: item.module for item in PERMISSIONS}

DEFAULT_ROLE_PERMISSIONS: dict[str, set[str]] = {
    "operator": {
        "product.select",
        "runtime.run",
    },
    "admin": {
        "product.select",
        "product.create",
        "product.delete",
        "runtime.run",
        "runtime.connect_camera",
        "runtime.release_ng",
        "runtime_records.view",
        "runtime_records.export",
        "io.debug",
        "camera.edit_params",
        "camera.open_mvs",
        "template.edit_roi",
        "template.edit_params",
        "sample.manage",
        "inspection.edit_items",
        "inspection.edit_limits",
        "model.train",
        "settings.tower_light",
        "settings.record_path",
        "settings.passwords",
        "audit.view",
        "audit.export",
        "software.version_log",
    },
    "super_admin": {item.key for item in PERMISSIONS},
}

DEFAULT_ROLE_NAMES = {
    "operator": "操作员",
    "admin": "管理员",
    "super_admin": "超级管理员",
}


@dataclass
class CurrentUser:
    user_name: str = DEFAULT_OPERATOR_USER
    role_key: str = "operator"
    role_name: str = "操作员"
    is_super_admin: bool = False
    must_change_password: bool = False


def default_audit_db_path() -> Path:
    return writable_embedding_test_root(__file__) / "records" / AUDIT_DB_FILENAME


def utcnow_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _initialize_runtime_record_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runtime_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_time TEXT NOT NULL,
            product_name TEXT NOT NULL,
            recipe_name TEXT NOT NULL DEFAULT '',
            final_result TEXT NOT NULL DEFAULT '',
            duration_ms INTEGER NOT NULL DEFAULT 0,
            error_message TEXT NOT NULL DEFAULT '',
            is_system_error INTEGER NOT NULL DEFAULT 0,
            image_paths_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_runs_time ON runtime_runs(record_time);
        CREATE INDEX IF NOT EXISTS idx_runtime_runs_product_time ON runtime_runs(product_name, record_time);
        CREATE INDEX IF NOT EXISTS idx_runtime_runs_result ON runtime_runs(final_result);
        CREATE TABLE IF NOT EXISTS runtime_roi_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            camera_id TEXT NOT NULL,
            item_id TEXT NOT NULL DEFAULT '',
            roi_label TEXT NOT NULL DEFAULT '',
            display_name TEXT NOT NULL DEFAULT '',
            result TEXT NOT NULL DEFAULT '',
            value REAL,
            unit TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            detail TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(run_id) REFERENCES runtime_runs(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_roi_run ON runtime_roi_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_runtime_roi_camera_result ON runtime_roi_results(camera_id, result);
        CREATE INDEX IF NOT EXISTS idx_runtime_roi_label ON runtime_roi_results(roi_label);
        """
    )


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        str(password or "").encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return salt.hex(), digest.hex()


def _verify_password(password: str, salt_hex: str, digest_hex: str) -> bool:
    _salt, candidate = _hash_password(password, salt_hex)
    return hmac.compare_digest(candidate, str(digest_hex or ""))


class AuditStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_audit_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS roles (
                    role_key TEXT PRIMARY KEY,
                    role_name TEXT NOT NULL,
                    editable INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS role_permissions (
                    role_key TEXT NOT NULL,
                    permission_key TEXT NOT NULL,
                    PRIMARY KEY(role_key, permission_key),
                    FOREIGN KEY(role_key) REFERENCES roles(role_key)
                );
                CREATE TABLE IF NOT EXISTS permission_migrations (
                    migration_key TEXT PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    role_key TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    is_super_admin INTEGER NOT NULL DEFAULT 0,
                    must_change_password INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT,
                    FOREIGN KEY(role_key) REFERENCES roles(role_key)
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    user_name TEXT NOT NULL,
                    role_name TEXT NOT NULL,
                    product_name TEXT,
                    module TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT,
                    before_value TEXT,
                    after_value TEXT,
                    result TEXT NOT NULL DEFAULT '成功',
                    remark TEXT,
                    software_version TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_events(created_at);
                CREATE INDEX IF NOT EXISTS idx_audit_product ON audit_events(product_name);
                CREATE INDEX IF NOT EXISTS idx_audit_module ON audit_events(module);
                CREATE TABLE IF NOT EXISTS software_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    version TEXT NOT NULL,
                    summary TEXT,
                    details TEXT,
                    operator TEXT,
                    software_version TEXT
                );
                """
            )
            _initialize_runtime_record_schema(conn)
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        with self.connect() as conn:
            for role_key, role_name in DEFAULT_ROLE_NAMES.items():
                editable = 0 if role_key == "super_admin" else 1
                conn.execute(
                    """
                    INSERT INTO roles(role_key, role_name, editable)
                    VALUES(?, ?, ?)
                    ON CONFLICT(role_key) DO UPDATE SET role_name=excluded.role_name
                    """,
                    (role_key, role_name, editable),
                )
                existing = conn.execute(
                    "SELECT COUNT(*) FROM role_permissions WHERE role_key=?",
                    (role_key,),
                ).fetchone()[0]
                if existing == 0:
                    self.set_role_permissions(role_key, DEFAULT_ROLE_PERMISSIONS[role_key], conn=conn)

            migration_key = "add_camera_open_mvs_permission"
            migration_applied = conn.execute(
                "SELECT 1 FROM permission_migrations WHERE migration_key=?",
                (migration_key,),
            ).fetchone()
            if migration_applied is None:
                conn.execute(
                    "INSERT OR IGNORE INTO role_permissions(role_key, permission_key) VALUES(?, ?)",
                    ("admin", "camera.open_mvs"),
                )
                conn.execute(
                    "INSERT INTO permission_migrations(migration_key) VALUES(?)",
                    (migration_key,),
                )

            migration_key = "add_runtime_records_view_permission"
            migration_applied = conn.execute(
                "SELECT 1 FROM permission_migrations WHERE migration_key=?",
                (migration_key,),
            ).fetchone()
            if migration_applied is None:
                conn.execute(
                    "INSERT OR IGNORE INTO role_permissions(role_key, permission_key) VALUES(?, ?)",
                    ("admin", "runtime_records.view"),
                )
                conn.execute(
                    "INSERT INTO permission_migrations(migration_key) VALUES(?)",
                    (migration_key,),
                )

            migration_key = "add_runtime_records_export_permission"
            migration_applied = conn.execute(
                "SELECT 1 FROM permission_migrations WHERE migration_key=?",
                (migration_key,),
            ).fetchone()
            if migration_applied is None:
                conn.execute(
                    "INSERT OR IGNORE INTO role_permissions(role_key, permission_key) VALUES(?, ?)",
                    ("admin", "runtime_records.export"),
                )
                conn.execute(
                    "INSERT INTO permission_migrations(migration_key) VALUES(?)",
                    (migration_key,),
                )

            row = conn.execute(
                "SELECT id FROM users WHERE user_name=?",
                (DEFAULT_SUPER_ADMIN_USER,),
            ).fetchone()
            if row is None:
                salt, digest = _hash_password(DEFAULT_SUPER_ADMIN_PASSWORD)
                conn.execute(
                    """
                    INSERT INTO users(
                        user_name, password_hash, password_salt, role_key,
                        enabled, is_super_admin, must_change_password, created_at
                    )
                    VALUES(?, ?, ?, 'super_admin', 1, 1, 0, ?)
                    """,
                    (DEFAULT_SUPER_ADMIN_USER, digest, salt, utcnow_text()),
                )
            conn.execute(
                "UPDATE users SET must_change_password=0 WHERE user_name=?",
                (DEFAULT_SUPER_ADMIN_USER,),
            )

            migration_key = "add_bootstrap_admin_users_20260728"
            migration_applied = conn.execute(
                "SELECT 1 FROM permission_migrations WHERE migration_key=?",
                (migration_key,),
            ).fetchone()
            if migration_applied is None:
                for user_name in DEFAULT_BOOTSTRAP_ADMIN_USERS:
                    salt, digest = _hash_password(DEFAULT_ADMIN_PASSWORD)
                    conn.execute(
                        """
                        INSERT INTO users(
                            user_name, password_hash, password_salt, role_key,
                            enabled, is_super_admin, must_change_password, created_at
                        )
                        VALUES(?, ?, ?, 'admin', 1, 0, 0, ?)
                        ON CONFLICT(user_name) DO NOTHING
                        """,
                        (user_name, digest, salt, utcnow_text()),
                    )
                conn.execute(
                    "INSERT INTO permission_migrations(migration_key) VALUES(?)",
                    (migration_key,),
                )

    def roles(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT role_key, role_name, editable FROM roles ORDER BY editable, role_key"
            ).fetchall()
            return [dict(row) for row in rows]

    def role_permissions(self, role_key: str) -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT permission_key FROM role_permissions WHERE role_key=?",
                (role_key,),
            ).fetchall()
            return {str(row["permission_key"]) for row in rows}

    def set_role_permissions(
        self,
        role_key: str,
        permissions: Iterable[str],
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        def _write(target_conn: sqlite3.Connection) -> None:
            perms = {str(item).strip() for item in permissions if str(item).strip()}
            if role_key == "super_admin":
                perms = {item.key for item in PERMISSIONS}
            target_conn.execute("DELETE FROM role_permissions WHERE role_key=?", (role_key,))
            target_conn.executemany(
                "INSERT OR IGNORE INTO role_permissions(role_key, permission_key) VALUES(?, ?)",
                [(role_key, permission) for permission in sorted(perms)],
            )

        if conn is not None:
            _write(conn)
            return
        with self.connect() as own_conn:
            _write(own_conn)

    def users(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT u.id, u.user_name, u.role_key, r.role_name, u.enabled,
                       u.is_super_admin, u.must_change_password, u.created_at, u.last_login_at
                FROM users u
                LEFT JOIN roles r ON r.role_key = u.role_key
                ORDER BY u.user_name
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def authenticate(self, user_name: str, password: str) -> CurrentUser | None:
        name = str(user_name or "").strip()
        if not name:
            return None
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT u.*, r.role_name
                FROM users u
                LEFT JOIN roles r ON r.role_key = u.role_key
                WHERE u.user_name=?
                """,
                (name,),
            ).fetchone()
            if row is None or not int(row["enabled"] or 0):
                return None
            if not _verify_password(password, row["password_salt"], row["password_hash"]):
                return None
            conn.execute(
                "UPDATE users SET last_login_at=? WHERE id=?",
                (utcnow_text(), int(row["id"])),
            )
            return CurrentUser(
                user_name=str(row["user_name"]),
                role_key=str(row["role_key"]),
                role_name=str(row["role_name"] or row["role_key"]),
                is_super_admin=bool(row["is_super_admin"]),
                must_change_password=bool(row["must_change_password"]),
            )

    def create_user(
        self,
        user_name: str,
        password: str,
        role_key: str,
        *,
        enabled: bool = True,
        must_change_password: bool = False,
    ) -> None:
        name = str(user_name or "").strip()
        if not name:
            raise ValueError("用户名不能为空")
        if len(str(password or "")) < 4:
            raise ValueError("密码至少需要 4 位")
        salt, digest = _hash_password(password)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO users(
                    user_name, password_hash, password_salt, role_key,
                    enabled, is_super_admin, must_change_password, created_at
                )
                VALUES(?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    name,
                    digest,
                    salt,
                    str(role_key or "admin").strip() or "admin",
                    1 if enabled else 0,
                    1 if must_change_password else 0,
                    utcnow_text(),
                ),
            )

    def update_user(
        self,
        user_id: int,
        *,
        role_key: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        fields: list[str] = []
        params: list[object] = []
        if role_key is not None:
            fields.append("role_key=?")
            params.append(str(role_key).strip() or "admin")
        if enabled is not None:
            fields.append("enabled=?")
            params.append(1 if enabled else 0)
        if not fields:
            return
        params.append(int(user_id))
        with self.connect() as conn:
            row = conn.execute(
                "SELECT is_super_admin FROM users WHERE id=?",
                (int(user_id),),
            ).fetchone()
            if row is not None and bool(row["is_super_admin"]):
                kept_fields: list[str] = []
                kept_params: list[object] = []
                for field, param in zip(fields, params):
                    if not field.startswith("role_key"):
                        kept_fields.append(field)
                        kept_params.append(param)
                fields = kept_fields
                params = kept_params
                if not fields:
                    return
                params.append(int(user_id))
            conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE id=?", params)

    def set_user_password(
        self,
        user_id: int,
        password: str,
        *,
        must_change_password: bool = False,
    ) -> None:
        if len(str(password or "")) < 4:
            raise ValueError("密码至少需要 4 位")
        salt, digest = _hash_password(password)
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET password_hash=?, password_salt=?, must_change_password=?
                WHERE id=?
                """,
                (digest, salt, 1 if must_change_password else 0, int(user_id)),
            )

    def set_user_password_by_name(
        self,
        user_name: str,
        password: str,
        *,
        must_change_password: bool = False,
    ) -> None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM users WHERE user_name=?",
                (str(user_name or "").strip(),),
            ).fetchone()
        if row is None:
            raise ValueError("用户不存在")
        self.set_user_password(
            int(row["id"]),
            password,
            must_change_password=must_change_password,
        )

    def permissions_for_user(self, user: CurrentUser) -> set[str]:
        if user.is_super_admin or user.role_key == "super_admin":
            return {item.key for item in PERMISSIONS}
        return self.role_permissions(user.role_key)

    def log_event(
        self,
        *,
        user_name: str,
        role_name: str,
        product_name: str = "",
        module: str,
        action: str,
        target: str = "",
        before_value: str = "",
        after_value: str = "",
        result: str = "成功",
        remark: str = "",
        software_version: str = "",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events(
                    created_at, user_name, role_name, product_name, module, action,
                    target, before_value, after_value, result, remark, software_version
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utcnow_text(),
                    str(user_name or DEFAULT_OPERATOR_USER),
                    str(role_name or "操作员"),
                    str(product_name or ""),
                    str(module or ""),
                    str(action or ""),
                    str(target or ""),
                    str(before_value or ""),
                    str(after_value or ""),
                    str(result or "成功"),
                    str(remark or ""),
                    str(software_version or ""),
                ),
            )

    def query_events(
        self,
        *,
        start_at: str = "",
        end_at: str = "",
        product_name: str = "",
        module: str = "",
        user_name: str = "",
        limit: int = 1000,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[object] = []
        if start_at:
            clauses.append("created_at >= ?")
            params.append(start_at)
        if end_at:
            clauses.append("created_at <= ?")
            params.append(end_at)
        if product_name:
            clauses.append("product_name LIKE ?")
            params.append(f"%{product_name}%")
        if module:
            clauses.append("module LIKE ?")
            params.append(f"%{module}%")
        if user_name:
            clauses.append("user_name LIKE ?")
            params.append(f"%{user_name}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT created_at, user_name, role_name, product_name, module, action,
                       target, before_value, after_value, result, remark, software_version
                FROM audit_events
                {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def audit_filter_values(self, column: str, *, limit: int = 200) -> list[str]:
        allowed_columns = {"product_name", "module", "user_name"}
        if column not in allowed_columns:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT {column} AS value
                FROM audit_events
                WHERE COALESCE({column}, '') <> ''
                ORDER BY value
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [str(row["value"]) for row in rows if str(row["value"] or "").strip()]

    def export_events_csv(self, rows: Iterable[Mapping[str, object]], path: str | Path) -> None:
        fieldnames = [
            "created_at",
            "user_name",
            "role_name",
            "product_name",
            "module",
            "action",
            "target",
            "before_value",
            "after_value",
            "result",
            "remark",
            "software_version",
        ]
        with Path(path).open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def add_software_version(
        self,
        *,
        version: str,
        summary: str,
        details: str = "",
        operator: str = "",
        software_version: str = "",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO software_versions(
                    created_at, version, summary, details, operator, software_version
                )
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    utcnow_text(),
                    str(version or ""),
                    str(summary or ""),
                    str(details or ""),
                    str(operator or ""),
                    str(software_version or ""),
                ),
            )

    def software_versions(self, *, limit: int = 500) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT created_at, version, summary, details, operator, software_version
                FROM software_versions
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [dict(row) for row in rows]


class RuntimeRecordStore:
    """Persist runtime inspection summaries and per-ROI results in audit.db."""

    RUN_COLUMNS = [
        "record_time",
        "product_name",
        "final_result",
    ]

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_audit_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            _initialize_runtime_record_schema(conn)

    def write_runtime_result(self, runtime_result: object) -> int:
        camera_results = dict(getattr(runtime_result, "camera_results", {}) or {})
        image_paths = {
            str(camera_id or "").strip(): str(getattr(camera_result, "image_path", "") or "").strip()
            for camera_id, camera_result in camera_results.items()
            if str(camera_id or "").strip() and str(getattr(camera_result, "image_path", "") or "").strip()
        }
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runtime_runs(
                    record_time, product_name, recipe_name, final_result, duration_ms,
                    error_message, is_system_error, image_paths_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utcnow_text(),
                    str(getattr(runtime_result, "product_name", "") or "").strip(),
                    str(getattr(runtime_result, "recipe_name", "") or "").strip(),
                    str(getattr(runtime_result, "final_result", "") or "").strip(),
                    int(getattr(runtime_result, "duration_ms", 0) or 0),
                    str(getattr(runtime_result, "error_message", "") or "").strip(),
                    int(bool(getattr(runtime_result, "is_system_error", False))),
                    json.dumps(image_paths, ensure_ascii=False, sort_keys=True),
                ),
            )
            run_id = int(cursor.lastrowid)
            rows: list[tuple[object, ...]] = []
            for item in list(getattr(runtime_result, "item_results", []) or []):
                value = getattr(item, "value", None)
                try:
                    numeric_value = float(value) if value is not None else None
                except (TypeError, ValueError):
                    numeric_value = None
                rows.append(
                    (
                        run_id,
                        str(getattr(item, "camera_id", "") or "").strip(),
                        str(getattr(item, "item_id", "") or "").strip(),
                        str(getattr(item, "roi_label", "") or "").strip(),
                        str(getattr(item, "display_name", "") or "").strip(),
                        str(getattr(item, "result", "") or "").strip(),
                        numeric_value,
                        str(getattr(item, "unit", "") or "").strip(),
                        int(bool(getattr(item, "enabled", True))),
                        str(getattr(item, "detail", "") or "").strip(),
                    )
                )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO runtime_roi_results(
                        run_id, camera_id, item_id, roi_label, display_name, result,
                        value, unit, enabled, detail
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            return run_id

    def query_runs(
        self,
        *,
        start_at: str = "",
        end_at: str = "",
        product_name: str = "",
        final_result: str = "",
        limit: int = 1000,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[object] = []
        if start_at:
            clauses.append("record_time >= ?")
            params.append(str(start_at))
        if end_at:
            clauses.append("record_time <= ?")
            params.append(str(end_at))
        if product_name:
            clauses.append("product_name LIKE ?")
            params.append(f"%{str(product_name)}%")
        if final_result:
            clauses.append("final_result = ?")
            params.append(str(final_result).strip())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 5000)))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, record_time, product_name, recipe_name, final_result,
                       duration_ms, error_message, is_system_error, image_paths_json
                FROM runtime_runs
                {where}
                ORDER BY record_time DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def runtime_filter_values(self, column: str, *, limit: int = 200) -> list[str]:
        allowed_columns = {"product_name", "final_result"}
        if column not in allowed_columns:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT {column} AS value
                FROM runtime_runs
                WHERE COALESCE({column}, '') <> ''
                ORDER BY value
                LIMIT ?
                """,
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
            return [str(row["value"] or "") for row in rows if str(row["value"] or "").strip()]

    def roi_results_for_run_ids(self, run_ids: Iterable[object]) -> dict[int, list[dict]]:
        normalized_ids = []
        for run_id in run_ids:
            try:
                value = int(run_id)
            except (TypeError, ValueError):
                continue
            if value > 0 and value not in normalized_ids:
                normalized_ids.append(value)
        results: dict[int, list[dict]] = {run_id: [] for run_id in normalized_ids}
        for start in range(0, len(normalized_ids), 900):
            chunk = normalized_ids[start:start + 900]
            placeholders = ", ".join("?" for _ in chunk)
            with self.connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT id, run_id, camera_id, item_id, roi_label, display_name,
                           result, value, unit, enabled, detail
                    FROM runtime_roi_results
                    WHERE run_id IN ({placeholders})
                    ORDER BY run_id, id
                    """,
                    chunk,
                ).fetchall()
            for row in rows:
                payload = dict(row)
                results.setdefault(int(payload["run_id"]), []).append(payload)
        return results

    def roi_results_for_run(self, run_id: object) -> list[dict]:
        try:
            normalized_id = int(run_id)
        except (TypeError, ValueError):
            return []
        return self.roi_results_for_run_ids([normalized_id]).get(normalized_id, [])

    @staticmethod
    def image_paths_text(value: object) -> str:
        try:
            payload = json.loads(str(value or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return str(value or "")
        if not isinstance(payload, dict):
            return ""
        return " | ".join(
            f"{camera_id}: {path}"
            for camera_id, path in sorted(payload.items())
            if str(path or "").strip()
        )

    @staticmethod
    def _csv_roi_column(row: Mapping[str, object]) -> str:
        camera_id = str(row.get("camera_id", "") or "").strip()
        display_name = str(row.get("display_name", "") or "").strip()
        roi_label = str(row.get("roi_label", "") or "").strip()
        item_id = str(row.get("item_id", "") or "").strip()
        roi_name = display_name or roi_label or item_id
        return f"{camera_id}.{roi_name}" if camera_id and roi_name else ""

    @staticmethod
    def _csv_roi_result(value: object) -> str:
        result = str(value or "").strip().upper()
        if result in {"OK", "PASS"}:
            return "OK"
        if result == "NG":
            return "NG"
        return ""

    def export_runs_csv(
        self,
        runs: Iterable[Mapping[str, object]],
        roi_results_by_run: Mapping[int, Iterable[Mapping[str, object]]],
        path: str | Path,
    ) -> None:
        run_rows = [dict(row) for row in runs]
        roi_columns: list[str] = []
        for run in run_rows:
            try:
                run_id = int(run.get("id", 0) or 0)
            except (TypeError, ValueError):
                continue
            for roi_row in roi_results_by_run.get(run_id, []):
                column = self._csv_roi_column(roi_row)
                if column and column not in roi_columns:
                    roi_columns.append(column)
        fieldnames = [*self.RUN_COLUMNS, *roi_columns]
        with Path(path).open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for run in run_rows:
                row = {key: run.get(key, "") for key in self.RUN_COLUMNS}
                try:
                    run_id = int(run.get("id", 0) or 0)
                except (TypeError, ValueError):
                    run_id = 0
                for roi_row in roi_results_by_run.get(run_id, []):
                    column = self._csv_roi_column(roi_row)
                    result = self._csv_roi_result(roi_row.get("result"))
                    if not column:
                        continue
                    previous = str(row.get(column, "") or "").strip().upper()
                    if previous == "NG" or result == "NG":
                        row[column] = "NG"
                    elif previous == "OK" or result == "OK":
                        row[column] = "OK"
                    else:
                        row.setdefault(column, "")
                writer.writerow(row)


class PermissionService:
    def __init__(self, store: AuditStore) -> None:
        self.store = store
        self.current_user = CurrentUser()

    @property
    def user_name(self) -> str:
        return self.current_user.user_name

    @property
    def role_name(self) -> str:
        return self.current_user.role_name

    def login(self, user_name: str, password: str) -> CurrentUser | None:
        user = self.store.authenticate(user_name, password)
        if user is None:
            return None
        self.current_user = user
        return user

    def logout(self) -> None:
        self.current_user = CurrentUser()

    def has(self, permission_key: str) -> bool:
        return permission_key in self.store.permissions_for_user(self.current_user)

    def audit_context(self) -> dict[str, str]:
        return {
            "user_name": self.current_user.user_name,
            "role_name": self.current_user.role_name,
        }
