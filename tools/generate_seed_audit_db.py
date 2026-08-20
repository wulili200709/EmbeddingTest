"""Create the clean user database shipped with the desktop application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.audit_store import AuditStore


DEFAULT_ADMIN_USERS = (
    "XD8XSA",
    "OA45EM",
    "D3BIPY",
    "SP763V",
    "VYZQXO",
    "YQCXN7",
    "DE7TPA",
    "BJA3BS",
    "X0OU72",
    "TSBCTK",
    "F3WMDT",
    "GZD6UP",
)
DEFAULT_ADMIN_PASSWORD = "123456"


def create_seed_database(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stale_temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    if stale_temporary_path.exists():
        stale_temporary_path.unlink()
    if output_path.exists():
        output_path.unlink()

    store = AuditStore(output_path)
    existing_users = {str(row.get("user_name", "")) for row in store.users()}
    for user_name in DEFAULT_ADMIN_USERS:
        if user_name in existing_users:
            continue
        store.create_user(
            user_name,
            DEFAULT_ADMIN_PASSWORD,
            "admin",
            enabled=True,
            must_change_password=False,
        )

    conn = store.connect()
    try:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("runtime_runs", "runtime_roi_results", "audit_events")
        }
        if any(counts.values()):
            raise RuntimeError(f"Seed database must have no history: {counts}")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    create_seed_database(args.output)
    print(f"Created clean account database: {args.output}")


if __name__ == "__main__":
    main()
