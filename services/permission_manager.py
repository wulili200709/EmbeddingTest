from __future__ import annotations

from dataclasses import dataclass
from sqlite3 import Error as DatabaseError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infrastructure.audit_store import RuntimeRecordStore


@dataclass(frozen=True)
class ReleaseStatus:
    is_locked: bool
    has_pending_release: bool
    release_consumed: bool


class PermissionManager:
    """Manage NG lock and one-time release consumption."""

    def __init__(self, run_password: str, *, interlock_store: RuntimeRecordStore | None = None) -> None:
        self._run_password = str(run_password)
        self._interlock_store = interlock_store
        self.storage_error = ""
        try:
            state = interlock_store.load_ng_interlock() if interlock_store is not None else "waiting"
        except (DatabaseError, OSError) as exc:
            self.storage_error = str(exc)
            state = "locked"
        self._locked = state == "locked"
        self._has_pending_release = state == "released"
        self._release_consumed = False

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def has_pending_release(self) -> bool:
        return self._has_pending_release

    def status(self) -> ReleaseStatus:
        return ReleaseStatus(
            is_locked=self._locked,
            has_pending_release=self._has_pending_release,
            release_consumed=self._release_consumed,
        )

    def lock_for_ng(self) -> None:
        self._locked = True
        self._has_pending_release = False
        self._release_consumed = False
        self._save_state("locked")

    def _save_state(self, state: str) -> bool:
        try:
            if self._interlock_store is not None:
                self._interlock_store.save_ng_interlock(state)
        except (DatabaseError, OSError) as exc:
            self.storage_error = str(exc)
            return False
        self.storage_error = ""
        return True

    def clear_lock(self) -> None:
        if (self._locked or self._has_pending_release) and not self._save_state("waiting"):
            raise RuntimeError(self.storage_error)
        self._locked = False
        self._has_pending_release = False
        self._release_consumed = False

    def update_password(self, password: str) -> None:
        self._run_password = str(password)

    def try_release_once(self, password: str) -> bool:
        if str(password) != self._run_password:
            return False
        return self._grant_release_once()

    def release_after_ng_box_confirmation(self) -> bool:
        if not self._locked:
            return False
        return self._grant_release_once()

    def _grant_release_once(self) -> bool:
        if not self._save_state("released"):
            return False
        self._locked = False
        self._has_pending_release = True
        self._release_consumed = False
        return True

    def consume_release_if_needed(self) -> bool:
        if not self._has_pending_release:
            return False
        if not self._save_state("waiting"):
            raise RuntimeError(self.storage_error)
        self._has_pending_release = False
        self._release_consumed = True
        return True

    def restore_pending_release(self) -> None:
        if self._release_consumed:
            if not self._save_state("released"):
                raise RuntimeError(self.storage_error)
            self._has_pending_release = True
            self._release_consumed = False

    def reset_after_success(self) -> None:
        self.clear_lock()
