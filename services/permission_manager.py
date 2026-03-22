from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReleaseStatus:
    is_locked: bool
    has_pending_release: bool
    release_consumed: bool


class PermissionManager:
    """Manage NG lock and one-time release consumption."""

    def __init__(self, run_password: str) -> None:
        self._run_password = str(run_password)
        self._locked = False
        self._has_pending_release = False
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

    def clear_lock(self) -> None:
        self._locked = False
        self._has_pending_release = False
        self._release_consumed = False

    def update_password(self, password: str) -> None:
        self._run_password = str(password)

    def try_release_once(self, password: str) -> bool:
        if str(password) != self._run_password:
            return False
        self._locked = False
        self._has_pending_release = True
        self._release_consumed = False
        return True

    def consume_release_if_needed(self) -> bool:
        if not self._has_pending_release:
            return False
        self._has_pending_release = False
        self._release_consumed = True
        return True

    def restore_pending_release(self) -> None:
        if self._release_consumed:
            self._has_pending_release = True
            self._release_consumed = False

    def reset_after_success(self) -> None:
        self._locked = False
        self._has_pending_release = False
        self._release_consumed = False
