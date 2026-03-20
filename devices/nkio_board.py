from __future__ import annotations

import threading
from pathlib import Path

from .nkio_errors import raise_for_code
from .nkio_raw import NkioRawLib


def get_bit(value: int, bit: int) -> bool:
    return bool(int(value) & (1 << int(bit)))


def set_bit(value: int, bit: int, on: bool) -> int:
    mask = 1 << int(bit)
    if on:
        return int(value) | mask
    return int(value) & ~mask


class NkioBoard:
    """Board-level wrapper that hides byte/word access details."""

    def __init__(self, config_file: str | Path, dll_path: str | Path | None = None) -> None:
        self.config_file = Path(config_file)
        self.raw = NkioRawLib(dll_path=dll_path)
        self._opened = False
        self._do_word_cache = 0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def do_word_cache(self) -> int:
        return int(self._do_word_cache)

    def open(self) -> None:
        if self._opened:
            return
        if not self.config_file.exists():
            raise FileNotFoundError(self.config_file)
        ret = self.raw.library_init(self.config_file)
        raise_for_code(ret, "NKDIO_LibraryInit", detail=str(self.config_file))
        self._opened = True
        self._do_word_cache = self._safe_read_do_word()

    def close(self) -> None:
        if not self._opened:
            return
        self.raw.library_deinit()
        self._opened = False

    def read_di_byte(self, index: int) -> int:
        self._ensure_open()
        ret, value = self.raw.read_di_byte(index)
        raise_for_code(ret, "NKDIO_PollingReadDiByte", detail=f"index={index}")
        return value

    def read_di_word(self) -> int:
        self._ensure_open()
        ret, value = self.raw.read_di_word(0)
        raise_for_code(ret, "NKDIO_PollingReadDiWord", detail="index=0")
        return value

    def read_do_byte(self, index: int) -> int:
        self._ensure_open()
        ret, value = self.raw.read_do_byte(index)
        raise_for_code(ret, "NKDIO_PollingReadDoByte", detail=f"index={index}")
        return value

    def read_do_word(self) -> int:
        self._ensure_open()
        with self._lock:
            value = self._safe_read_do_word()
            self._do_word_cache = value
            return value

    def write_do_byte(self, index: int, value: int) -> None:
        self._ensure_open()
        with self._lock:
            ret = self.raw.write_do_byte(index, value)
            raise_for_code(ret, "NKDIO_PollingWriteDoByte", detail=f"index={index}, value=0x{value:02X}")
            self._do_word_cache = self._safe_read_do_word()

    def write_do_word(self, value: int) -> None:
        self._ensure_open()
        value &= 0xFFFF
        with self._lock:
            ret = self.raw.write_do_word(0, value)
            raise_for_code(ret, "NKDIO_PollingWriteDoWord", detail=f"index=0, value=0x{value:04X}")
            self._do_word_cache = value

    def read_di_channel(self, channel: int) -> bool:
        self._validate_channel(channel)
        return get_bit(self.read_di_word(), channel)

    def read_do_channel(self, channel: int) -> bool:
        self._validate_channel(channel)
        return get_bit(self.read_do_word(), channel)

    def write_do_channel(self, channel: int, on: bool) -> None:
        self._validate_channel(channel)
        with self._lock:
            current = self._do_word_cache if self._opened else 0
            if self._opened:
                current = self._safe_read_do_word()
            updated = set_bit(current, channel, on)
            ret = self.raw.write_do_word(0, updated)
            raise_for_code(
                ret,
                "NKDIO_PollingWriteDoWord",
                detail=f"index=0, channel={channel}, on={int(bool(on))}, value=0x{updated:04X}",
            )
            self._do_word_cache = updated

    def set_do_channels(self, updates: dict[int, bool]) -> None:
        self._ensure_open()
        for channel in updates:
            self._validate_channel(channel)
        with self._lock:
            current = self._safe_read_do_word()
            updated = current
            for channel, on in updates.items():
                updated = set_bit(updated, channel, bool(on))
            ret = self.raw.write_do_word(0, updated)
            raise_for_code(ret, "NKDIO_PollingWriteDoWord", detail=f"index=0, value=0x{updated:04X}")
            self._do_word_cache = updated

    def _safe_read_do_word(self) -> int:
        ret, value = self.raw.read_do_word(0)
        raise_for_code(ret, "NKDIO_PollingReadDoWord", detail="index=0")
        return value

    def _ensure_open(self) -> None:
        if not self._opened:
            raise RuntimeError("NkioBoard is not open")

    @staticmethod
    def _validate_channel(channel: int) -> None:
        channel = int(channel)
        if channel < 0 or channel > 15:
            raise ValueError(f"channel out of range: {channel}")
