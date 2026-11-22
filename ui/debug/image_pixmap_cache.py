"""Share a few decoded file images between the annotation and main canvases."""
from __future__ import annotations

import os
from collections import OrderedDict

from PySide6 import QtGui

_MAX_BYTES = 48 * 1024 * 1024
_MAX_ENTRIES = 3
_entries: OrderedDict[str, tuple[object, QtGui.QPixmap, int]] = OrderedDict()


def pixmap_from_path(path: str) -> QtGui.QPixmap:
    key = os.path.normcase(os.path.abspath(path))
    try:
        stat = os.stat(key)
        version = stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino
    except OSError:
        _entries.pop(key, None)
        return QtGui.QPixmap()
    cached = _entries.get(key)
    if cached is not None and cached[0] == version:
        _entries.move_to_end(key)
        return QtGui.QPixmap(cached[1])
    _entries.pop(key, None)
    pixmap = QtGui.QPixmap(key)
    size = pixmap.width() * pixmap.height() * pixmap.depth() // 8
    if not pixmap.isNull() and size <= _MAX_BYTES:
        _entries[key] = version, QtGui.QPixmap(pixmap), size
        while len(_entries) > _MAX_ENTRIES or sum(e[2] for e in _entries.values()) > _MAX_BYTES:
            _entries.popitem(last=False)
    return pixmap
