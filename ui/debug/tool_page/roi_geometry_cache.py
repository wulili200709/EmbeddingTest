"""Bounded, file-versioned geometry snapshots for annotation UI reads."""
from __future__ import annotations

import os
import math
from collections import OrderedDict

from common import labelme_io


def shape_geometry(shape: dict) -> tuple[list[tuple[float, float]] | None, tuple[int, int, int, int] | None]:
    try:
        points = [(float(x), float(y)) for x, y in shape.get("points", [])]
        if not points or any(not math.isfinite(v) for point in points for v in point):
            return None, None
        xs, ys = zip(*points)
        xywh = (int(round(min(xs))), int(round(min(ys))), int(round(max(xs)-min(xs))), int(round(max(ys)-min(ys))))
        if xywh[2] <= 0 or xywh[3] <= 0:
            xywh = None
        polygon = points if shape.get("shape_type") == "polygon" and len(points) >= 3 else None
        return polygon, xywh
    except (TypeError, ValueError, OverflowError):
        return None, None


def _file_version(path: str) -> tuple[int, int, int, int] | None:
    try:
        stat = os.stat(path)
        return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino
    except OSError:
        return None


class RoiGeometryCache:
    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[object, dict[str, dict]]] = OrderedDict()

    def shapes_for_path(self, image_path: str) -> dict[str, dict]:
        """Return a read-only-by-convention index; never cache decoded images."""
        path = os.path.normcase(os.path.abspath(labelme_io.labelme_json_of_image(image_path)))
        version = _file_version(path)
        if version is None:
            self._entries.pop(path, None)
            return {}
        fingerprint = version, _file_version(path + ".bak")
        cached = self._entries.get(path)
        if cached is not None and cached[0] == fingerprint:
            self._entries.move_to_end(path)
            return cached[1]
        shapes: dict[str, dict] = {}
        for shape in labelme_io.list_shapes_from_labelme(path):
            label = str(shape.get("label", ""))
            if label:
                shapes.setdefault(label, shape)
        # The backup reader may have restored a damaged primary file.
        fingerprint = _file_version(path), _file_version(path + ".bak")
        self._entries[path] = fingerprint, shapes
        self._entries.move_to_end(path)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return shapes
