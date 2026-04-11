from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from algorithms.image_io import imread, imwrite


class ImageIoUnicodePathTest(unittest.TestCase):
    def _unicode_dir(self) -> Path:
        base = Path(tempfile.mkdtemp())
        target = base / "中文目录"
        target.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(base, ignore_errors=True))
        return target

    def test_imread_supports_non_ascii_path(self) -> None:
        folder = self._unicode_dir()
        image_path = folder / "样本图.png"
        image = np.zeros((8, 9, 3), dtype=np.uint8)
        image[:, :, 1] = 128
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        encoded.tofile(str(image_path))

        loaded = imread(str(image_path), cv2.IMREAD_COLOR)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, image.shape)
        self.assertTrue(np.array_equal(loaded, image))

    def test_imwrite_supports_non_ascii_path(self) -> None:
        folder = self._unicode_dir()
        image_path = folder / "输出图.png"
        image = np.zeros((6, 7, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        image[:, :, 2] = 64

        self.assertTrue(imwrite(str(image_path), image))
        self.assertTrue(image_path.exists())

        raw = np.fromfile(str(image_path), dtype=np.uint8)
        loaded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, image.shape)
        self.assertTrue(np.array_equal(loaded, image))


if __name__ == "__main__":
    unittest.main()
