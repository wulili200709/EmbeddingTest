"""Runtime multi-pin label layout regression checks."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets

from ui.window_common import _draw_runtime_pin_label


class RuntimeMultiPinOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_labels_stay_in_image_and_use_enough_rows_to_avoid_overlap(self):
        size = QtCore.QSize(420, 240)
        image = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(image)
        occupied: list[QtCore.QRectF] = []
        points = [
            (1.0, 232.0),
            (178.0, 232.0),
            (186.0, 232.0),
            (194.0, 232.0),
            (202.0, 232.0),
            (210.0, 232.0),
            (419.0, 232.0),
        ]
        try:
            boxes = [
                _draw_runtime_pin_label(
                    painter,
                    point,
                    f"P{index}-P{index + 1}: 20.0px",
                    QtGui.QColor("#22c55e"),
                    source_size=size,
                    occupied_boxes=occupied,
                )
                for index, point in enumerate(points, start=1)
            ]
        finally:
            painter.end()

        image_rect = QtCore.QRectF(0.0, 0.0, float(size.width()), float(size.height()))
        self.assertEqual(len(occupied), len(points))
        self.assertTrue(all(image_rect.contains(box) for box in boxes))
        self.assertTrue(all(box.top() < points[index][1] for index, box in enumerate(boxes)))
        for index, box in enumerate(boxes):
            for other in boxes[index + 1 :]:
                self.assertFalse(box.intersects(other))
        self.assertGreater(len({round(box.top(), 3) for box in boxes}), 2)
        self.assertGreaterEqual(boxes[0].left(), 0.0)
        self.assertLessEqual(boxes[-1].right(), float(size.width()))


if __name__ == "__main__":
    unittest.main()
