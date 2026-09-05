"""Interaction checks for zooming and panning the debug ROI canvas."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from ui.debug.roi_canvas_pyside6 import RoiCanvas


class RoiCanvasPanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_canvas(self) -> RoiCanvas:
        canvas = RoiCanvas()
        canvas.resize(200, 200)
        pixmap = QtGui.QPixmap(1000, 500)
        pixmap.fill(QtGui.QColor("white"))
        canvas.set_image("memory", pixmap)
        canvas.show()
        self.app.processEvents()
        canvas.set_zoom(2.0)
        return canvas

    def test_middle_button_pans_without_creating_roi(self):
        canvas = self.make_canvas()
        try:
            QtTest.QTest.mousePress(
                canvas,
                QtCore.Qt.MouseButton.MiddleButton,
                pos=QtCore.QPoint(100, 100),
            )
            QtTest.QTest.mouseMove(canvas, QtCore.QPoint(50, 100), delay=5)
            QtTest.QTest.mouseRelease(
                canvas,
                QtCore.Qt.MouseButton.MiddleButton,
                pos=QtCore.QPoint(50, 100),
            )
            self.app.processEvents()
            self.assertAlmostEqual(canvas.pan_offset().x(), -50.0, places=1)
            self.assertEqual(canvas.roi.xywh, None)
        finally:
            canvas.close()

    def test_space_left_button_pans_and_plain_left_button_still_draws_roi(self):
        canvas = self.make_canvas()
        try:
            QtTest.QTest.keyPress(canvas, QtCore.Qt.Key.Key_Space)
            QtTest.QTest.mousePress(
                canvas,
                QtCore.Qt.MouseButton.LeftButton,
                pos=QtCore.QPoint(100, 100),
            )
            QtTest.QTest.mouseMove(canvas, QtCore.QPoint(140, 100), delay=5)
            QtTest.QTest.mouseRelease(
                canvas,
                QtCore.Qt.MouseButton.LeftButton,
                pos=QtCore.QPoint(140, 100),
            )
            QtTest.QTest.keyRelease(canvas, QtCore.Qt.Key.Key_Space)
            self.app.processEvents()
            self.assertAlmostEqual(canvas.pan_offset().x(), 40.0, places=1)
            self.assertEqual(canvas.roi.xywh, None)

            QtTest.QTest.mousePress(
                canvas,
                QtCore.Qt.MouseButton.LeftButton,
                pos=QtCore.QPoint(80, 70),
            )
            QtTest.QTest.mouseMove(canvas, QtCore.QPoint(120, 110), delay=5)
            QtTest.QTest.mouseRelease(
                canvas,
                QtCore.Qt.MouseButton.LeftButton,
                pos=QtCore.QPoint(120, 110),
            )
            self.app.processEvents()
            self.assertIsNotNone(canvas.roi.xywh)
        finally:
            canvas.close()

    def test_middle_double_click_resets_zoom_and_pan(self):
        canvas = self.make_canvas()
        try:
            canvas._pan_offset = QtCore.QPointF(-60.0, 0.0)
            canvas._update_scaled_pixmap()
            QtTest.QTest.mouseDClick(
                canvas,
                QtCore.Qt.MouseButton.MiddleButton,
                pos=QtCore.QPoint(100, 100),
            )
            self.app.processEvents()
            self.assertAlmostEqual(canvas.zoom_factor(), 1.0)
            self.assertEqual(canvas.pan_offset(), QtCore.QPointF())
        finally:
            canvas.close()


if __name__ == "__main__":
    unittest.main()
