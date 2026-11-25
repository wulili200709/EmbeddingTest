from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets

from ui.runtime.runtime_mode_pyside6 import RuntimeImageView


class RuntimeImageViewZoomTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.view = RuntimeImageView("Cam1")
        self.view.resize(400, 300)
        self.view.show()
        self.app.processEvents()
        self.pixmap = QtGui.QPixmap(800, 400)
        self.pixmap.fill(QtGui.QColor("red"))
        self.view.set_runtime_pixmap(self.pixmap)

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        self.app.processEvents()

    def test_zoom_keeps_mouse_anchor_and_reset_restores_fit(self) -> None:
        self.assertEqual(self.view.zoom_factor(), 1.0)

        self.view.set_zoom(2.0, anchor=QtCore.QPointF(300.0, 150.0))

        self.assertEqual(self.view.zoom_factor(), 2.0)
        self.assertLess(self.view.pan_offset().x(), 0.0)
        self.assertEqual(self.view.pan_offset().y(), 0.0)
        self.assertEqual(self.view.pixmap().size(), self.view.size())

        self.view.reset_view()

        self.assertEqual(self.view.zoom_factor(), 1.0)
        self.assertEqual(self.view.pan_offset(), QtCore.QPointF())

    def test_same_size_frame_preserves_zoom_but_resolution_change_resets_it(self) -> None:
        self.view.set_zoom(3.0)
        replacement = QtGui.QPixmap(800, 400)
        replacement.fill(QtGui.QColor("blue"))

        self.view.set_runtime_pixmap(replacement)

        self.assertEqual(self.view.zoom_factor(), 3.0)

        different_size = QtGui.QPixmap(640, 480)
        different_size.fill(QtGui.QColor("green"))
        self.view.set_runtime_pixmap(different_size)

        self.assertEqual(self.view.zoom_factor(), 1.0)
        self.assertEqual(self.view.pan_offset(), QtCore.QPointF())

    def test_wheel_event_changes_zoom(self) -> None:
        local_pos = QtCore.QPointF(250.0, 150.0)
        global_pos = QtCore.QPointF(self.view.mapToGlobal(local_pos.toPoint()))
        event = QtGui.QWheelEvent(
            local_pos,
            global_pos,
            QtCore.QPoint(),
            QtCore.QPoint(0, 120),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.ScrollUpdate,
            False,
        )

        self.view.wheelEvent(event)

        self.assertTrue(event.isAccepted())
        self.assertGreater(self.view.zoom_factor(), 1.0)

    def test_left_drag_pans_and_double_click_resets_view(self) -> None:
        self.view.set_zoom(2.0)
        def mouse_event(event_type, pos, button, buttons):
            global_pos = QtCore.QPointF(self.view.mapToGlobal(pos.toPoint()))
            return QtGui.QMouseEvent(
                event_type,
                pos,
                pos,
                global_pos,
                button,
                buttons,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )

        press = mouse_event(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(200.0, 150.0),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
        )
        move = mouse_event(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(250.0, 150.0),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.LeftButton,
        )
        release = mouse_event(
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QPointF(250.0, 150.0),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.NoButton,
        )

        self.view.mousePressEvent(press)
        self.view.mouseMoveEvent(move)
        self.view.mouseReleaseEvent(release)

        self.assertGreater(self.view.pan_offset().x(), 0.0)

        double_click = mouse_event(
            QtCore.QEvent.Type.MouseButtonDblClick,
            QtCore.QPointF(250.0, 150.0),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
        )
        self.view.mouseDoubleClickEvent(double_click)

        self.assertEqual(self.view.zoom_factor(), 1.0)
        self.assertEqual(self.view.pan_offset(), QtCore.QPointF())


if __name__ == "__main__":
    unittest.main()
