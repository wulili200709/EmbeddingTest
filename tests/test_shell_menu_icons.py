import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets

from ui.shell.main_window import MainWindow
from ui.tool_icons import tool_icon


class ShellMenuIconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_every_menu_entry_uses_the_shared_icons(self):
        # Build the production menu without starting cameras or loading a product.
        window = MainWindow.__new__(MainWindow)
        QtWidgets.QMainWindow.__init__(window)
        self.addCleanup(window.deleteLater)
        window.tool_page = Mock()
        window.runtime_page = Mock()
        window._runtime_capture_policy = "all"
        icon_keys = set()

        def record_icon(name, **options):
            icon = tool_icon(name, **options)
            icon_keys.add(icon.cacheKey())
            return icon

        with patch("ui.shell.chrome.tool_icon", side_effect=record_icon):
            window._build_menu_bar()

        entries = list(window.menuWidget().findChildren(QtWidgets.QToolButton))
        for menu in window.findChildren(QtWidgets.QMenu):
            entries.extend(action for action in menu.actions() if not action.isSeparator())
        for entry in entries:
            with self.subTest(text=entry.text()):
                icon = entry.icon()
                self.assertIn(icon.cacheKey(), icon_keys)
                self.assertFalse(icon.pixmap(16, 16).isNull())
                if entry.isCheckable():
                    size = QtCore.QSize(16, 16)
                    on = icon.pixmap(size, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On)
                    off = icon.pixmap(size, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
                    self.assertNotEqual(on.toImage(), off.toImage())
