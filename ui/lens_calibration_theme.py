"""Window-local dark theme for the standalone and embedded calibration demo."""
from __future__ import annotations

from PySide6 import QtGui

IMAGE_BACKGROUND = "#1f1f1f"

STYLE = """
QWidget { font-family: 'Microsoft YaHei UI'; font-size: 13px; color: #e0e0e0; background: #2d2d2d; }
QLabel { background: transparent; }
QLabel#heading { font-size: 25px; font-weight: 700; color: #f0f0f0; }
QLabel#note { color: #b8b8b8; font-size: 12px; }
QLabel#profile { background: #2d3c4e; color: #d1e6ff; border: 1px solid #435b78; border-radius: 4px; padding: 10px; }
QLabel#viewTitle { font-size: 14px; font-weight: 700; padding: 7px; }
QLabel#metrics { background: #293b31; color: #c0e9ce; border: 1px solid #41634d; padding: 12px; border-radius: 4px; }
QLabel#status { padding: 8px; color: #d8d8d8; }
QLabel#status[error=true] { color: #ffb3ad; background: #4a2b2b; border: 1px solid #78504b; border-radius: 4px; }
QGroupBox { background: #363636; border: 1px solid #505050; border-radius: 4px; margin-top: 12px; padding-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #e0e0e0; }
QTableWidget { background: #262626; alternate-background-color: #2c2c2c; border: 1px solid #505050; selection-background-color: #36556d; selection-color: white; }
QTableWidget::item { padding: 3px 6px; border-bottom: 1px solid #363636; }
QTableWidget::item:selected { background: #36556d; color: #ffffff; }
QHeaderView::section { background: #393939; color: #d8d8d8; padding: 6px; border: none; border-bottom: 1px solid #545454; }
QPushButton { background: #444444; color: #e0e0e0; border: 1px solid #5a5a5a; border-radius: 3px; padding: 7px 10px; }
QPushButton:hover { background: #505050; border-color: #7b7b7b; }
QPushButton:pressed { background: #383838; }
QPushButton:focus { border-color: #6ec0ff; }
QPushButton#primary { background: #286d9f; color: #ffffff; border-color: #4c94c5; }
QPushButton#primary:hover { background: #347fae; border-color: #6ec0ff; }
QPushButton#primary:pressed { background: #225c87; }
QPushButton:disabled, QPushButton#primary:disabled { background: #333333; color: #909090; border-color: #484848; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #333333; color: #ededed; border: 1px solid #626262; border-radius: 3px; padding: 5px; selection-background-color: #6ec0ff; selection-color: #1a1a1a; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #6ec0ff; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { background: #303030; color: #939393; border-color: #484848; }
QComboBox QAbstractItemView { background: #333333; color: #ededed; selection-background-color: #426784; selection-color: #ffffff; border: 1px solid #626262; }
QCheckBox, QRadioButton { background: transparent; spacing: 6px; }
QCheckBox:disabled, QRadioButton:disabled { color: #939393; }
QListWidget, QPlainTextEdit, QTextEdit { background: #262626; color: #e0e0e0; border: 1px solid #5a5a5a; border-radius: 3px; selection-background-color: #426784; selection-color: #ffffff; }
QListWidget::item:selected { background: #426784; color: #ffffff; }
QListWidget::item:hover { background: #3d4d59; }
QPlainTextEdit { font-family: Consolas, 'Microsoft YaHei UI'; font-size: 12px; }
QTabWidget::pane { background: #2d2d2d; border: 1px solid #505050; }
QTabBar::tab { background: #3a3a3a; color: #bcbcbc; padding: 9px 16px; border: 1px solid #505050; border-bottom: 2px solid #505050; }
QTabBar::tab:selected { background: #454545; color: #ffffff; border-bottom-color: #6ec0ff; }
QGraphicsView { background: #1f1f1f; border: 1px solid #505050; }
QSplitter::handle { background: #454545; }
QProgressBar { background: #262626; color: #ffffff; border: 1px solid #565656; border-radius: 3px; text-align: center; max-height: 16px; }
QProgressBar::chunk { background: #396c91; border-radius: 2px; }
QScrollBar:vertical { background: #292929; width: 12px; margin: 0px; }
QScrollBar:horizontal { background: #292929; height: 12px; margin: 0px; }
QScrollBar::handle:vertical { background: #606060; min-height: 26px; border-radius: 3px; }
QScrollBar::handle:horizontal { background: #606060; min-width: 26px; border-radius: 3px; }
QScrollBar::handle:hover { background: #797979; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; }
QScrollBar::add-page, QScrollBar::sub-page { background: #292929; }
QToolTip { background: #3a3a3a; color: #ededed; border: 1px solid #777777; padding: 4px; }
QMenu { background: #333333; color: #e0e0e0; border: 1px solid #626262; }
QMenu::item:selected { background: #426784; color: #ffffff; }
"""


def apply_lens_calibration_theme(window):
    """Style this window without replacing the application's shared theme."""
    palette = QtGui.QPalette(window.palette())
    roles = QtGui.QPalette.ColorRole
    for role, color in (
        (roles.Window, "#2d2d2d"), (roles.WindowText, "#e0e0e0"),
        (roles.Base, "#262626"), (roles.AlternateBase, "#363636"),
        (roles.Text, "#e0e0e0"), (roles.Button, "#444444"),
        (roles.ButtonText, "#e0e0e0"), (roles.Highlight, "#6ec0ff"),
        (roles.HighlightedText, "#1a1a1a"), (roles.ToolTipBase, "#3a3a3a"),
        (roles.ToolTipText, "#ededed"), (roles.Link, "#8dccff"),
    ):
        palette.setColor(role, QtGui.QColor(color))
    for role in (roles.Text, roles.ButtonText, roles.WindowText):
        palette.setColor(QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor("#939393"))
    window.setPalette(palette)
    window.setStyleSheet(STYLE)
