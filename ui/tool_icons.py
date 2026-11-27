"""Scalable line icons shared by the application menus, navigation and controls."""
from PySide6 import QtCore, QtGui, QtSvg


_DRAWINGS = {
    # Menu bar
    "file": '<path d="M5 3h9l5 5v13H5zM14 3v6h5M8 13h8m-8 4h6"/>',
    "view": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "tools": '<path d="M21 6a6 6 0 0 1-8 8L6 21a2.1 2.1 0 0 1-3-3l7-7a6 6 0 0 1 8-8l-4 4 3 3z"/>',
    "control": '<path d="M3 7h5m5 0h8M3 17h10m5 0h3"/><circle cx="10.5" cy="7" r="2.5"/><circle cx="15.5" cy="17" r="2.5"/>',
    "path": '<path d="M3 8V5h6l3 3h9v3M3 10h19l-3 10H3z"/>',
    "system": '<path d="m10 3-.7 2.5-2 .9-2.2-1.1-2 3.4L5 10.4v2.9L3.1 15l2 3.5 2.4-.8 2 .9L10 21h4l.6-2.4 2-.9 2.4.8 2-3.5-1.9-1.7v-2.9L21 8.7l-2-3.4-2.2 1.1-2-.9L14 3z"/><circle cx="12" cy="12" r="3"/>',
    "language": '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.6 2.6 0 1 1 4.4 1.9c-1.1.8-1.9 1.2-1.9 3M12 17h.01"/>',

    # File and workspace selection
    "exit": '<path d="M12 2v9M7 5a8.5 8.5 0 1 0 10 0"/>',
    "switch_debug": '<rect x="7" y="7" width="10" height="13" rx="5"/><path d="m8 3 2 4m6-4-2 4M3 10l4 1m10 0 4-1M3 16l4-1m10 0 4 1M12 11v5"/>',
    "switch_runtime": '<rect x="2.5" y="4" width="19" height="14" rx="2"/><path d="m10 8 5 3-5 3zM8 21h8"/>',

    # Inspection tools
    "hardware": '<rect x="5" y="5" width="14" height="14" rx="3"/><path d="M9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3"/><rect x="9" y="9" width="6" height="6" rx="1"/>',
    "camera_tool": '<path d="M3 7h4l2-3h6l2 3h4v13H3z"/><circle cx="12" cy="13" r="4"/>',
    "io_tool": '<path d="M2 8h11m-4-4 4 4-4 4m13 4H11m4-4-4 4 4 4"/>',
    "current_product_capture_plan": '<path d="M7 3v18M11 6h9m-9 6h6m-6 6h9"/><path d="M4 6h3m-3 6h3m-3 6h3"/>',
    "positioning": '<circle cx="12" cy="12" r="7"/><path d="M12 2v5m0 10v5M2 12h5m10 0h5"/><circle cx="12" cy="12" r="1"/>',
    "template_editor": '<path d="M3 17 6 5l7 4 5-3 3 12-9-2z"/><circle cx="6" cy="5" r="1.5"/><circle cx="18" cy="6" r="1.5"/>',
    "ncc_tool": '<rect x="3" y="3" width="12" height="12" rx="1"/><path d="M9 3v12M3 9h12m4 10 3 3"/><circle cx="16" cy="16" r="4"/>',
    "auto_region": '<path d="M3 9V3h6m6 0h6v6m0 6v6h-6m-6 0H3v-6"/><rect x="7" y="8" width="10" height="8" rx="1"/>',
    "algo": '<path d="M3 3v17h18M6 15l4-5 4 2 5-7"/>',
    "margin_validation": '<path d="M4 3v18m16-18v18M7 12h10m-7-3-3 3 3 3m4-6 3 3-3 3"/>',
    "embedding_analysis": '<circle cx="5" cy="17" r="2"/><circle cx="11" cy="11" r="2"/><circle cx="17" cy="5" r="2"/><circle cx="18" cy="16" r="2"/>',
    "baseline_debug": '<path d="M2 16h20M3 12h4l3-7 4 13 3-7h4"/>',
    "lens_correction": '<circle cx="12" cy="12" r="9"/><path d="m12 3 4 7m4-3-4 7m4 3h-8m0 4-4-7m-4 3 4-7M4 7h8"/>',
    "reload_debug": '<path d="M20 10a8 8 0 1 0 0 5m0-12v7h-7"/>',

    # Runtime and image retention
    "refresh_cameras": '<path d="M12 5H8L6 8H3v11h11"/><circle cx="10" cy="13" r="2.6"/><path d="M21 12a4 4 0 1 0 .2 5m-.2-8v4h-4"/>',
    "open_mvs": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 9h18m-15-2h.1"/><circle cx="12" cy="14" r="2.7"/>',
    "connect_camera": '<path d="m14 4 6 6m-5-5 3-3m1 7 3-3M5.5 11.5l7 7m-6-6 5-5 5 5-5 5a3.5 3.5 0 0 1-5-5zM6 18l-4 4"/>',
    "disconnect_camera": '<path d="m14 5 5 5 3-3m-8-2 3-3M3 15l6 6 3-3-6-6zM3 21l2-2M4 6l3 2m1-6 1 3m7 13 1 3m2-7 3 2"/>',
    "capture": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8" cy="9" r="1.5"/><path d="m4 17 5-5 3 3 3-4 5 6"/>',
    "keep_all": '<path d="M3 8v12h15"/><rect x="6" y="4" width="15" height="13" rx="1.5"/><path d="m8 14 4-4 3 3 2-2 2 2"/>',
    "keep_ng_only": '<path d="M3 4h18l-7 8v7l-4 2v-9z"/><path d="m17 14 5 5m0-5-5 5"/>',
    "tower_light": '<rect x="8" y="2.5" width="8" height="14" rx="2"/><path d="M8 7h8m-8 5h8m-4 5v4m-5 0h10"/>',
    "foot_trigger": '<path d="M3 18h18v3H3zM5 15l12-9 3 5-8 4zM6 3v6m-3-3 3 3 3-3"/>',
    "password_release": '<rect x="4" y="10" width="15" height="11" rx="2"/><path d="M9 10V7a4 4 0 0 1 8 0m-9 8 2 2 5-5"/>',
    "change_release_password": '<circle cx="7" cy="8" r="4"/><path d="m10 11 8 8h3v-3l-2-2h-2v-2h-2M6 7h.1"/>',

    # File locations
    "open_system_root": '<path d="m2 11 10-8 10 8M5 10v10h14V10M10 20v-7h4v7"/>',
    "open_product_dir": '<path d="m3 7 9-4 9 4v10l-9 4-9-4zM3 7l9 4 9-4m-9 4v10m-5-16 10 4"/>',
    "open_session_dir": '<path d="M3 7h7l2 2h9v11H3zM5 7V4h6l2 2h6"/><path d="M7 13h6m-6 3h9"/>',
    "open_runtime_images": '<rect x="3" y="3" width="18" height="16" rx="1.5"/><circle cx="8" cy="8" r="1.5"/><path d="m4 17 5-5 3 3 4-5 4 5"/>',
    "save_image_path": '<path d="M3 12V5h6l2 2h9v5M3 12h10l-2 7H3zM17 12v9m-3-3 3 3 3-3"/>',
    "open_runtime_records": '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 4 16 4 16 0V6M4 12c0 4 16 4 16 0"/>',
    "save_runtime_records": '<path d="M13 21H4V3h12l3 3v5M7 7h6m-6 4h5m-5 4h4m7-2v8m-3-3 3 3 3-3"/>',

    # System and accounts
    "auth_login": '<path d="M12 3h8v18h-8M2 12h13m-5-5 5 5-5 5"/>',
    "auth_logout": '<path d="M11 3H3v18h8m-2-9h12m-5-5 5 5-5 5"/>',
    "change_current_password": '<rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-9 5h.1m4.9 0h.1m4.9 0h.1"/>',
    "user_permissions": '<circle cx="8" cy="7" r="3"/><path d="M2 20v-2a6 6 0 0 1 9-5m6-3 5 2v3c0 3-2 5-5 7-3-2-5-4-5-7v-3z"/>',
    "audit_log": '<rect x="4" y="3" width="16" height="18" rx="1.5"/><path d="m7 8 1 1 2-2m3 1h4M7 13h10m-10 4h7"/>',
    "runtime_records_query": '<path d="M12 20H3V3h13v5M6 7h6m-6 4h3m-3 4h2"/><circle cx="15.5" cy="14" r="4"/><path d="m18.5 17 3.5 4"/>',
    "software_versions": '<path d="m3 7 9-4 9 4-9 4zM3 12l9 4 9-4M3 17l9 4 9-4"/>',

    # Language and help
    "language_zh": '<path d="M4 6h16v10H4zM12 2v20"/>',
    "language_en": '<path d="M10 6H3v12h7M3 12h6m5 6V6l7 12V6"/>',
    "about_version": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m-2 0h4m-2-10h.01"/>',
    "detailed_camera_diagnostics": '<path d="M3 7h4l2-3h6l2 3h4v12H3z"/><path d="M5 13h4l2-4 2 7 2-4h4"/>',
}


def tool_icon(name: str, *, checkable: bool = False) -> QtGui.QIcon:
    icon = QtGui.QIcon()
    drawings = {QtGui.QIcon.State.Off: _DRAWINGS[name]}
    if checkable:
        drawings[QtGui.QIcon.State.On] = (
            _DRAWINGS[name]
            + '<circle cx="17.5" cy="17.5" r="6.3" fill="#b5def4" stroke="#303840" stroke-width="1"/>'
              '<path d="m14 17.5 2.4 2.4 4.4-5" stroke="#183244" stroke-width="2"/>'
        )
    for state, drawing in drawings.items():
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
               f'fill="none" stroke="#b5def4" stroke-width="1.8" stroke-linecap="round" '
               f'stroke-linejoin="round">{drawing}</svg>')
        renderer = QtSvg.QSvgRenderer(svg.encode())
        for size in (16, 24, 32, 48):
            pixmap = QtGui.QPixmap(size, size)
            pixmap.fill(QtCore.Qt.GlobalColor.transparent)
            with QtGui.QPainter(pixmap) as painter:
                renderer.render(painter)
            icon.addPixmap(pixmap, QtGui.QIcon.Mode.Normal, state)
    return icon
