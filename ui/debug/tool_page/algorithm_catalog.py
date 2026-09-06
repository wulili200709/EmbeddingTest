from __future__ import annotations

from ui.i18n import tr


ALGORITHM_GROUPS = [
    (
        "learning",
        [
            ("High-Accuracy Learning Tool", "efficientnet_b0", True),
            ("Lightweight Learning Tool", "mobilenet_v3_small", True),
            ("Balanced Learning Tool", "mobilenet_v3_large", True),
        ],
    ),
    (
        "traditional",
        [
            ("Hue Tool", "meanhsv_h", True),
            ("Grayscale Tool", "meanintensity", True),
            ("Deviation Tool", "meanstd", True),
            ("Value Tool", "meanhsv_v", True),
            ("Saturation Tool", "meanhsv_s", True),
        ],
    ),
    (
        "measurement",
        [
            ("Find Line", "find_line", True),
            ("Pin Tip Point", "pin_tip_point", True),
            ("Multi Pin Measurement", "multi_pin_tip_height", True),
            ("Bright Block Center", "bright_block_center", True),
            ("Center Distance", "center_distance", True),
            ("Find Circle", "find_circle", False),
        ],
    ),
]

ALGORITHM_DISPLAY_NAMES = {
    code: label
    for _group_name, items in ALGORITHM_GROUPS
    for label, code, enabled in items
    if enabled
}

ALGORITHM_GROUP_KEYS = {
    "learning": "debug.algorithm_group.learning",
    "traditional": "debug.algorithm_group.traditional",
    "measurement": "debug.algorithm_group.measurement",
}

ALGORITHM_DISPLAY_KEYS = {
    "efficientnet_b0": "debug.algorithm.efficientnet_b0",
    "mobilenet_v3_small": "debug.algorithm.mobilenet_v3_small",
    "mobilenet_v3_large": "debug.algorithm.mobilenet_v3_large",
    "meanhsv_h": "debug.algorithm.meanhsv_h",
    "meanintensity": "debug.algorithm.meanintensity",
    "meanstd": "debug.algorithm.meanstd",
    "meanhsv_v": "debug.algorithm.meanhsv_v",
    "meanhsv_s": "debug.algorithm.meanhsv_s",
    "find_circle": "debug.algorithm.find_circle",
    "find_line": "debug.algorithm.find_line",
    "find_line_subpix": "debug.algorithm.find_line_subpix",
    "pin_tip_point": "debug.algorithm.pin_tip_point",
    "multi_pin_tip_height": "debug.algorithm.multi_pin_tip_height",
    "bright_block_center": "debug.algorithm.bright_block_center",
    "pin_center_distance": "debug.algorithm.pin_center_distance",
    "bright_block_y_distance": "debug.algorithm.bright_block_y_distance",
    "center_distance": "debug.algorithm.center_distance",
    "line_distance": "debug.algorithm.line_distance",
    "line_distance_ref_normal": "debug.algorithm.line_distance_ref_normal",
    "point_line_distance": "debug.algorithm.point_line_distance",
}


def algorithm_group_display_name(group_name: str) -> str:
    key = ALGORITHM_GROUP_KEYS.get(str(group_name or ""))
    return tr(key) if key else str(group_name or "")


def algorithm_display_name(label: str, code: str) -> str:
    key = ALGORITHM_DISPLAY_KEYS.get(str(code or ""))
    return tr(key) if key else str(label or code or "")
