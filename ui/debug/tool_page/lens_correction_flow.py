"""Keep authoring samples in the currently selected camera coordinate space."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6 import QtWidgets

from services import lens_correction
from ui.debug.tool_page.camera_roles import camera_role_from_path
from ui.lens_calibration_text import diagnostic_text, lens_label


def _stamp(path):
    try:
        stat = Path(path).stat()
        return stat.st_size, stat.st_mtime_ns
    except OSError:
        return None


def sample_source(owner, path: str, role: str) -> str:
    """Old manually exported Demo inputs retain their original disabled behavior."""
    if lens_correction.get_config(owner.session.product_dir, role).enabled:
        return lens_correction.source_image_path(path)
    if lens_correction.is_managed_image(path):
        return lens_correction.source_image_path(path)
    return str(path)


def prepare_sample(owner, path: str, *, camera_role: str = "") -> str:
    role = camera_role or camera_role_from_path(path) or owner.current_camera_role()
    validator = getattr(owner, "_validate_lens_input", None)
    if callable(validator):
        validator(path, role)
    product_dir = owner.session.product_dir
    # Returning to the original source is deliberate when the user switches
    # versions. Never apply a new map to the pixels of a previous correction.
    source = sample_source(owner, path, role)
    key = (product_dir, role, str(path), source,
           lens_correction.active_signature(product_dir, role),
           _stamp(source), _stamp(Path(path).with_suffix(".undistortion.json")))
    cache = getattr(owner, "_lens_sample_path_cache", None)
    if cache is None:
        cache = owner._lens_sample_path_cache = {}
    entry = cache.get(key)
    if entry and entry[1:] == (_stamp(entry[0]), _stamp(Path(entry[0]).with_suffix(".undistortion.json"))):
        return entry[0]
    prepared = lens_correction.prepare_path(source, product_dir, role)
    lens_correction.validate_prepared_path(prepared, product_dir, role)
    if len(cache) >= 1024:
        cache.clear()
    cache[key] = (prepared, _stamp(prepared), _stamp(Path(prepared).with_suffix(".undistortion.json")))
    return prepared


def refresh_sample_paths(owner) -> None:
    """Retain failed samples visibly, but block their display/train/test paths."""
    errors = {}
    converted = {}
    for attr in ("train_files", "test_files", "ok_files", "ng_files"):
        values = []
        for path in list(getattr(owner, attr, []) or []):
            if path not in converted:
                try:
                    converted[path] = prepare_sample(owner, path)
                except Exception as exc:
                    converted[path] = path
                    errors[path] = str(exc)
            values.append(converted[path])
        setattr(owner, attr, list(dict.fromkeys(values)))
    # OK/NG classifications survive a coordinate change. Rectangle/polygon
    # geometry is stored in separate image JSON files and is never copied.
    annotations = getattr(owner, "_sample_roi_annotations_by_path", {})
    for old, new in converted.items():
        old_key, new_key = os.path.normpath(old), os.path.normpath(new)
        if old_key != new_key and old_key in annotations and new_key not in annotations:
            annotations[new_key] = dict(annotations[old_key])
    owner._lens_sample_errors = errors


def open_settings(owner) -> None:
    from ui.lens_correction_dialog import open_lens_correction_settings

    data = owner.session.load_session()
    open_lens_correction_settings(
        owner, owner.session.product_dir, owner.configured_camera_roles(),
        data.runtime_camera_serials,
    )


def reload_context(owner) -> None:
    """Clear coordinate-dependent caches after an explicit settings save."""
    previous_image = owner.canvas.image_path()
    owner._lens_sample_path_cache = {}
    owner.session._refresh_paths()
    owner.algo.load_params(owner.session.product_params_path)
    owner.algo.model = None
    owner._shape_recipes_by_role = {}
    owner._shape_match_ms_by_image = {}
    owner._shape_autogen_ms_by_image = {}
    owner._roi_results_by_image = {}
    owner._current_result_rows = []
    owner._clear_training_roi_review_state()
    owner.table.setRowCount(0)
    owner.canvas.clear_image()
    owner.test_execution_controller._runtime_roi_shapes_by_image.clear()
    owner._apply_current_role_recipe_state()
    owner._refresh_lists()
    owner._reload_inspection_items()
    owner._sync_footer()
    owner._save_session()
    owner._save_sample_roi_annotations()
    owner.inspectionItemsChanged.emit()
    owner.sessionLoaded.emit()
    if previous_image:
        try:
            owner._load_canvas_image(prepare_sample(owner, previous_image))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(owner, lens_label("镜头矫正", "Lens correction"), diagnostic_text(exc))
    text = lens_label("镜头矫正设置已保存", "Lens settings saved")
    if owner._lens_sample_errors:
        text += "\n" + "\n".join(diagnostic_text(err) for err in list(owner._lens_sample_errors.values())[:3])
        QtWidgets.QMessageBox.warning(owner, lens_label("镜头矫正", "Lens correction"), text)
    owner.lbl_status.setText(text)
