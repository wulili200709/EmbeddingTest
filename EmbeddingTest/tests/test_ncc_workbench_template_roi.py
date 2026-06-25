from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


from ncc.model import NccMatchModel, NccMatchRect, NccReferenceRegion, load_model, save_model
from ncc.ui.workbench_dialog import NccMatchWorkbenchDialog


class _Value:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _Check:
    def __init__(self, checked: bool):
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _LineEdit:
    def __init__(self, text: str):
        self._text = text

    def text(self) -> str:
        return self._text


class _TextBox:
    def setPlainText(self, _text: str) -> None:
        pass


class _Signal:
    def emit(self, *_args) -> None:
        pass


class _Canvas:
    def __init__(self, roi):
        self._roi = roi

    def roi_xywh(self):
        return self._roi

    def has_image(self) -> bool:
        return True


class _MoveCanvas:
    _pixmap = None


def _dialog_for_options(model_path: str, model: NccMatchModel) -> NccMatchWorkbenchDialog:
    dialog = NccMatchWorkbenchDialog.__new__(NccMatchWorkbenchDialog)
    dialog._model_path = model_path
    dialog._model = model
    dialog._loading_model = False
    dialog._reference_regions = []
    dialog.modelSaved = _Signal()
    dialog.txt_model_summary = _TextBox()
    dialog.edt_display_name = _LineEdit(model.display_name)
    dialog.spn_target_num = _Value(7)
    dialog.spn_score = _Value(0.42)
    dialog.spn_overlap = _Value(0.1)
    dialog.spn_min_area = _Value(128)
    dialog.spn_angle_start = _Value(-90.0)
    dialog.spn_angle_end = _Value(90.0)
    dialog.chk_use_simd = _Check(False)
    dialog.chk_use_subpixel = _Check(True)
    dialog.chk_bitwise_not = _Check(True)
    dialog.chk_stop_layer1 = _Check(False)
    dialog.chk_pose_refine = _Check(True)
    return dialog


class NccWorkbenchTemplateRoiTest(unittest.TestCase):
    def test_saving_find_options_does_not_change_template_roi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = str(Path(tmp) / "model.json")
            model = NccMatchModel(
                template_roi=NccMatchRect(x=11, y=22, width=333, height=444),
            ).normalized()
            save_model(model_path, model)

            dialog = _dialog_for_options(model_path, model)
            dialog.spn_roi_x = _Value(900)
            dialog.spn_roi_y = _Value(901)
            dialog.spn_roi_w = _Value(902)
            dialog.spn_roi_h = _Value(903)

            NccMatchWorkbenchDialog._save_find_options_to_model(dialog)

            saved = load_model(model_path)
            self.assertEqual(saved.template_roi.to_xywh(), (11, 22, 333, 444))
            self.assertEqual(saved.options.target_num, 7)
            self.assertEqual(saved.options.score_threshold, 0.42)
            self.assertEqual(saved.pose_refinement, "saturation_rect")

    def test_apply_current_template_roi_saves_template(self) -> None:
        dialog = NccMatchWorkbenchDialog.__new__(NccMatchWorkbenchDialog)
        dialog.source_canvas = _Canvas((1, 2, 30, 40))
        calls: list[bool] = []
        dialog._save_template = lambda *args, **kwargs: calls.append(True)

        NccMatchWorkbenchDialog._apply_current_template_roi(dialog)

        self.assertEqual(calls, [True])

    def test_drawing_template_roi_auto_applies_current_box(self) -> None:
        dialog = NccMatchWorkbenchDialog.__new__(NccMatchWorkbenchDialog)
        dialog._syncing_roi = False
        dialog._loading_model = False
        dialog._suppress_source_roi_auto_apply = False
        dialog.source_canvas = _Canvas((5, 6, 70, 80))
        seen: list[tuple[int, int, int, int]] = []
        calls: list[bool] = []
        dialog._set_roi_spin_values = lambda roi: seen.append(roi)
        dialog._save_template = lambda *args, **kwargs: calls.append(True)

        NccMatchWorkbenchDialog._sync_roi_from_canvas(dialog)

        self.assertEqual(seen, [(5, 6, 70, 80)])
        self.assertEqual(calls, [True])

    def test_loading_template_roi_does_not_auto_apply_current_box(self) -> None:
        dialog = NccMatchWorkbenchDialog.__new__(NccMatchWorkbenchDialog)
        dialog._syncing_roi = False
        dialog._loading_model = True
        dialog._suppress_source_roi_auto_apply = False
        dialog.source_canvas = _Canvas((5, 6, 70, 80))
        seen: list[tuple[int, int, int, int]] = []
        calls: list[bool] = []
        dialog._set_roi_spin_values = lambda roi: seen.append(roi)
        dialog._save_template = lambda: calls.append(True)

        NccMatchWorkbenchDialog._sync_roi_from_canvas(dialog)

        self.assertEqual(seen, [(5, 6, 70, 80)])
        self.assertEqual(calls, [])

    def test_loading_source_image_without_roi_does_not_reset_template_roi_form(self) -> None:
        dialog = NccMatchWorkbenchDialog.__new__(NccMatchWorkbenchDialog)
        dialog._syncing_roi = False
        dialog._loading_model = True
        dialog._suppress_source_roi_auto_apply = False
        dialog.source_canvas = _Canvas(None)
        seen: list[tuple[int, int, int, int]] = []
        calls: list[bool] = []
        dialog._set_roi_spin_values = lambda roi: seen.append(roi)
        dialog._save_template = lambda: calls.append(True)

        NccMatchWorkbenchDialog._sync_roi_from_canvas(dialog)

        self.assertEqual(seen, [])
        self.assertEqual(calls, [])

    def test_translate_reference_move_selection_moves_all_selected_regions(self) -> None:
        dialog = NccMatchWorkbenchDialog.__new__(NccMatchWorkbenchDialog)
        dialog.ref_canvas = _MoveCanvas()
        dialog._reference_regions = [
            NccReferenceRegion(
                label_name="roi1",
                display_name="Hole",
                shape_type="rectangle",
                points=[(10.0, 20.0), (30.0, 40.0)],
            ).normalized(),
            NccReferenceRegion(
                label_name="roi2",
                display_name="Pusher",
                shape_type="rectangle",
                points=[(50.0, 60.0), (70.0, 80.0)],
            ).normalized(),
            NccReferenceRegion(
                label_name="roi3",
                display_name="Pusher",
                shape_type="rectangle",
                points=[(90.0, 100.0), (110.0, 120.0)],
            ).normalized(),
        ]
        dialog._reference_move_original = {
            0: [(10.0, 20.0), (30.0, 40.0)],
            2: [(90.0, 100.0), (110.0, 120.0)],
        }
        dialog._refresh_reference_region_list = lambda: None
        dialog._refresh_reference_region_fields = lambda: None
        dialog._refresh_reference_canvas = lambda: None

        NccMatchWorkbenchDialog._translate_reference_move_selection(dialog, 5.0, -3.0)

        self.assertEqual(dialog._reference_regions[0].points, [(15.0, 17.0), (35.0, 37.0)])
        self.assertEqual(dialog._reference_regions[1].points, [(50.0, 60.0), (70.0, 80.0)])
        self.assertEqual(dialog._reference_regions[2].points, [(95.0, 97.0), (115.0, 117.0)])


if __name__ == "__main__":
    unittest.main()
