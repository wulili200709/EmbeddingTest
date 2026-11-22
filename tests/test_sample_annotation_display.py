"""Viewing a crowded sample must not label invisible ROIs or change annotations."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from ui.debug.tool_page.sample_annotation_canvas import _SampleAnnotationCanvas
from ui.debug.tool_page.sample_annotation_dialog import _SampleAnnotationPreviewDialog


class AnnotationOwner(QtWidgets.QWidget):
    roiGeometryChanged = QtCore.Signal()
    inspectionItemsChanged = QtCore.Signal()

    def __init__(self, paths):
        super().__init__()
        self.paths = [str(p) for p in paths]
        shapes = json.loads(Path(paths[0]).with_suffix(".json").read_text(encoding="utf-8"))["shapes"]
        self.labels = [s["label"] for s in shapes]
        self.inspection_items = [SimpleNamespace(roi_label=l, display_name=l, camera_id="cam1") for l in self.labels]
        self.selected = self.inspection_items[-1]
        self.edits = []

    def current_product_name(self): return "Preview"
    def configured_camera_roles(self): return ["cam1"]
    def current_camera_role(self): return "cam1"
    def _current_sample_tab_kind(self): return "train"
    def _current_selected_path(self): return self.paths[0]
    def _selected_inspection_item(self): return self.selected
    def _sample_paths_for_kind(self, *_): return self.paths
    def _sample_item_display_text(self, path, *_): return Path(path).name
    def _sample_usage_text(self, *_): return "Train"
    def _sample_annotation_state_for_path(self, *_): return "Complete"
    def _inspection_label_names_for_role(self, *_): return self.labels
    def _path_has_roi_geometry(self, *_): return True
    def _sample_roi_status_for_path(self, *_): return "OK"
    def _set_sample_roi_status_for_path(self, *args): self.edits.append(args)
    def _refresh_lists(self): pass
    def _update_sample_panel_widgets(self): pass
    def _set_current_camera_role(self, *_args, **_kwargs): pass
    def _select_path_in_current_tab(self, *_): pass


class SampleAnnotationDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        pixmap = QtGui.QPixmap(1000, 800)
        pixmap.fill(QtGui.QColor("#555555"))
        shapes = [
            {"label": "roi1", "shape_type": "rectangle", "points": [[10,10],[300,300]]},
            {"label": "roi2", "shape_type": "polygon", "points": [[50,50],[100,50],[100,100],[50,100]]},
            {"label": "roi33", "shape_type": "rectangle", "points": [[300,380],[700,420]]},
        ]
        paths=[]
        for i in range(2):
            path=root/f"image{i}.png"
            pixmap.save(str(path))
            path.with_suffix(".json").write_text(json.dumps({"shapes":shapes}),encoding="utf-8")
            paths.append(path)
        self.owner=AnnotationOwner(paths)
        self.dialog=_SampleAnnotationPreviewDialog(self.owner)
        self.dialog.show()
        self.app.processEvents()

    def tearDown(self):
        self.assertEqual(self.owner.edits, [])
        self.dialog.close()
        self.owner.close()
        self.dialog.deleteLater()
        self.owner.deleteLater()
        self.app.processEvents()
        self.directory.cleanup()

    def test_default_all_and_current_only_does_not_hit_hidden_overlapping_roi(self):
        self.assertEqual(self.dialog._active_roi_label,"roi33")
        self.assertEqual(self.dialog.cmb_overlay_mode.currentData(),"all")
        self.assertEqual(len(self.dialog.preview_canvas._overlays),3)
        self.dialog.cmb_overlay_mode.setCurrentIndex(self.dialog.cmb_overlay_mode.findData("current"))
        self.assertEqual(len(self.dialog.preview_canvas._overlays),1)
        self.dialog.cmb_active_roi.setCurrentIndex(self.dialog.cmb_active_roi.findData("roi1"))
        self.assertEqual(self.dialog._find_roi_label_at_point(60,60),"roi1")
        self.dialog.cmb_overlay_mode.setCurrentIndex(self.dialog.cmb_overlay_mode.findData("all"))
        self.assertEqual(len(self.dialog.preview_canvas._overlays),3)
        self.assertEqual(self.dialog._find_roi_label_at_point(60,60),"roi2")
        self.dialog.cmb_overlay_mode.setCurrentIndex(self.dialog.cmb_overlay_mode.findData("hidden"))
        self.assertEqual(self.dialog.preview_canvas._overlays,[])
        self.assertEqual(self.dialog._find_roi_label_at_point(60,60),"")

    def test_current_item_survives_next_image_and_table_selection_stays_in_sync(self):
        self.dialog.cmb_overlay_mode.setCurrentIndex(self.dialog.cmb_overlay_mode.findData("current"))
        self.dialog.roi_table.setCurrentCell(1,0)
        self.assertEqual(self.dialog.cmb_active_roi.currentData(),"roi2")
        self.dialog._step_selection(1)
        self.assertEqual(self.dialog.cmb_active_roi.currentData(),"roi2")
        self.assertEqual(self.dialog._active_roi_label,"roi2")
        self.assertEqual(len(self.dialog.preview_canvas._overlays),1)

    def test_space_hides_paint_and_hit_testing_then_restores_without_losing_boxes(self):
        canvas=self.dialog.preview_canvas
        overlays=list(canvas._overlays)
        QtTest.QTest.keyPress(canvas,QtCore.Qt.Key.Key_Space)
        self.assertTrue(canvas.is_original_preview())
        self.assertEqual(self.dialog._find_roi_label_at_point(400,400),"")
        QtTest.QTest.keyRelease(canvas,QtCore.Qt.Key.Key_Space)
        self.assertFalse(canvas.is_original_preview())
        self.assertEqual(canvas._overlays,overlays)
        self.assertEqual(self.dialog._find_roi_label_at_point(400,400),"roi33")
        QtTest.QTest.keyPress(canvas,QtCore.Qt.Key.Key_Space)
        QtWidgets.QApplication.sendEvent(canvas,QtGui.QFocusEvent(QtCore.QEvent.Type.FocusOut))
        self.assertFalse(canvas.is_original_preview())

    def test_focus_button_zooms_to_current_roi_without_changing_the_sample(self):
        canvas=self.dialog.preview_canvas
        self.dialog._focus_current_roi()
        self.assertGreater(canvas._zoom,1.)
        center=QtCore.QPointF(500*canvas._scale,400*canvas._scale)+canvas._image_offset
        self.assertAlmostEqual(center.x(),canvas.width()/2,delta=2)
        self.assertAlmostEqual(center.y(),canvas.height()/2,delta=2)

    def test_switch_reads_geometry_once_and_reuses_widgets_for_current_image_labels(self):
        from common import labelme_io
        old_combo=self.dialog.roi_table.cellWidget(0,2)
        with patch('common.labelme_io.load_json_with_backup',wraps=labelme_io.load_json_with_backup) as reads, \
             patch.object(self.owner,'_set_current_camera_role') as role_switch:
            self.dialog._step_selection(1)
            self.assertEqual(reads.call_count,1)
            self.dialog.cmb_overlay_mode.setCurrentIndex(self.dialog.cmb_overlay_mode.findData('current'))
            self.assertEqual(reads.call_count,1)
            role_switch.assert_not_called()
        self.assertIs(self.dialog.roi_table.cellWidget(0,2),old_combo)
        with patch.object(self.owner,'_set_sample_roi_status_for_path') as save:
            old_combo.setCurrentIndex(old_combo.findData('NG'))
            save.assert_called_once_with(self.owner.paths[1],'cam1','roi1','NG')

    def test_reloading_list_loads_selected_image_once(self):
        with patch.object(self.dialog,'_load_canvas_preview',wraps=self.dialog._load_canvas_preview) as load:
            self.dialog._reload_samples()
            self.assertEqual(load.call_count,1)

    def test_external_geometry_changes_are_visible_without_reopening_dialog(self):
        path=Path(self.owner.paths[0]).with_suffix('.json')
        data=json.loads(path.read_text(encoding='utf-8'))
        data['shapes'][-1]['points']=[[310,380],[710,420]]
        replacement=path.with_suffix('.new')
        replacement.write_text(json.dumps(data),encoding='utf-8')
        os.replace(replacement,path)
        self.dialog._refresh_display()
        entry=next(s for s in self.dialog._canvas_shapes if s['label']=='roi33')
        self.assertEqual(entry['xywh'],(310,380,400,40))
        data['shapes']=data['shapes'][:-1]
        replacement.write_text(json.dumps(data),encoding='utf-8')
        os.replace(replacement,path)
        self.dialog._on_sample_selected()
        self.assertFalse(self.dialog.roi_table.cellWidget(2,2).isEnabled())
        self.assertFalse(any(s['label']=='roi33' for s in self.dialog._canvas_shapes))

    def test_original_preview_click_does_not_emit_annotation_even_if_space_released_mid_click(self):
        canvas=_SampleAnnotationCanvas()
        canvas.resize(500,400)
        pixmap=QtGui.QPixmap(1000,800)
        pixmap.fill(QtGui.QColor("white"))
        canvas.set_image(pixmap)
        canvas.show()
        self.app.processEvents()
        spy=QtTest.QSignalSpy(canvas.imagePressed)
        QtTest.QTest.keyPress(canvas,QtCore.Qt.Key.Key_Space)
        QtTest.QTest.mousePress(canvas,QtCore.Qt.MouseButton.LeftButton,pos=QtCore.QPoint(200,200))
        QtTest.QTest.keyRelease(canvas,QtCore.Qt.Key.Key_Space)
        QtTest.QTest.mouseRelease(canvas,QtCore.Qt.MouseButton.LeftButton,pos=QtCore.QPoint(200,200))
        self.assertEqual(spy.count(),0)
        QtTest.QTest.mouseClick(canvas,QtCore.Qt.MouseButton.LeftButton,pos=QtCore.QPoint(200,200))
        self.assertEqual(spy.count(),1)
        canvas.close()


if __name__=="__main__":unittest.main()
