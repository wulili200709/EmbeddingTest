"""UI caches must notice geometry/image edits, preserve pixels, and stay bounded."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from PySide6 import QtGui,QtWidgets
from common import labelme_io
from ui.debug import image_pixmap_cache
from ui.debug.tool_page.roi_geometry_cache import RoiGeometryCache,shape_geometry


class AnnotationReadCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.root=Path(self.directory.name)
        image_pixmap_cache._entries.clear()

    def tearDown(self):
        image_pixmap_cache._entries.clear()
        self.directory.cleanup()

    def write_geometry(self,name,label):
        path=self.root/f'{name}.json'
        staged=path.with_suffix('.new')
        staged.write_text(json.dumps({'shapes':[{'label':label,'shape_type':'rectangle','points':[[1,2],[8,9]]}]}),encoding='utf-8')
        os.replace(staged,path)
        return path.with_suffix('.png')

    def test_geometry_version_change_delete_and_recreate_are_not_stale(self):
        cache=RoiGeometryCache()
        image=self.write_geometry('sample','roi1')
        with patch('common.labelme_io.load_json_with_backup',wraps=labelme_io.load_json_with_backup) as read:
            self.assertIn('roi1',cache.shapes_for_path(str(image)))
            cache.shapes_for_path(str(image))
            self.assertEqual(read.call_count,1)
            self.write_geometry('sample','roi2')
            self.assertEqual(set(cache.shapes_for_path(str(image))),{'roi2'})
            self.assertEqual(read.call_count,2)
            image.with_suffix('.json').unlink()
            self.assertEqual(cache.shapes_for_path(str(image)),{})
            self.write_geometry('sample','roi3')
            self.assertEqual(set(cache.shapes_for_path(str(image))),{'roi3'})

    def test_backup_repair_is_cached_under_restored_file_version(self):
        image=self.write_geometry('sample','roi1')
        path=image.with_suffix('.json')
        path.with_name(path.name+'.bak').write_bytes(path.read_bytes())
        path.write_text('broken JSON',encoding='utf-8')
        cache=RoiGeometryCache()
        with patch('common.labelme_io.load_json_with_backup',wraps=labelme_io.load_json_with_backup) as read:
            self.assertIn('roi1',cache.shapes_for_path(str(image)))
            self.assertIn('roi1',cache.shapes_for_path(str(image)))
            self.assertEqual(read.call_count,1)
        self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['shapes'][0]['label'],'roi1')

    def test_geometry_cache_is_bounded_and_bad_shapes_do_not_crash_view(self):
        cache=RoiGeometryCache(max_entries=2)
        for n in range(3):cache.shapes_for_path(str(self.write_geometry(str(n),'roi1')))
        self.assertEqual(len(cache._entries),2)
        self.assertEqual(shape_geometry({'points':[['bad',1],[2,3]]}),(None,None))
        self.assertEqual(shape_geometry({'points':[[float('nan'),1],[2,3]]}),(None,None))

    def test_shared_pixmap_copy_cannot_modify_cache_and_replacement_is_visible(self):
        path=self.root/'image.png'
        original=QtGui.QPixmap(20,30)
        original.fill(QtGui.QColor('blue'))
        original.save(str(path))
        first=image_pixmap_cache.pixmap_from_path(str(path))
        second=image_pixmap_cache.pixmap_from_path(str(path))
        self.assertEqual(first.cacheKey(),second.cacheKey())
        first.fill(QtGui.QColor('red'))
        self.assertEqual(image_pixmap_cache.pixmap_from_path(str(path)).toImage().pixelColor(0,0),QtGui.QColor('blue'))
        replacement=QtGui.QPixmap(40,50)
        replacement.fill(QtGui.QColor('green'))
        staged=path.with_name('new.png')
        replacement.save(str(staged))
        os.replace(staged,path)
        changed=image_pixmap_cache.pixmap_from_path(str(path))
        self.assertEqual((changed.width(),changed.height()),(40,50))
        self.assertEqual(changed.toImage().pixelColor(0,0),QtGui.QColor('green'))

    def test_decoded_cache_has_count_and_memory_limits(self):
        for n in range(5):
            path=self.root/f'{n}.png'
            pixmap=QtGui.QPixmap(20,30)
            pixmap.fill(QtGui.QColor('blue'))
            pixmap.save(str(path))
            image_pixmap_cache.pixmap_from_path(str(path))
        self.assertLessEqual(len(image_pixmap_cache._entries),3)
        image_pixmap_cache._entries.clear()
        with patch.object(image_pixmap_cache,'_MAX_BYTES',100):
            image_pixmap_cache.pixmap_from_path(str(path))
            self.assertEqual(len(image_pixmap_cache._entries),0)

    def test_annotation_progress_uses_current_labels_and_changed_geometry(self):
        from ui.debug.tool_page.roi_annotation_controller import RoiAnnotationController
        from ui.i18n import tr
        image=self.write_geometry('sample','roi1')
        path=image.with_suffix('.json')
        data=json.loads(path.read_text(encoding='utf-8'))
        data['shapes'].append(dict(data['shapes'][0],label='roi2'))
        path.write_text(json.dumps(data),encoding='utf-8')
        owner=SimpleNamespace(
            current_camera_role=lambda:'cam1',
            _inspection_label_names_for_role=lambda *_:['roi1','roi2'],
            inspection_items=[SimpleNamespace(roi_label=label,camera_id='cam1',enabled=True,algorithm_code='shared_backbone_register') for label in ('roi1','roi2')],
            _sample_roi_annotations_by_path={str(image):{'cam1::roi1':'OK'}},
        )
        controller=RoiAnnotationController(owner)
        self.assertEqual(controller.progress_for_path(str(image),'cam1'),(1,2))
        self.assertEqual(controller.state_for_path(str(image),'cam1'),tr('sample.partial'))
        owner._sample_roi_annotations_by_path[str(image)]['cam1::roi2']='NG'
        self.assertEqual(controller.state_for_path(str(image),'cam1'),tr('sample.complete'))
        self.write_geometry('sample','roi2')
        self.assertFalse(controller.has_geometry(str(image),'roi1'))
        self.assertEqual(controller.state_for_path(str(image),'cam1'),tr('sample.missing_roi'))


if __name__=='__main__':unittest.main()
