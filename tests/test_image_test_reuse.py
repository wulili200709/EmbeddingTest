from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from application.algorithm_controller import AlgorithmController
from shape.core import template_core
from shape.like_matcher import ShapeLikeDetector, encode_png_base64


class ImageTestReuseTests(unittest.TestCase):
    def test_repeated_image_tests_reuse_models_but_read_new_parameters_and_files(self):
        controller = AlgorithmController()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            params = root / 'product_params.json'
            params.write_text(json.dumps({'algorithm': 'b0', 'margin': .02}), encoding='utf-8')
            model_path = root / 'cam1__roi1_register_model_b0.npz'
            model_path.write_bytes(b'first')
            controller.load_params(str(params))
            with patch('application.algorithm_controller.qr_core.load_register_model_npz',
                       side_effect=lambda _path: SimpleNamespace(backbone='efficientnet_b0')) as load:
                first, _ = controller.load_model_for_algorithm('b0', folder, model_key='cam1__roi1')
                params.write_text(json.dumps({'algorithm': 'b0', 'margin': .08}), encoding='utf-8')
                controller.load_params(str(params), reuse_embedding_models=True)
                second, _ = controller.load_model_for_algorithm('b0', folder, model_key='cam1__roi1')
                self.assertIs(first, second)
                self.assertEqual(second.margin, .08)
                self.assertEqual(load.call_count, 1)

                model_path.write_bytes(b'newly-trained-model')
                controller.load_params(str(params), reuse_embedding_models=True)
                third, _ = controller.load_model_for_algorithm('b0', folder, model_key='cam1__roi1')
                self.assertIsNot(second, third)
                self.assertEqual(load.call_count, 2)

                controller.load_params(str(params))
                fourth, _ = controller.load_model_for_algorithm('b0', folder, model_key='cam1__roi1')
                self.assertIsNot(third, fourth)
                self.assertEqual(load.call_count, 3)

            other = root / 'other-product' / 'product_params.json'
            other.parent.mkdir()
            other.write_text('{}', encoding='utf-8')
            self.assertTrue(controller._embedding_model_cache)
            controller.load_params(str(other), reuse_embedding_models=True)
            self.assertFalse(controller._embedding_model_cache)

    def test_template_geometry_reuses_decoding_and_refreshes_after_source_edits(self):
        detector = ShapeLikeDetector()
        source = {'source': {
            'roi_png': encode_png_base64(np.full((17, 31, 3), 42, np.uint8)),
            'mask_png': encode_png_base64(np.full((17, 31), 255, np.uint8)),
            'roi_x': 5, 'roi_y': 8, 'roi_w': 31, 'roi_h': 17,
        }}
        detector.set_class_source('part', source)
        with patch.object(template_core, 'load_class_source_assets',
                          wraps=template_core.load_class_source_assets) as decode:
            first = template_core.load_class_source_geometry(detector, 'part')
            second = template_core.load_class_source_geometry(detector, 'part')
            self.assertEqual(first, second)
            self.assertEqual(decode.call_count, 1)
            # Returned rectangles cannot change the cached geometry.
            first[2].x = 1000
            self.assertEqual(template_core.load_class_source_geometry(detector, 'part')[2].x, 5)

            detector.class_sources['part']['source']['roi_x'] = 9
            self.assertEqual(template_core.load_class_source_geometry(detector, 'part')[2].x, 9)
            self.assertEqual(decode.call_count, 2)
            # The source's decoded dimensions remain authoritative, including
            # legacy models whose rectangle dimensions differ from the image.
            source['source']['roi_png'] = encode_png_base64(np.full((19, 33, 3), 5, np.uint8))
            detector.set_class_source('part', source)
            third = template_core.load_class_source_geometry(detector, 'part')
            self.assertEqual(third[:2], (33, 19))
            self.assertEqual((third[2].w, third[2].h), (31, 17))
            self.assertEqual(decode.call_count, 3)

    def test_invalid_template_source_is_rejected_after_a_cached_success(self):
        detector = ShapeLikeDetector()
        source = {'source': {
            'roi_png': encode_png_base64(np.zeros((5, 7, 3), np.uint8)),
            'mask_png': encode_png_base64(np.ones((5, 7), np.uint8)),
        }}
        detector.set_class_source('part', source)
        template_core.load_class_source_geometry(detector, 'part')
        detector.class_sources['part']['source']['mask_png'] = ''
        with self.assertRaisesRegex(ValueError, 'missing embedded ROI image or mask'):
            template_core.load_class_source_geometry(detector, 'part')


if __name__ == '__main__':
    unittest.main()
