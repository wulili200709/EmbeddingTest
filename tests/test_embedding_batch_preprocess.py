"""Fast ORT inputs must preserve the existing trained-model preprocessing."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from algorithms import embedding


class RecordingOrt:
    runtime_backend='ort'

    def run(self,batch,**kwargs):
        self.batch=np.asarray(batch)
        return self.batch.mean(axis=(2,3))


class EmbeddingBatchPreprocessTests(unittest.TestCase):
    def test_camera_rgb_conversion_preserves_pixels_for_strided_color_and_gray(self):
        rng=np.random.default_rng(12)
        image=rng.integers(0,256,(91,137,4),dtype=np.uint8)
        for source in (image[:,:,:3],image[::2,::-1,:3],image,image[:,:,0]):
            if source.ndim==2:
                original=Image.fromarray(np.ascontiguousarray(source)).convert('RGB')
            else:
                original=Image.fromarray(np.ascontiguousarray(source[:,:,:3][:,:,::-1])).convert('RGB')
            np.testing.assert_array_equal(np.asarray(embedding._image_from_bgr_array(source)),np.asarray(original))

    def test_fast_rgb_inputs_match_original_resize_and_normalization_exactly(self):
        rng=np.random.default_rng(9)
        images=[Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8)) for h,w in ((9,17),(224,224),(84,365),(81,472),(399,87))]
        original=torch.stack([embedding.TF(image) for image in images]).numpy()
        fast=embedding._preprocess_roi_batch(images,RecordingOrt())
        np.testing.assert_array_equal(fast,original)
        self.assertEqual(fast.dtype,np.float32)
        self.assertTrue(fast.flags.c_contiguous)
        self.assertIsInstance(embedding._preprocess_roi_batch(images,SimpleNamespace(runtime_backend='torch')),torch.Tensor)

    def test_file_and_memory_paths_keep_polygon_masks_clipping_and_label_order(self):
        rng=np.random.default_rng(3)
        image=rng.integers(0,256,(130,180,3),dtype=np.uint8)
        shapes={
            'roi1':{'shape_type':'rectangle','points':[[2,7],[61,48]]},
            'roi2':{'shape_type':'polygon','points':[[90,90],[70,15],[170,25]]},
            'roi3':{'shape_type':'polygon','points':[[-4,70],[50,70],[50,125],[-4,125]]},
        }
        labels=['roi3','roi1','roi2']
        context=embedding._build_array_roi_context(image,shape_by_label=shapes)
        expected=torch.stack([embedding.TF(embedding._resolve_roi_image(context,label_name=l)) for l in labels]).numpy()
        runner=RecordingOrt()
        embedding.embed_batch_from_array(image,runner,labels,shape_by_label=shapes,device='cpu')
        np.testing.assert_array_equal(runner.batch,expected)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'image.png'
            Image.fromarray(image[:,:,::-1]).save(path)
            path.with_suffix('.json').write_text(json.dumps({'shapes':[dict(s,label=l) for l,s in shapes.items()]}),encoding='utf-8')
            embedding.embed_batch(str(path),runner,labels,device='cpu')
            np.testing.assert_array_equal(runner.batch,expected)

    def test_ort_cpu_threads_are_applied_without_changing_model_or_chunk(self):
        session=SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='input',shape=['batch',3,224,224])],
                                get_outputs=lambda:[SimpleNamespace(name='embedding')])
        with patch.object(embedding,'_ensure_ort_model',return_value=Path('existing.ort')), \
             patch.object(embedding,'_ort_providers_for_device',return_value=('CPUExecutionProvider',)), \
             patch.object(embedding,'_configured_cpu_inference_threads',return_value=4), \
             patch.object(embedding,'_configured_cpu_inference_chunk_size',return_value=2), \
             patch.object(embedding.ort,'InferenceSession',return_value=session) as create:
            runner=embedding._load_ort_backbone('b0',device='cpu')
        options=create.call_args.kwargs['sess_options']
        self.assertEqual(options.intra_op_num_threads,4)
        self.assertEqual(options.inter_op_num_threads,1)
        self.assertEqual(options.get_session_config_entry('session.intra_op.allow_spinning'),'0')
        self.assertEqual(runner.cpu_inference_threads,4)
        self.assertEqual(runner.cpu_inference_chunk_size,2)

    def test_cpu_thread_setting_survives_settings_load_save_and_environment_override(self):
        from ui.shell.dialogs import RuntimeModeSettingsStore
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'runtime_mode_settings.json'
            store=RuntimeModeSettingsStore()
            with patch.object(store,'path',return_value=path):
                settings=store.default_settings()
                settings.update(cpu_inference_threads=4,cpu_inference_chunk_size=2)
                store.save(settings)
                loaded=store.load()
                self.assertEqual(loaded['cpu_inference_threads'],4)
                self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['cpu_inference_threads'],4)
            with patch.object(embedding,'writable_embedding_test_root',return_value=Path(folder)), \
                 patch.dict(os.environ,{'LC_CPU_INFERENCE_THREADS':'2'}):
                self.assertEqual(embedding._configured_cpu_inference_threads(),2)
        for raw,expected in ((-2,0),('bad',0),(500,64),(4,4)):
            self.assertEqual(RuntimeModeSettingsStore._normalize_cpu_inference_threads(raw),expected)


if __name__=='__main__':unittest.main()
