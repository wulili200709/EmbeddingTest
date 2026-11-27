from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from algorithms.image_io import imread, imwrite
from services import lens_correction as service


def pinhole(k1=-0.18):
    return {"schema": "lc-lens-calibration-v1", "image_size": [96, 72],
            "camera_matrix": [[100., 0., 47.5], [0., 99., 35.5], [0., 0., 1.]],
            "dist_coeffs": [k1, 0.03, 0.001, -0.002, 0.0], "pattern": "chessboard",
            "rms": 0.2, "views": [], "warnings": []}


def single(mode="radial"):
    grid = np.mgrid[0:5, 0:4].T.reshape(-1, 2).astype(float)
    grid = (grid - [2, 1.5]) / 4
    matrix = np.array([[0.65, 0.03, 0.0], [0.01, 0.6, 0.0], [0.03, -0.02, 1.]])
    projected = np.column_stack((grid, np.ones(len(grid)))) @ matrix.T
    q = projected[:, :2] / projected[:, 2:]
    radius2 = (q * q).sum(axis=1, keepdims=True)
    points = [47.5, 35.5] + 96 * q * (1 - 0.15 * radius2 + 0.02 * radius2 ** 2)
    return {"schema": "lc-lens-single-v1", "image_size": [96, 72], "mode": mode,
            "center": [47.5, 35.5], "radial_coeffs": [-0.15, 0.02], "homography": matrix.tolist(),
            "pattern_size": [5, 4], "spacing_mm": 2., "rms": 0., "points": points.tolist(),
            "views": [], "warnings": [], "coverage": {}}


class LensCorrectionServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "产品"
        self.root.mkdir()
        self.role_patch = patch("services.lens_camera_roles.resolve_correction_role", side_effect=lambda _root, role: role)
        self.role_patch.start()
        self.addCleanup(self.role_patch.stop)
        self.profile = self.write_json("原始标定.json", pinhole())
        self.image = np.random.default_rng(77).integers(0, 256, (72, 96, 3), np.uint8)
        self.source = self.root / "Cam1_原图.bmp"
        self.assertTrue(imwrite(self.source, self.image))
        service._MAP_CACHE.clear()
        service._VALIDATED_CACHE.clear()

    def write_json(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def enable(self, **kwargs):
        return service.save_config(self.root, "cam1", enabled=True, profile_path=self.profile, **kwargs)

    def test_disabled_is_identity_and_does_not_load_calibration(self):
        with patch.object(service, "_profile", side_effect=AssertionError("loaded profile")), \
             patch.object(service, "_cached", side_effect=AssertionError("built map")):
            self.assertFalse(service.get_config(self.root, "cam1").enabled)
            result = service.prepare_array(self.image, self.root, "cam1")
            self.assertIs(result.image, self.image)
            self.assertFalse(result.corrected)
            self.assertEqual(result.profile_id, "raw")
            self.assertEqual(service.prepare_path("missing.bmp", self.root, "cam1"), "missing.bmp")
            service.warmup(self.root, "cam1")
            self.assertEqual(service.asset_root(self.root, "cam1"), str(self.root))
        script = """
import sys, tempfile
from services import lens_correction as s
assert 'scipy' not in sys.modules
assert 'algorithms.lens_single_calibration' not in sys.modules
with tempfile.TemporaryDirectory() as d:
    token = object()
    assert s.prepare_array(token, d, 'cam1').image is token
    assert s.prepare_path('missing.bmp', d, 'cam1') == 'missing.bmp'
    s.warmup(d, 'cam1')
assert 'scipy' not in sys.modules
assert 'algorithms.lens_calibration' not in sys.modules
"""
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=40)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_explicit_import_and_per_camera_isolation(self):
        with self.assertRaises(service.LensCorrectionError):
            service.save_config(self.root, "cam1", enabled=True)
        first = self.enable(camera_serial="SERIAL-A", alpha=0.6)
        second_path = self.write_json("第二相机.json", pinhole(0.15))
        second = service.save_config(self.root, "cam2", enabled=True, profile_path=second_path, camera_serial="SERIAL-B")
        self.assertNotEqual(first.profile_id, second.profile_id)
        self.assertEqual(service.get_config(self.root, "cam1"), first)
        self.assertNotEqual(service.asset_root(self.root, "cam1"), service.asset_root(self.root, "cam2"))
        self.assertTrue(Path(first.profile_path).is_file())
        self.assertIn("calibration", Path(first.profile_path).parts)
        self.profile.write_text(json.dumps(pinhole(0.4)), encoding="utf-8")
        self.assertEqual(service.active_signature(self.root, "cam1"), first.profile_id)
        result = service.prepare_array(self.image, self.root, "cam1", camera_serial="SERIAL-A")
        with self.assertRaises(service.LensCorrectionError):
            service.prepare_array(self.image, self.root, "cam1", camera_serial="SERIAL-B")
        from algorithms.lens_calibration import CalibrationProfile, Undistorter
        expected = Undistorter(CalibrationProfile.from_dict(pinhole()), alpha=0.6).apply(self.image)
        np.testing.assert_array_equal(result.image, expected)

    def test_logical_channels_share_physical_calibration_but_not_assets(self):
        first = self.enable(camera_serial="SERIAL-A")
        with patch("services.lens_camera_roles.resolve_correction_role", return_value="cam1"):
            self.assertEqual(service.get_config(self.root, "cam2"), first)
            self.assertFalse(service.get_config(self.root, "cam2", physical=True).enabled)
            self.assertNotEqual(service.asset_root(self.root, "cam1"), service.asset_root(self.root, "cam2"))
            self.assertEqual(service.active_signature(self.root, "cam2"), first.profile_id)

    def test_map_cache_fresh_results_and_exact_dtype(self):
        self.enable()
        original_make = service._make_rectifier
        with patch.object(service, "_make_rectifier", wraps=original_make) as build:
            service.warmup(self.root, "cam1")
            first = service.prepare_array(self.image, self.root, "cam1")
            second = service.prepare_array(self.image, self.root, "cam1")
            self.assertEqual(build.call_count, 1)
            self.assertFalse(np.shares_memory(first.image, second.image))
            first.image[:] = 0
            self.assertTrue(np.any(second.image))
            for dtype in (np.uint16, np.float32, np.float64):
                result = service.prepare_array(self.image[:, :, 0].astype(dtype), self.root, "cam1")
                self.assertEqual(result.image.dtype, dtype)
                self.assertEqual(result.image.shape, (72, 96))
        with self.assertRaises(service.LensCorrectionError):
            service.prepare_array(np.zeros((73, 96), np.uint8), self.root, "cam1")

    def test_concurrent_first_use_shares_maps_but_not_output_arrays(self):
        self.enable()
        with patch.object(service, "_make_rectifier", wraps=service._make_rectifier) as build:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda _: service.prepare_array(self.image, self.root, "cam1"), range(8)))
            self.assertEqual(build.call_count, 1)
        for result in results[1:]:
            np.testing.assert_array_equal(result.image, results[0].image)
            self.assertFalse(np.shares_memory(result.image, results[0].image))

    def test_geometry_fingerprint_ignores_reports_but_tracks_alpha(self):
        first = service.inspect_profile(self.profile, alpha=0.7)
        changed = pinhole()
        changed.update(rms=1.3, views=[{"path": "other", "status": "used"}], warnings=["different report"])
        other = self.write_json("different-report.json", changed)
        self.assertEqual(first.profile_id, service.inspect_profile(other, alpha=0.7).profile_id)
        self.assertNotEqual(first.profile_id, service.inspect_profile(other, alpha=1).profile_id)

    def test_batch_validation_is_atomic_and_disabling_preserves_version(self):
        config = self.enable(alpha=0.6)
        before = (self.root / "lens_correction_settings.json").read_bytes()
        invalid = self.write_json("bad.json", {"schema": "unsupported"})
        with self.assertRaises(service.LensCorrectionError):
            service.save_configs(self.root, {
                "cam1": {"enabled": True, "profile_path": self.profile},
                "cam2": {"enabled": True, "profile_path": invalid},
            })
        self.assertEqual((self.root / "lens_correction_settings.json").read_bytes(), before)
        with patch.object(service, "_profile", side_effect=AssertionError("disabled loaded profile")):
            disabled = service.save_config(self.root, "cam1", enabled=False)
        self.assertFalse(disabled.enabled)
        self.assertEqual(disabled.profile_id, config.profile_id)
        self.assertEqual(disabled.alpha, config.alpha)
        self.assertEqual(disabled.profile_path, config.profile_path)

    def test_snapshot_tampering_is_not_implicit_new_version(self):
        config = self.enable()
        service.warmup(self.root, "cam1")
        path = Path(config.profile_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["dist_coeffs"][0] += 0.1
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(service.LensCorrectionError):
            service.prepare_array(self.image, self.root, "cam1")

    def test_path_cache_and_annotation_coordinate_isolation(self):
        config = self.enable(camera_serial="SERIAL-A")
        raw_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.source.with_suffix(".json").write_text('{"shapes": []}', encoding="utf-8")
        corrected = Path(service.prepare_path(self.source, self.root, "cam1"))
        self.assertTrue(corrected.name.startswith("cam1_"))
        self.assertIn(config.profile_id, corrected.parts)
        self.assertFalse(corrected.with_suffix(".json").exists())
        timestamp = corrected.stat().st_mtime_ns
        self.assertEqual(service.prepare_path(self.source, self.root, "cam1"), str(corrected))
        self.assertEqual(corrected.stat().st_mtime_ns, timestamp)
        self.assertEqual(service.prepare_path(corrected, self.root, "cam1"), str(corrected))
        self.assertEqual(service.source_image_path(corrected), str(self.source.resolve()))
        marker = json.loads(corrected.with_suffix(".undistortion.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["camera_role"], "cam1")
        self.assertEqual(marker["camera_serial"], "SERIAL-A")
        self.assertEqual(marker["profile_id"], config.profile_id)
        self.assertIn("camera_matrix", marker["profile"])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), raw_hash)
        result = service.prepare_array(imread(corrected), self.root, "cam1", source_path=str(corrected))
        np.testing.assert_array_equal(result.image, imread(corrected))
        self.assertEqual(result.correction_ms, 0)
        service.validate_prepared_path(corrected, self.root, "cam1")
        with self.assertRaises(service.LensCorrectionError):
            service.validate_prepared_path(self.source, self.root, "cam1")
        service.save_config(self.root, "cam1", enabled=False)
        self.assertEqual(service.prepare_path(corrected, self.root, "cam1"), str(corrected))
        with self.assertRaises(service.LensCorrectionError):
            service.validate_prepared_path(corrected, self.root, "cam1")

    def test_cache_names_keep_long_product_root_and_sidecar_below_legacy_windows_limit(self):
        # A product under the actual desktop project is longer than tempfile's
        # root. Exercise the sidecar too, since its suffix exceeds '.png'.
        pad = max(1, 112 - len(str(self.root.resolve())) - 1)
        product = self.root / ("p" * pad)
        product.mkdir()
        long_source = product / (("very_long_camera_image_name_" * 4) + ".bmp")
        self.assertTrue(imwrite(long_source, self.image))
        lens_config = service.save_config(product, "cam1", enabled=True, profile_path=self.profile)
        corrected = Path(service.prepare_path(long_source, product, "cam1"))
        marker = corrected.with_suffix(".undistortion.json")
        self.assertLessEqual(len(str(marker.resolve())), 245)
        self.assertTrue(marker.is_file())
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual(len(metadata["source_fingerprint"]), 64)
        self.assertIn(lens_config.profile_id, corrected.parts)
        self.assertEqual(service.source_image_path(corrected), str(long_source.resolve()))

    def test_legacy_demo_marker_matches_geometry_not_claimed_identifier(self):
        config = self.enable(alpha=0.6)
        exported = self.root / "demo.png"
        image = service.prepare_array(self.image, self.root, "cam1").image
        self.assertTrue(imwrite(exported, image))
        marker = {"already_undistorted": True, "source_image": str(self.source),
                  "profile": pinhole(), "alpha": 0.6}
        sidecar = exported.with_suffix(".undistortion.json")
        sidecar.write_text(json.dumps(marker), encoding="utf-8")
        self.assertEqual(service.prepare_path(exported, self.root, "cam1"), str(exported))
        self.assertEqual(service.source_image_path(exported), str(self.source))
        marker["profile_id"] = config.profile_id
        marker["profile"]["dist_coeffs"][0] = 0.4
        sidecar.write_text(json.dumps(marker), encoding="utf-8")
        with self.assertRaises(service.LensCorrectionError):
            service.prepare_path(exported, self.root, "cam1")
        self.assertFalse(service.is_managed_image(exported))
        service.save_config(self.root, "cam1", enabled=False)
        service.validate_prepared_path(exported, self.root, "cam1")

    def test_old_export_version_stays_accurate_after_configuration_changes(self):
        old = self.enable(alpha=0.6)
        old_image = service.prepare_array(self.image, self.root, "cam1").image
        exported = self.root / "historical.png"
        self.assertTrue(imwrite(exported, old_image))
        new = self.enable(alpha=1.0)
        service.mark_corrected_path(exported, self.root, "cam1", profile_id=old.profile_id, source_path=str(self.source))
        marker = json.loads(exported.with_suffix(".undistortion.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["profile_id"], old.profile_id)
        self.assertEqual(marker["alpha"], 0.6)
        self.assertNotEqual(marker["profile_id"], new.profile_id)
        with self.assertRaises(service.LensCorrectionError):
            service.prepare_path(exported, self.root, "cam1")

    def test_reference_validation_cache_refreshes_when_image_or_marker_changes(self):
        self.enable()
        path = Path(service.prepare_path(self.source, self.root, "cam1"))
        with patch.object(service, "_read_image", wraps=service._read_image) as decode:
            service.validate_prepared_path(path, self.root, "cam1")
            service.validate_prepared_path(path, self.root, "cam1")
            self.assertEqual(decode.call_count, 1)
            imwrite(path, np.zeros((71, 96, 3), np.uint8))
            with self.assertRaises(service.LensCorrectionError):
                service.validate_prepared_path(path, self.root, "cam1")
            self.assertEqual(decode.call_count, 2)
        marker = path.with_suffix(".undistortion.json")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["alpha"] = 0.5
        marker.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(service.LensCorrectionError):
            service.validate_prepared_path(path, self.root, "cam1")

    def test_provenance_missing_source_and_cycles(self):
        one = self.root / "one.png"
        two = self.root / "two.png"
        imwrite(one, self.image)
        imwrite(two, self.image)
        one.with_suffix(".undistortion.json").write_text(json.dumps({"already_undistorted": True, "source_path": str(two)}))
        two.with_suffix(".undistortion.json").write_text(json.dumps({"already_undistorted": True, "source_path": str(one)}))
        with self.assertRaises(service.LensCorrectionError):
            service.source_image_path(one)
        two.unlink()
        self.assertEqual(service.source_image_path(one), str(one))

    def test_single_modes_use_existing_float_map_unchanged(self):
        from algorithms.lens_single_calibration import SingleCalibrationProfile, SingleUndistorter
        for mode in ("radial", "radial_perspective"):
            with self.subTest(mode=mode):
                payload = single(mode)
                path = self.write_json(f"{mode}.json", payload)
                config = service.save_config(self.root, "cam1", enabled=True, profile_path=path)
                self.assertEqual(config.mode, mode)
                result = service.prepare_array(self.image, self.root, "cam1")
                expected = SingleUndistorter(SingleCalibrationProfile.from_dict(payload)).apply(self.image)
                np.testing.assert_array_equal(result.image, expected)
                with self.assertRaises(service.LensCorrectionError):
                    service.inspect_profile(path, alpha=0.5)


if __name__ == "__main__":
    unittest.main()
