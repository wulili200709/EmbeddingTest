from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import common.model_library as model_library
from common.model_library import ModelLibrary, make_library_model_key, parse_library_model_key


class ModelLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.npz"
        self.source.write_bytes(b"immutable model sample")
        self.library = ModelLibrary(self.root / "library")

    def publish(self, **kwargs):
        return self.library.publish("针检测", "efficientnet_b0", model_file=self.source, **kwargs)

    def test_published_asset_survives_source_deletion(self) -> None:
        published = self.publish(parameters={"score_mode": "topk", "topk": 5, "margin": 0.1})
        self.source.unlink()
        result = ModelLibrary(self.library.root).resolve(published["model_id"], published["version"], "b0")
        self.assertEqual(Path(result["model_file"]).read_bytes(), b"immutable model sample")
        self.assertEqual(result["parameters"], {"score_mode": "topk", "margin": 0.1, "topk": 5})

    def test_new_versions_never_overwrite_or_follow_latest(self) -> None:
        first = self.publish()
        self.source.write_bytes(b"a newly trained model")
        second = self.publish(model_id=first["model_id"])
        self.assertEqual(first["version"], "v000001")
        self.assertEqual(second["version"], "v000002")
        self.assertEqual(first["model_id"], second["model_id"])
        self.assertEqual(Path(first["model_file"]).read_bytes(), b"immutable model sample")
        self.assertEqual(Path(second["model_file"]).read_bytes(), b"a newly trained model")
        self.assertEqual(self.library.resolve(first["model_id"], first["version"])["version"], "v000001")

    def test_identical_names_do_not_merge_model_identities(self) -> None:
        first, second = self.publish(), self.publish()
        self.assertNotEqual(first["model_id"], second["model_id"])
        self.assertEqual(len(self.library.list_models()), 2)

    def test_algorithm_mismatch_is_rejected_even_after_cached_load(self) -> None:
        published = self.publish()
        with self.assertRaisesRegex(ValueError, "algorithm"):
            self.library.resolve(published["model_id"], published["version"], "mobilenet_v3_small")

    def test_cached_verification_detects_changed_asset(self) -> None:
        published = self.publish()
        fresh_library = ModelLibrary(self.library.root)
        with patch.object(model_library, "_sha256", wraps=model_library._sha256) as digest:
            fresh_library.resolve(published["model_id"], published["version"])
            fresh_library.resolve(published["model_id"], published["version"])
            self.assertEqual(digest.call_count, 1)
            Path(published["model_file"]).write_bytes(b"corrupted model sample")
            with self.assertRaisesRegex(ValueError, "checksum"):
                fresh_library.resolve(published["model_id"], published["version"])
            self.assertEqual(digest.call_count, 2)

    def test_cached_manifest_and_result_cannot_be_mutated_by_caller(self) -> None:
        published = self.publish()
        published["parameters"]["margin"] = 100
        resolved = self.library.resolve(published["model_id"], published["version"])
        self.assertEqual(resolved["parameters"]["margin"], 0.02)

    def test_changed_manifest_does_not_use_cached_model(self) -> None:
        published = self.publish()
        manifest_path = Path(published["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["entry_file"] = "../source.npz"
        manifest["assets"][0]["file"] = "../source.npz"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "filename"):
            self.library.resolve(published["model_id"], published["version"])
        self.assertEqual(self.library.list_models(), [])

    def test_missing_fixed_version_never_falls_back(self) -> None:
        published = self.publish()
        with self.assertRaises(FileNotFoundError):
            self.library.resolve(published["model_id"], "v000002")
        Path(published["model_file"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.library.resolve(published["model_id"], published["version"])

    def test_invalid_identity_and_paths_are_rejected(self) -> None:
        for model_id, version in [("../outside", "v000001"), ("a" * 32, "../other"), ("a" * 32, "v000000")]:
            with self.subTest(model_id=model_id, version=version), self.assertRaises(ValueError):
                self.library.resolve(model_id, version)

    def test_symlink_cannot_escape_library_root(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        self.library.root.mkdir()
        identity = "a" * 32
        try:
            (self.library.root / identity).symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Directory symlinks unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.library.resolve(identity, "v000001")
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.publish(model_id=identity)

    def test_traditional_threshold_is_a_standalone_asset(self) -> None:
        threshold = {
            "algorithm": "meanintensity", "threshold": 80, "ok_when": "greater_equal",
            "ok_mean": 120, "ng_mean": 40, "accuracy": 1.0, "roi_label": "roi1",
        }
        published = self.library.publish("亮度", "meanintensity", traditional_model=threshold)
        threshold["threshold"] = 999
        fresh = ModelLibrary(self.library.root).resolve(published["model_id"], published["version"], "meanintensity")
        self.assertEqual(fresh["kind"], "traditional")
        self.assertEqual(fresh["traditional_model"]["threshold"], 80)
        self.assertEqual(fresh["model_file"], "")

    def test_invalid_publication_does_not_create_partial_versions(self) -> None:
        with self.assertRaises(ValueError):
            self.library.publish("bad", "meanintensity", model_file=self.source)
        with self.assertRaises(ValueError):
            self.publish(parameters={"margin": float("nan")})
        self.assertEqual(self.library.list_models(), [])

    def test_library_keys_are_strict_and_distinct_from_local_keys(self) -> None:
        key = make_library_model_key("cam1", "a" * 32, "v000001")
        self.assertEqual(parse_library_model_key(key), ("a" * 32, "v000001"))
        self.assertIsNone(parse_library_model_key("cam1__roi1"))
        with self.assertRaises(ValueError):
            parse_library_model_key("cam1__library__../../other__v000001")
        with self.assertRaises(ValueError):
            make_library_model_key("cam1", "a" * 32, "../other")


if __name__ == "__main__":
    unittest.main()
