from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.product_session import ProductSession, SessionData


_LOCAL_TMP_ROOT = PROJECT_DIR / ".tmp_test_runs"
_LOCAL_TMP_ROOT.mkdir(exist_ok=True)


def _make_session_root() -> Path:
    root = _LOCAL_TMP_ROOT / f"session_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


class ProductSessionTriggerDelayTest(unittest.TestCase):
    def test_session_persists_foot_trigger_delay_ms(self) -> None:
        session_root = _make_session_root()
        try:
            session = ProductSession(str(session_root))
            session.load()
            session.switch_product("Default")

            session.save_session(SessionData(foot_trigger_delay_ms=275))

            loaded = session.load_session()
            self.assertEqual(loaded.foot_trigger_delay_ms, 275)

            raw = json.loads(Path(session.session_json).read_text(encoding="utf-8"))
            self.assertEqual(raw["foot_trigger_delay_ms"], 275)
        finally:
            shutil.rmtree(session_root, ignore_errors=True)

    def test_negative_trigger_delay_is_clamped_to_zero(self) -> None:
        session_root = _make_session_root()
        try:
            session = ProductSession(str(session_root))
            session.load()
            session.switch_product("Default")

            session.save_session(SessionData(foot_trigger_delay_ms=-35))

            loaded = session.load_session()
            self.assertEqual(loaded.foot_trigger_delay_ms, 0)

            raw = json.loads(Path(session.session_json).read_text(encoding="utf-8"))
            self.assertEqual(raw["foot_trigger_delay_ms"], 0)
        finally:
            shutil.rmtree(session_root, ignore_errors=True)

    def test_generic_session_save_keeps_existing_trigger_delay(self) -> None:
        session_root = _make_session_root()
        try:
            session = ProductSession(str(session_root))
            session.load()
            session.switch_product("Default")

            session.save_session(SessionData(foot_trigger_delay_ms=420))
            session.save_session(SessionData(ref_image=""))

            loaded = session.load_session()
            self.assertEqual(loaded.foot_trigger_delay_ms, 420)
        finally:
            shutil.rmtree(session_root, ignore_errors=True)

    def test_trigger_delay_is_scoped_per_product(self) -> None:
        session_root = _make_session_root()
        try:
            session = ProductSession(str(session_root))
            session.load()
            self.assertEqual(session.create_product("Small"), "")
            self.assertEqual(session.create_product("Large"), "")

            session.switch_product("Small")
            session.save_session(SessionData(foot_trigger_delay_ms=120))

            session.switch_product("Large")
            session.save_session(SessionData(foot_trigger_delay_ms=850))

            session.switch_product("Small")
            self.assertEqual(session.load_session().foot_trigger_delay_ms, 120)

            session.switch_product("Large")
            self.assertEqual(session.load_session().foot_trigger_delay_ms, 850)
        finally:
            shutil.rmtree(session_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
