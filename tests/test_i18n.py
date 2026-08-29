from __future__ import annotations

import string
import unittest

from ui import i18n
from ui.runtime.runtime_mode_pyside6 import _conveyor_operator_fault_text


def _format_fields(text: str) -> set[str]:
    return {
        field_name
        for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(text)
        if field_name
    }


class I18nResourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_language = i18n.language_code()

    def tearDown(self) -> None:
        i18n.set_language(self._original_language, persist=False)

    def test_language_resources_have_identical_keys(self) -> None:
        zh = i18n._TRANSLATIONS[i18n.LANG_ZH]
        en = i18n._TRANSLATIONS[i18n.LANG_EN]
        self.assertEqual(set(zh), set(en))

    def test_translation_placeholders_match(self) -> None:
        zh = i18n._TRANSLATIONS[i18n.LANG_ZH]
        en = i18n._TRANSLATIONS[i18n.LANG_EN]
        mismatches = {
            key: (_format_fields(zh[key]), _format_fields(en[key]))
            for key in zh
            if _format_fields(zh[key]) != _format_fields(en[key])
        }
        self.assertEqual(mismatches, {})

    def test_conveyor_fault_text_follows_selected_language(self) -> None:
        i18n.set_language(i18n.LANG_ZH, persist=False)
        self.assertIn("DI7", _conveyor_operator_fault_text("GOOD_OUTLET_TIMEOUT", ""))
        self.assertIn("清线", _conveyor_operator_fault_text("GOOD_OUTLET_TIMEOUT", ""))

        i18n.set_language(i18n.LANG_EN, persist=False)
        text = _conveyor_operator_fault_text("GOOD_OUTLET_TIMEOUT", "")
        self.assertIn("DI7", text)
        self.assertIn("purge", text.lower())

    def test_jam_detail_takes_priority_over_other_active_inputs(self) -> None:
        i18n.set_language(i18n.LANG_EN, persist=False)
        text = _conveyor_operator_fault_text(
            "JAM_DETECTED",
            "waste_outlet_sensor active too long",
            {
                "end_test_sensor": True,
                "waste_outlet_sensor": True,
            },
        )
        self.assertIn("DI8", text)

    def test_unknown_fault_keeps_engineering_code(self) -> None:
        i18n.set_language(i18n.LANG_ZH, persist=False)
        self.assertIn("CUSTOM_FAULT", _conveyor_operator_fault_text("CUSTOM_FAULT", ""))


if __name__ == "__main__":
    unittest.main()
