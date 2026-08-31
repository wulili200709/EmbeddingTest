from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.app_paths import writable_embedding_test_root
from ui.i18n_resources.en_us import TRANSLATIONS as EN_TRANSLATIONS
from ui.i18n_resources.zh_cn import TRANSLATIONS as ZH_TRANSLATIONS


LANG_ZH = "zh_CN"
LANG_EN = "en_US"
SUPPORTED_LANGUAGES = (LANG_ZH, LANG_EN)

_TRANSLATIONS: dict[str, dict[str, str]] = {
    LANG_ZH: ZH_TRANSLATIONS,
    LANG_EN: EN_TRANSLATIONS,
}

_STATUS_TEXT_KEYS = {
    "未检测": "runtime.untested",
    "等待触发": "runtime.state.WaitingTrigger",
    "已放行，待消耗": "runtime.state.ReleasedPendingConsume",
    "已消耗一次放行": "runtime.state.ReleasedPendingConsume",
    "未锁定": "runtime.unlocked",
    "未初始化": "status.io_uninitialized",
    "NG锁定": "runtime.state.LockedByNg",
    "NG 锁定": "runtime.state.LockedByNg",
    "采集中(相机1)": "runtime.state.CapturingCam1",
    "采集中（相机1）": "runtime.state.CapturingCam1",
    "采集中(相机2)": "runtime.state.CapturingCam2",
    "采集中（相机2）": "runtime.state.CapturingCam2",
    "采集中(相机3)": "runtime.state.CapturingCam3",
    "采集中（相机3）": "runtime.state.CapturingCam3",
    "检测中": "runtime.state.Inspecting",
    "汇总结论": "runtime.state.Aggregating",
    "汇总结果": "runtime.state.Aggregating",
    "本轮完成 OK": "runtime.state.CompletedOk",
    "本轮 NG": "runtime.state.CompletedNg",
    "运行异常": "runtime.state.Error",
    "服务不可用": "runtime.state.Unavailable",
    "服务导入失败": "runtime.state.Unavailable",
    "相机未接入": "runtime.not_connected",
    "已禁用": "debug.status.disabled",
    "未连接相机": "runtime.no_camera_connected",
    "未连接": "runtime.not_connected",
}

_language = LANG_ZH


def _settings_path() -> Path:
    return writable_embedding_test_root(__file__) / "config" / "ui_settings.json"


def language_code() -> str:
    return _language


def set_language(code: str, *, persist: bool = True) -> str:
    global _language
    normalized = str(code or "").strip()
    if normalized not in SUPPORTED_LANGUAGES:
        normalized = LANG_ZH
    _language = normalized
    if persist:
        save_language(normalized)
    return _language


def load_language() -> str:
    path = _settings_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set_language(LANG_ZH, persist=False)
    return set_language(str(payload.get("language") or LANG_ZH), persist=False)


def save_language(code: str) -> None:
    path = _settings_path()
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    payload["language"] = code
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def tr(key: str, **kwargs: object) -> str:
    text = _TRANSLATIONS.get(_language, {}).get(key)
    if text is None:
        text = _TRANSLATIONS[LANG_ZH].get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


def tr_runtime_state(state: str) -> str:
    value = str(state or "").strip()
    return tr(f"runtime.state.{value}") if value else ""


def tr_status_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    key = _STATUS_TEXT_KEYS.get(value)
    if key:
        return tr(key)
    if value.startswith("已连接:"):
        return "Connected: " + value.split(":", 1)[1].strip() if _language == LANG_EN else value
    return value


load_language()
