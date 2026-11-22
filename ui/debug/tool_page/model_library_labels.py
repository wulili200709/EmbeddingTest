"""Readable identities and provenance for public model selection."""
from __future__ import annotations

from datetime import datetime, timezone

from ui.algorithm_labels import algorithm_display_name
from ui.i18n import tr


def short_model_id(model_id, models=()) -> str:
    identifier = str(model_id or "")
    others = {str(model.get("model_id", "")) for model in models} - {identifier}
    length = min(8, len(identifier))
    while length < len(identifier) and any(other.startswith(identifier[:length]) for other in others):
        length += 1
    return identifier[:length] or "-"


def library_model_label(model: dict, models=()) -> str:
    algorithm = str(model.get("algorithm", "") or "")
    return tr(
        "debug.model_source.library_option",
        name=model.get("name") or "-", version=model.get("version") or "-",
        identifier=short_model_id(model.get("model_id"), models),
        algorithm=algorithm_display_name(algorithm) or "-",
    )


def library_model_origin(model: dict) -> str:
    source = model.get("origin", model.get("source", {}))
    if not isinstance(source, dict):
        source = {}
    camera = str(source.get("camera_id", "") or "")
    if not camera:
        key_camera = str(source.get("model_key", "")).partition("__")[0]
        camera = key_camera if key_camera in {"cam1", "cam2", "cam3"} else ""
    parts = [str(source.get("order", "") or "-")]
    if camera:
        parts.append(camera)
    if source.get("group"):
        parts.append(tr("debug.model_source.group_status", group=source["group"]))
    roi = str(source.get("roi_display_name") or source.get("roi") or "")
    if roi:
        parts.append(roi)
    return " / ".join(parts)


def library_model_details(model: dict) -> str:
    stamp = str(model.get("created_at", "") or "-")
    try:
        # Stored publication timestamps are UTC; keep the timezone explicit.
        published = datetime.fromisoformat(stamp)
        if published.tzinfo is not None:
            stamp = published.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        pass
    return tr(
        "debug.model_source.library_details",
        name=model.get("name") or "-", version=model.get("version") or "-",
        algorithm=algorithm_display_name(model.get("algorithm")) or "-",
        origin=library_model_origin(model), published=stamp,
        identifier=model.get("model_id") or "-",
    )
