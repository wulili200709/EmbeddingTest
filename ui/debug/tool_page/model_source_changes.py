"""Apply an explicit tool choice without altering a shared model."""
from __future__ import annotations

from common.algorithm_codes import storage_code_backbone


def detach_for_tool_change(owner, item, algorithm: str) -> bool:
    if item is None or getattr(item, "model_source", "independent") == "independent":
        return False
    try:
        previous = owner.algo.resolve_item_algorithm(item)
    except (OSError, ValueError, KeyError):
        previous = None
    if previous is not None and storage_code_backbone(previous) == storage_code_backbone(algorithm):
        return False
    item.model_source = "independent"
    item.task_group = ""
    item.legacy_model_key = ""
    item._model_source_explicit = True
    item.library_model_id = ""
    item.library_version = ""
    owner.algo.model = None
    owner.algo._loaded_embedding_model_key = ("", "")
    owner.algo._loaded_yolo_model_key = ""
    owner.algo._active_library_parameters = None
    owner.algo.clear_embedding_model_cache()
    owner._clear_training_roi_review_state(item.camera_id)
    return True
