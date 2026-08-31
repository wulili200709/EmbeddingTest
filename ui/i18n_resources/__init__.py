"""Language resource dictionaries used by :mod:`ui.i18n`."""

from __future__ import annotations

from collections.abc import Mapping


def merge_resources(*resources: Mapping[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for resource in resources:
        duplicates = set(merged).intersection(resource)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate i18n resource key(s): {names}")
        merged.update({str(key): str(value) for key, value in resource.items()})
    return merged


__all__ = ["merge_resources"]
