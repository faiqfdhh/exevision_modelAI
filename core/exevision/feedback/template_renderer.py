"""Template rendering helpers for deterministic phrase selection and slot filling."""

from __future__ import annotations

import hashlib
import re


class TemplateRenderer:
    """Handles deterministic phrase selection and template slot replacement."""

    _SLOT_PATTERN = re.compile(r"\[([A-Z_]+)\]")

    @staticmethod
    def select_phrase(phrases: list[str], video_id: str, rep_id: int, metric_key: str) -> str:
        """Select one phrase deterministically from a list."""
        if not phrases:
            raise ValueError("phrases list cannot be empty")

        key = f"{video_id}:{rep_id}:{metric_key}"
        hash_int = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return phrases[hash_int % len(phrases)]

    @classmethod
    def fill_slots(cls, template: str, slots: dict[str, str]) -> str:
        """Replace [SLOT] placeholders in template with values from slots."""

        def _replace(match: re.Match[str]) -> str:
            slot_name = match.group(1)
            if slot_name not in slots:
                raise KeyError(f"Slot [{slot_name}] not found in slots dict")
            return str(slots[slot_name])

        return cls._SLOT_PATTERN.sub(_replace, template)

    @staticmethod
    def humanize_metric(metric_key: str) -> str:
        """Convert metric keys like forward_lean into user-friendly labels."""
        return metric_key.replace("_", " ")
