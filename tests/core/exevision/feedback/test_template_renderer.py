"""Tests for TemplateRenderer phrase selection and slot filling."""

import pytest

from core.exevision.feedback.template_renderer import TemplateRenderer


class TestPhraseSelection:
    def test_select_phrase_is_deterministic(self):
        renderer = TemplateRenderer()
        phrases = ["Great job!", "Excellent!", "Solid work!"]

        phrase1 = renderer.select_phrase(phrases, "vid_001", 1, "depth")
        phrase2 = renderer.select_phrase(phrases, "vid_001", 1, "depth")

        assert phrase1 == phrase2

    def test_select_phrase_is_varied(self):
        renderer = TemplateRenderer()
        phrases = ["Great job!", "Excellent!", "Solid work!"]

        phrase1 = renderer.select_phrase(phrases, "vid_001", 1, "depth")
        phrase2 = renderer.select_phrase(phrases, "vid_001", 1, "lean")

        assert phrase1 in phrases
        assert phrase2 in phrases

    def test_select_phrase_all_indices_valid(self):
        renderer = TemplateRenderer()
        phrases = ["A", "B", "C", "D", "E"]

        for idx in range(100):
            phrase = renderer.select_phrase(phrases, f"vid_{idx}", idx, f"metric_{idx}")
            assert phrase in phrases

    def test_select_phrase_empty_raises(self):
        renderer = TemplateRenderer()
        with pytest.raises(ValueError):
            renderer.select_phrase([], "vid_001", 1, "depth")


class TestSlotFilling:
    def test_fill_single_slot(self):
        renderer = TemplateRenderer()
        template = "Work on [ISSUE]. Try [CUE]."
        slots = {"ISSUE": "forward lean", "CUE": "leaning back more"}

        result = renderer.fill_slots(template, slots)

        assert result == "Work on forward lean. Try leaning back more."

    def test_fill_missing_slot_raises_error(self):
        renderer = TemplateRenderer()
        template = "Work on [ISSUE]. Try [CUE]."
        slots = {"ISSUE": "forward lean"}

        with pytest.raises(KeyError):
            renderer.fill_slots(template, slots)

    def test_fill_no_slots(self):
        renderer = TemplateRenderer()
        template = "Great form!"

        result = renderer.fill_slots(template, {})

        assert result == "Great form!"
