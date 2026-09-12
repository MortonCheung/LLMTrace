"""Measurement resolution note（Task 33）.

The Quick Suite declaration is a machine-readable screening label: it must stay
exactly as specified, carry no tolerance numbers, and be immutable once built.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmtrace.analysis.resolution import (
    MEASUREMENT_RESOLUTION_LABEL,
    QUICK_SUITE_MEASUREMENT_RESOLUTION,
    QUICK_SUITE_RESOLUTION_DISCLAIMER,
    MeasurementResolution,
    MeasurementResolutionNote,
    quick_suite_resolution_note,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestResolutionConstants:
    def test_label_is_the_report_section_name(self) -> None:
        assert MEASUREMENT_RESOLUTION_LABEL == "Measurement Resolution"

    def test_quick_suite_resolution_is_screening(self) -> None:
        assert QUICK_SUITE_MEASUREMENT_RESOLUTION == "SCREENING"

    def test_disclaimer_states_screening_and_requires_validation(self) -> None:
        assert "screening measurement" in QUICK_SUITE_RESOLUTION_DISCLAIMER
        assert "independent validation" in QUICK_SUITE_RESOLUTION_DISCLAIMER


# ---------------------------------------------------------------------------
# MeasurementResolution
# ---------------------------------------------------------------------------


class TestMeasurementResolution:
    def test_enum_has_only_the_screening_level(self) -> None:
        assert [member.name for member in MeasurementResolution] == ["SCREENING"]

    def test_enum_member_value_is_the_specified_string(self) -> None:
        assert MeasurementResolution.SCREENING == QUICK_SUITE_MEASUREMENT_RESOLUTION


# ---------------------------------------------------------------------------
# MeasurementResolutionNote
# ---------------------------------------------------------------------------


class TestMeasurementResolutionNote:
    def test_quick_suite_note_carries_label_resolution_and_disclaimer(self) -> None:
        note = quick_suite_resolution_note()

        assert note.label == "Measurement Resolution"
        assert note.resolution is MeasurementResolution.SCREENING
        assert note.disclaimer == QUICK_SUITE_RESOLUTION_DISCLAIMER

    def test_label_defaults_to_the_section_name(self) -> None:
        note = MeasurementResolutionNote(
            resolution=MeasurementResolution.SCREENING,
            disclaimer="Screening only.",
        )

        assert note.label == MEASUREMENT_RESOLUTION_LABEL

    def test_note_is_frozen(self) -> None:
        note = quick_suite_resolution_note()

        with pytest.raises(ValidationError):
            note.resolution = MeasurementResolution.SCREENING  # type: ignore[misc]

    def test_note_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            MeasurementResolutionNote(
                resolution=MeasurementResolution.SCREENING,
                disclaimer="Screening only.",
                tolerance=0.05,  # type: ignore[call-arg]
            )

    @pytest.mark.parametrize("disclaimer", ["", None])
    def test_disclaimer_is_required_and_non_empty(self, disclaimer: str | None) -> None:
        with pytest.raises(ValidationError):
            MeasurementResolutionNote(
                resolution=MeasurementResolution.SCREENING,
                disclaimer=disclaimer,  # type: ignore[arg-type]
            )

    def test_empty_label_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeasurementResolutionNote(
                label="",
                resolution=MeasurementResolution.SCREENING,
                disclaimer="Screening only.",
            )
