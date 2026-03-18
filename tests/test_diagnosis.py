"""Tests for DiagnosisEngine."""

import pytest

from src.reasoning.diagnosis import DiagnosisEngine, DiagnosisResult


def _make_detection(class_name: str, confidence: float = 0.9) -> dict:
    return {
        "bbox": [0.0, 0.0, 100.0, 100.0],
        "confidence": confidence,
        "class_id": 0,
        "class_name": class_name,
    }


# ---------------------------------------------------------------------------
# Rule-matching tests
# ---------------------------------------------------------------------------

def test_point_hotspot_soiling() -> None:
    """hotspot IR + soiling RGB → low priority physical contamination."""
    engine = DiagnosisEngine()
    ir_dets = [_make_detection("hotspot")]
    rgb_dets = [_make_detection("soiling")]

    result = engine.diagnose(ir_dets, rgb_dets)

    assert result.priority == "low"
    assert "contamination" in result.diagnosis.lower() or "soiling" in result.diagnosis.lower() or "physical" in result.diagnosis.lower()


def test_point_hotspot_clean() -> None:
    """hotspot IR + clean-category RGB → high priority cell defect."""
    engine = DiagnosisEngine()
    ir_dets = [_make_detection("hotspot")]
    rgb_dets = [_make_detection("hotspot")]  # maps to 'clean' rgb_pattern

    result = engine.diagnose(ir_dets, rgb_dets)

    assert result.priority == "high"
    assert "defect" in result.diagnosis.lower() or "short" in result.diagnosis.lower()


def test_row_heating_clean() -> None:
    """bypass_diode_fault IR + clean RGB → critical priority."""
    engine = DiagnosisEngine()
    ir_dets = [_make_detection("bypass_diode_fault")]
    rgb_dets = [_make_detection("hotspot")]  # maps to 'clean'

    result = engine.diagnose(ir_dets, rgb_dets)

    assert result.priority == "critical"
    assert "diode" in result.diagnosis.lower() or "bypass" in result.diagnosis.lower()


def test_patchy_heating_crack() -> None:
    """crack IR + crack RGB → medium priority mechanical damage."""
    engine = DiagnosisEngine()
    ir_dets = [_make_detection("crack")]
    rgb_dets = [_make_detection("crack")]

    result = engine.diagnose(ir_dets, rgb_dets)

    assert result.priority == "medium"
    assert "mechanical" in result.diagnosis.lower() or "damage" in result.diagnosis.lower()


def test_no_detections() -> None:
    """Empty detection lists must yield the 'No fault detected' result."""
    engine = DiagnosisEngine()
    result = engine.diagnose([], [])

    assert "no fault" in result.diagnosis.lower()
    assert result.priority == "none"
    assert result.ir_pattern == "none"
    assert result.rgb_pattern == "none"


# ---------------------------------------------------------------------------
# DiagnosisResult structure tests
# ---------------------------------------------------------------------------

def test_diagnosis_result_fields() -> None:
    """DiagnosisResult must expose all required fields."""
    required_fields = {"ir_pattern", "rgb_pattern", "diagnosis", "priority", "action"}
    assert set(DiagnosisResult._fields) == required_fields


def test_diagnosis_result_is_named_tuple() -> None:
    """DiagnosisResult must be a named tuple (supports both index and name access)."""
    result = DiagnosisResult(
        ir_pattern="point_hotspot",
        rgb_pattern="soiling",
        diagnosis="Test",
        priority="low",
        action="Clean it",
    )
    assert result[0] == "point_hotspot"
    assert result.ir_pattern == "point_hotspot"


# ---------------------------------------------------------------------------
# Pattern classifier edge cases
# ---------------------------------------------------------------------------

def test_unknown_class_name_returns_none_pattern() -> None:
    """Unknown class names should produce 'none' patterns without errors."""
    engine = DiagnosisEngine()
    result = engine.diagnose(
        [_make_detection("unknown_fault")],
        [_make_detection("unknown_visual")],
    )
    assert result.ir_pattern == "none"
    assert result.rgb_pattern == "none"


def test_priority_ordering_ir_row_heating_wins() -> None:
    """row_heating should take priority over point_hotspot in IR pattern."""
    engine = DiagnosisEngine()
    ir_dets = [
        _make_detection("hotspot"),          # → point_hotspot
        _make_detection("bypass_diode_fault"),  # → row_heating (higher priority)
    ]
    rgb_dets = [_make_detection("hotspot")]

    result = engine.diagnose(ir_dets, rgb_dets)

    assert result.ir_pattern == "row_heating"
