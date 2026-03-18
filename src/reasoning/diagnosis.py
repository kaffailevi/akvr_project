"""Rule-based diagnosis engine for solar panel fault analysis."""

from collections import namedtuple
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Named tuple for structured diagnosis output
# ---------------------------------------------------------------------------

DiagnosisResult = namedtuple(
    "DiagnosisResult",
    ["ir_pattern", "rgb_pattern", "diagnosis", "priority", "action"],
)


class DiagnosisEngine:
    """Autonomous reasoning engine that maps detection patterns to diagnoses.

    The engine consults a built-in rule table that correlates IR thermal
    patterns with RGB visual patterns to produce actionable fault diagnoses.
    """

    DIAGNOSIS_TABLE: List[Dict] = [
        {
            "ir_pattern": "point_hotspot",
            "rgb_pattern": "soiling",
            "diagnosis": "Physical contamination (bird droppings / leaf)",
            "priority": "low",
            "action": "Cleaning required",
        },
        {
            "ir_pattern": "point_hotspot",
            "rgb_pattern": "clean",
            "diagnosis": "Cell defect / Internal short circuit",
            "priority": "high",
            "action": "Panel replacement required",
        },
        {
            "ir_pattern": "row_heating",
            "rgb_pattern": "clean",
            "diagnosis": "Bypass diode fault",
            "priority": "critical",
            "action": "Immediate service required",
        },
        {
            "ir_pattern": "patchy_heating",
            "rgb_pattern": "crack",
            "diagnosis": "Mechanical damage",
            "priority": "medium",
            "action": "Monitoring and scheduled repair",
        },
    ]

    # Mapping from YOLO class names → ir_pattern strings
    _IR_PATTERN_MAP: Dict[str, str] = {
        "hotspot": "point_hotspot",
        "cell_defect": "point_hotspot",
        "bypass_diode_fault": "row_heating",
        "soiling": "patchy_heating",
        "crack": "patchy_heating",
    }

    # Mapping from YOLO class names → rgb_pattern strings
    _RGB_PATTERN_MAP: Dict[str, str] = {
        "soiling": "soiling",
        "crack": "crack",
        "hotspot": "clean",
        "cell_defect": "clean",
        "bypass_diode_fault": "clean",
    }

    # Default result when no rule matches
    _NO_FAULT = DiagnosisResult(
        ir_pattern="none",
        rgb_pattern="none",
        diagnosis="No fault detected",
        priority="none",
        action="No action required",
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(
        self,
        ir_detections: List[Dict],
        rgb_detections: List[Dict],
    ) -> DiagnosisResult:
        """Match detection patterns against the rule table and return a diagnosis.

        Args:
            ir_detections: List of detection dicts from the IR image (each dict
                           must have a ``class_name`` key).
            rgb_detections: List of detection dicts from the RGB image.

        Returns:
            :class:`DiagnosisResult` named tuple.  Returns a "No fault detected"
            result when no rule matches.
        """
        ir_pattern = self._classify_ir_pattern(ir_detections)
        rgb_pattern = self._classify_rgb_pattern(rgb_detections)

        for rule in self.DIAGNOSIS_TABLE:
            if rule["ir_pattern"] == ir_pattern and rule["rgb_pattern"] == rgb_pattern:
                return DiagnosisResult(
                    ir_pattern=ir_pattern,
                    rgb_pattern=rgb_pattern,
                    diagnosis=rule["diagnosis"],
                    priority=rule["priority"],
                    action=rule["action"],
                )

        return self._NO_FAULT

    # ------------------------------------------------------------------
    # Pattern classifiers
    # ------------------------------------------------------------------

    def _classify_ir_pattern(self, detections: List[Dict]) -> str:
        """Determine the dominant IR thermal pattern from a list of detections.

        Priority order: ``row_heating`` > ``point_hotspot`` > ``patchy_heating``.

        Args:
            detections: Detection dicts from the IR image.

        Returns:
            Pattern string or ``"none"`` when no detections exist.
        """
        if not detections:
            return "none"

        patterns = {
            self._IR_PATTERN_MAP.get(d.get("class_name", ""), "")
            for d in detections
        } - {""}

        # Use priority ordering
        for pattern in ("row_heating", "point_hotspot", "patchy_heating"):
            if pattern in patterns:
                return pattern

        return "none"

    def _classify_rgb_pattern(self, detections: List[Dict]) -> str:
        """Determine the dominant RGB visual pattern from a list of detections.

        Priority order: ``crack`` > ``soiling`` > ``clean``.

        Args:
            detections: Detection dicts from the RGB image.

        Returns:
            Pattern string or ``"none"`` when no detections exist.
        """
        if not detections:
            return "none"

        patterns = {
            self._RGB_PATTERN_MAP.get(d.get("class_name", ""), "")
            for d in detections
        } - {""}

        for pattern in ("crack", "soiling", "clean"):
            if pattern in patterns:
                return pattern

        return "none"
