"""Streamlit dashboard for solar panel monitoring."""

import io
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

try:
    from src.fusion.fusion import EarlyFusion
    from src.fusion.registration import ImageRegistrator
    from src.reasoning.diagnosis import DiagnosisEngine

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

try:
    from src.models.detector import SolarPanelDetector

    _DETECTOR_AVAILABLE = True
except ImportError:
    _DETECTOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Priority display helpers
# ---------------------------------------------------------------------------

_PRIORITY_ICONS = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🔴",
    "critical": "🚨",
    "none": "✅",
}


def _priority_icon(priority: str) -> str:
    return _PRIORITY_ICONS.get(priority.lower(), "❓")


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_image_as_numpy(uploaded_file) -> Optional[np.ndarray]:
    """Read a Streamlit UploadedFile as an (H, W, C) uint8 numpy array."""
    if uploaded_file is None:
        return None
    img = Image.open(io.BytesIO(uploaded_file.read()))
    return np.array(img.convert("RGB"))


def _load_ir_as_numpy(uploaded_file) -> Optional[np.ndarray]:
    """Read a Streamlit UploadedFile as an (H, W) uint8 numpy array (grayscale)."""
    if uploaded_file is None:
        return None
    img = Image.open(io.BytesIO(uploaded_file.read()))
    return np.array(img.convert("L"))


# ---------------------------------------------------------------------------
# Main dashboard function
# ---------------------------------------------------------------------------

def run_dashboard() -> None:
    """Entry point for the Streamlit solar panel monitoring dashboard."""
    st.set_page_config(
        page_title="Solar Panel Monitoring",
        page_icon="☀️",
        layout="wide",
    )
    st.title("☀️ Solar Panel Monitoring System")
    st.markdown(
        "Upload **RGB** and **IR/thermal** images of a solar panel to detect "
        "faults and receive an automated diagnosis."
    )

    # ------------------------------------------------------------------ #
    #  Sidebar configuration
    # ------------------------------------------------------------------ #
    st.sidebar.header("⚙️ Configuration")

    conf_threshold: float = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    fusion_method: str = st.sidebar.radio(
        "Fusion method",
        options=["Early Fusion", "Mid-Level Fusion"],
        index=0,
    )

    detection_mode: str = st.sidebar.radio(
        "Detection mode",
        options=["IR Only", "RGB + IR (Multimodal)"],
        index=1,
    )

    # ------------------------------------------------------------------ #
    #  Image uploaders
    # ------------------------------------------------------------------ #
    st.subheader("📸 Image Upload")
    col_rgb, col_ir = st.columns(2)

    with col_rgb:
        rgb_file = st.file_uploader(
            "Upload RGB image", type=["jpg", "jpeg", "png"], key="rgb_uploader"
        )

    with col_ir:
        ir_file = st.file_uploader(
            "Upload IR / Thermal image", type=["jpg", "jpeg", "png"], key="ir_uploader"
        )

    # ------------------------------------------------------------------ #
    #  Preview uploaded images
    # ------------------------------------------------------------------ #
    rgb_np: Optional[np.ndarray] = None
    ir_np: Optional[np.ndarray] = None

    if rgb_file is not None:
        rgb_np = _load_image_as_numpy(rgb_file)

    if ir_file is not None:
        ir_np = _load_ir_as_numpy(ir_file)

    if rgb_np is not None or ir_np is not None:
        st.subheader("🖼️ Uploaded Images")
        prev_col1, prev_col2 = st.columns(2)
        with prev_col1:
            if rgb_np is not None:
                st.image(rgb_np, caption="RGB Image", use_container_width=True)
            else:
                st.info("RGB image not yet uploaded.")
        with prev_col2:
            if ir_np is not None:
                st.image(ir_np, caption="IR / Thermal Image", use_container_width=True)
            else:
                st.info("IR image not yet uploaded.")

    # ------------------------------------------------------------------ #
    #  Analysis
    # ------------------------------------------------------------------ #
    if rgb_np is not None and ir_np is not None:
        if st.button("🔍 Analyze", type="primary"):
            _run_analysis(
                rgb_np=rgb_np,
                ir_np=ir_np,
                conf_threshold=conf_threshold,
                fusion_method=fusion_method,
                detection_mode=detection_mode,
            )
    elif rgb_np is None and ir_np is None:
        st.info("👆 Please upload both an RGB and an IR image to enable analysis.")
    elif rgb_np is None:
        st.warning("Please also upload an **RGB** image.")
    else:
        st.warning("Please also upload an **IR/Thermal** image.")


def _run_analysis(
    rgb_np: np.ndarray,
    ir_np: np.ndarray,
    conf_threshold: float,
    fusion_method: str,
    detection_mode: str,
) -> None:
    """Execute the full analysis pipeline and render results."""
    import cv2  # local import

    st.subheader("🔬 Analysis Results")

    # ---- Fusion & registration ---------------------------------------- #
    if detection_mode == "RGB + IR (Multimodal)" and _CORE_AVAILABLE:
        with st.spinner("Registering images…"):
            registrator = ImageRegistrator()
            registered_ir, H = registrator.register(rgb_np, ir_np)

        if fusion_method == "Early Fusion":
            with st.spinner("Applying Early Fusion…"):
                fuser = EarlyFusion()
                rgb_t, ir_t = EarlyFusion.to_tensor(rgb_np, registered_ir)
                fused_tensor = fuser(rgb_t, ir_t)  # (4, H, W)

            # Display as RGBA by mapping the 4th channel to alpha
            fused_np = (fused_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            st.image(
                fused_np,
                caption=f"Early Fusion output (4-ch shown as RGBA, shape {tuple(fused_tensor.shape)})",
                use_container_width=True,
            )
        else:
            st.info(
                "Mid-Level Fusion operates at the feature level inside the model — "
                "no fused image to display."
            )
    else:
        registered_ir = ir_np

    # ---- Detection --------------------------------------------------- #
    ir_detections = []
    rgb_detections = []

    if not _DETECTOR_AVAILABLE:
        st.error(
            "⚠️ `ultralytics` is not installed. Skipping detection. "
            "Install it with `pip install ultralytics`."
        )
    else:
        with st.spinner("Running detection…"):
            try:
                detector = SolarPanelDetector(device="cpu")

                if detection_mode in ("IR Only", "RGB + IR (Multimodal)"):
                    ir_detections = detector.detect_hotspots(
                        registered_ir, conf_threshold=conf_threshold
                    )

                if detection_mode == "RGB + IR (Multimodal)":
                    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                    rgb_detections = detector.detect(
                        bgr, conf_threshold=conf_threshold
                    )
            except (RuntimeError, ValueError, OSError) as exc:
                st.error(f"Detection failed: {exc}")

        # Show detection tables
        col_det1, col_det2 = st.columns(2)
        with col_det1:
            st.markdown("**IR Detections**")
            if ir_detections:
                st.table(ir_detections)
            else:
                st.info("No IR detections above threshold.")

        with col_det2:
            st.markdown("**RGB Detections**")
            if rgb_detections:
                st.table(rgb_detections)
            else:
                st.info("No RGB detections above threshold.")

    # ---- Diagnosis --------------------------------------------------- #
    if _CORE_AVAILABLE:
        st.subheader("🩺 Automated Diagnosis")
        engine = DiagnosisEngine()
        result = engine.diagnose(ir_detections, rgb_detections)
        icon = _priority_icon(result.priority)

        st.markdown(f"### {icon} {result.diagnosis}")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Priority", f"{icon} {result.priority.upper()}")
        col_b.metric("IR Pattern", result.ir_pattern)
        col_c.metric("RGB Pattern", result.rgb_pattern)
        st.info(f"**Recommended action:** {result.action}")


if __name__ == "__main__":
    run_dashboard()
