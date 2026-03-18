"""YOLOv8-based detector for solar panel fault detection."""

from typing import Dict, List, Optional

import numpy as np
import torch


class SolarPanelDetector:
    """Wraps a YOLOv8 model for solar panel fault detection.

    Args:
        model_path: Path to a YOLOv8 model weights file (e.g. ``yolov8n.pt``).
        device: Target device string (``"cuda"``, ``"cpu"``).  If ``None`` the
                device is selected automatically.
    """

    FAULT_CLASSES: Dict[int, str] = {
        0: "hotspot",
        1: "bypass_diode_fault",
        2: "cell_defect",
        3: "soiling",
        4: "crack",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: Optional[str] = None,
    ) -> None:
        from ultralytics import YOLO  # local import keeps the module importable

        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Dict]:
        """Run object detection on a single image.

        Args:
            image: Image as a numpy array (H, W, C) in BGR or grayscale uint8.
            conf_threshold: Minimum confidence to keep a detection.
            iou_threshold: IoU threshold for NMS.

        Returns:
            List of detection dicts, each with keys:
            ``bbox`` ([x1, y1, x2, y2]), ``confidence``, ``class_id``,
            ``class_name``.
        """
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.FAULT_CLASSES.get(cls_id, f"class_{cls_id}")
                detections.append(
                    {
                        "bbox": [round(v, 2) for v in xyxy],
                        "confidence": round(conf, 4),
                        "class_id": cls_id,
                        "class_name": cls_name,
                    }
                )
        return detections

    def detect_hotspots(
        self,
        ir_image: np.ndarray,
        conf_threshold: float = 0.3,
    ) -> List[Dict]:
        """Detect hotspots in an IR/thermal image.

        The grayscale IR image is converted to 3-channel BGR before inference
        so the model receives the expected input format.

        Args:
            ir_image: Grayscale IR image (H, W) or (H, W, 1).
            conf_threshold: Minimum confidence for detections.

        Returns:
            List of detection dicts (same structure as :meth:`detect`).
        """
        import cv2  # local import to avoid hard dependency at module level

        if ir_image.ndim == 2:
            bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
        elif ir_image.shape[2] == 1:
            bgr = cv2.cvtColor(ir_image[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            bgr = ir_image

        return self.detect(bgr, conf_threshold=conf_threshold)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
    ) -> None:
        """Fine-tune the model on a custom dataset.

        Args:
            data_yaml: Path to the YOLO dataset configuration YAML file.
            epochs: Number of training epochs.
            imgsz: Training image size (square).
            batch: Batch size.
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
        )
