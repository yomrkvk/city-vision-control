"""
YOLOv8 детектор баннеров с OCR — финальная версия.
ml/detectors/yolo_detector.py
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger(__name__)


class YOLOBannerDetector:
    def __init__(
        self,
        model_path: str = "weights/best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        enable_ocr: bool = True,
        ocr_min_confidence: float = 0.2,
        roi_padding: int = 15,
        ocr_debug: bool = False,
    ):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_ocr = enable_ocr
        self.ocr_min_confidence = ocr_min_confidence
        self.roi_padding = roi_padding
        self.ocr_debug = ocr_debug

        logger.info(f"Детектор загружен: {model_path}, OCR={'вкл' if enable_ocr else 'выкл'}")

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Не удалось загрузить: {image}")
            return img
        return image.copy()

    def detect(self, image: Union[str, Path, np.ndarray]) -> list:
        """
        Детектирует баннеры и читает с них текст через OCR.

        Returns: список dict:
          {
            bbox: [x1,y1,x2,y2],
            confidence: float,
            class_id: int,
            class_name: str,
            ocr: { text: str, confidence: float, words: [...] }
          }
        """
        from ocr_service import read_text_from_roi

        img = self._load_image(image)
        h, w = img.shape[:2]

        results = self.model(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names.get(class_id, str(class_id))

            detection = {
                "bbox":       [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "class_id":   class_id,
                "class_name": class_name,
            }

            if self.enable_ocr:
                rx1 = max(0, x1 - self.roi_padding)
                ry1 = max(0, y1 - self.roi_padding)
                rx2 = min(w, x2 + self.roi_padding)
                ry2 = min(h, y2 + self.roi_padding)
                roi = img[ry1:ry2, rx1:rx2]

                detection["ocr"] = read_text_from_roi(
                    roi,
                    min_confidence=self.ocr_min_confidence,
                    debug=self.ocr_debug,
                )

            detections.append(detection)

        return detections

    def detect_and_draw(
        self,
        image: Union[str, Path, np.ndarray],
        show_ocr_text: bool = True,
    ) -> tuple:
        """
        Детектирует и рисует результат на изображении.
        Returns: (annotated_image, detections)
        """
        img = self._load_image(image)
        detections = self.detect(img)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            class_name = det["class_name"]

            # Bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)

            # Текст лейбла
            ocr_text = ""
            ocr_conf = 0.0
            if show_ocr_text and "ocr" in det:
                ocr_text = det["ocr"].get("text", "")
                ocr_conf = det["ocr"].get("confidence", 0.0)

            if ocr_text:
                display = ocr_text[:70] + ("…" if len(ocr_text) > 70 else "")
                label = f"{class_name} {conf:.2f} | \"{display}\" ({ocr_conf:.2f})"
            else:
                label = f"{class_name} {conf:.2f}"

            # Фон под текст
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 220, 0), -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), thickness)

        return img, detections
