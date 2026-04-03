import sys
import os
import logging

# Настройка логов — INFO покажет что делает OCR по каждому варианту
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s"
)

sys.path.insert(0, "ml/detectors")

import cv2
from yolo_detector import YOLOBannerDetector

# ─── Настройки ───────────────────────────────
IMG_PATH = r"d:\train\images\img_116.jpg"   # ← меняй путь
OUTPUT   = "result.jpg"
# ─────────────────────────────────────────────

detector = YOLOBannerDetector(
    model_path="weights/best.pt",
    enable_ocr=True,
    ocr_min_confidence=0.2,
    roi_padding=15,
    ocr_debug=True,        # подробный лог по каждому варианту предобработки
)

annotated, detections = detector.detect_and_draw(IMG_PATH)
cv2.imwrite(OUTPUT, annotated)
print(f"\nРезультат сохранён: {OUTPUT}")
print(f"Найдено баннеров: {len(detections)}\n")

for i, d in enumerate(detections, 1):
    print(f"[{i}] bbox={d['bbox']}  детекция={d['confidence']:.2f}")
    if "ocr" in d:
        ocr = d["ocr"]
        print(f"     текст:      '{ocr['text']}'")
        print(f"     уверенность: {ocr['confidence']:.2f}")
        if ocr.get("words"):
            print("     слова:")
            for w in ocr["words"]:
                print(f"       '{w['text']}' ({w['confidence']:.2f})")
    print()
