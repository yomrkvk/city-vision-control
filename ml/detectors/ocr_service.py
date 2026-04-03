"""
OCR сервис — EasyOCR + стратегия полос для многострочных баннеров.
ml/detectors/ocr_service.py
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

_reader = None


def get_ocr_reader():
    global _reader
    if _reader is None:
        import easyocr
        logger.info("Инициализация EasyOCR...")
        _reader = easyocr.Reader(['ru', 'en'], gpu=False)
    return _reader


# ─────────────────────────────────────────────
#  Постобработка — исправление смешанных букв
# ─────────────────────────────────────────────

_CONFUSABLE = {
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М',
    'H': 'Н', 'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т',
    'Y': 'У', 'X': 'Х',
    'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
    'y': 'у', 'x': 'х',
}

def fix_mixed_text(text: str) -> str:
    words = text.split()
    result = []
    for word in words:
        cyr = sum(1 for c in word if '\u0400' <= c <= '\u04FF')
        lat = sum(1 for c in word if c.isalpha() and c.isascii())
        if cyr > 0 and lat > 0 and cyr >= lat:
            word = ''.join(_CONFUSABLE.get(c, c) for c in word)
        result.append(word)
    return ' '.join(result)


# ─────────────────────────────────────────────
#  Анализ ROI
# ─────────────────────────────────────────────

def analyze_roi(roi: np.ndarray) -> dict:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
    h, w = gray.shape
    brightness = float(gray.mean())
    contrast = float(gray.std())

    is_red_bg = False
    if len(roi.shape) == 3:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
        red_pixels = cv2.countNonZero(mask1) + cv2.countNonZero(mask2)
        is_red_bg = red_pixels > (h * w * 0.2)

    return {
        "is_dark":         brightness < 80,
        "is_medium_dark":  80 <= brightness < 130,
        "is_small":        min(h, w) < 100,
        "is_low_contrast": contrast < 30,
        "is_red_bg":       is_red_bg,
        "brightness":      round(brightness, 1),
        "contrast":        round(contrast, 1),
        "size":            (w, h),
    }


# ─────────────────────────────────────────────
#  Предобработка
# ─────────────────────────────────────────────

def _upscale(img: np.ndarray, min_h=220, min_w=350) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(1.0, min_h / h, min_w / w)
    if scale <= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_LANCZOS4)


def get_preprocessing_variants(roi: np.ndarray, info: dict) -> list:
    big = _upscale(roi)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    variants = []

    if info["is_red_bg"]:
        b_channel = big[:, :, 0]
        variants += [
            ("blue_channel", b_channel),
            ("inv_blue",     cv2.bitwise_not(b_channel)),
            ("color",        big),
            ("clahe",        clahe.apply(gray)),
        ]
    elif info["is_dark"]:
        variants += [
            ("inv_clahe", clahe.apply(cv2.bitwise_not(gray))),
            ("inverted",  cv2.bitwise_not(gray)),
            ("color",     big),
            ("clahe",     clahe.apply(gray)),
        ]
    elif info["is_medium_dark"]:
        variants += [
            ("clahe",     clahe.apply(gray)),
            ("color",     big),
            ("inv_clahe", clahe.apply(cv2.bitwise_not(gray))),
            ("sharp",     cv2.filter2D(gray, -1, kernel)),
        ]
    else:
        variants += [
            ("color",     big),
            ("clahe",     clahe.apply(gray)),
            ("sharp",     cv2.filter2D(gray, -1, kernel)),
            ("inv_clahe", clahe.apply(cv2.bitwise_not(gray))),
        ]

    if info["is_low_contrast"]:
        strong_clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(2, 2))
        variants.insert(0, ("strong_clahe", strong_clahe.apply(gray)))

    return variants


# ─────────────────────────────────────────────
#  EasyOCR runner
# ─────────────────────────────────────────────

def _run_ocr(reader, img: np.ndarray, min_conf: float) -> list:
    # EasyOCR принимает BGR или grayscale numpy array
    results = reader.readtext(
        img,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        text_threshold=0.5,
        low_text=0.3,
    )
    words = []
    for item in results:
        if len(item) != 3:
            continue
        bbox, text, conf = item
        text = text.strip()
        if conf >= min_conf and text:
            text = fix_mixed_text(text)
            words.append({
                "text":       text,
                "confidence": round(float(conf), 3),
                "bbox":       bbox,
            })
    return words


def _build_output(words: list, info: dict = None) -> dict:
    full_text = " ".join(w["text"] for w in words)
    avg_conf = sum(w["confidence"] for w in words) / len(words) if words else 0.0
    result = {"text": full_text, "words": words, "confidence": round(avg_conf, 3)}
    if info:
        result["roi_info"] = info
    return result


def _score(out: dict) -> tuple:
    return (len(out.get("words", [])), out.get("confidence", 0.0))


# ─────────────────────────────────────────────
#  Стратегия полос
# ─────────────────────────────────────────────

def _read_with_strips(reader, img: np.ndarray, min_conf: float,
                      n_strips: int = 3, debug: bool = False) -> list:
    """Делит изображение на горизонтальные полосы и читает каждую отдельно."""
    h = img.shape[0]
    strip_h = h // n_strips
    overlap = int(strip_h * 0.2)
    all_words = []

    for i in range(n_strips):
        y1 = max(0, i * strip_h - overlap)
        y2 = min(h, (i + 1) * strip_h + overlap)
        strip = img[y1:y2]
        words = _run_ocr(reader, strip, min_conf)
        if debug and words:
            logger.info(f"    полоса {i}: {[w['text'] for w in words]}")
        all_words.extend(words)

    return all_words


# ─────────────────────────────────────────────
#  Публичный API
# ─────────────────────────────────────────────

def read_text_from_roi(
    roi: np.ndarray,
    min_confidence: float = 0.2,
    preprocess: bool = True,
    debug: bool = False,
) -> dict:
    """
    Читает текст с вырезанного баннера.

    Стратегия:
      1. Пробуем варианты предобработки на целом ROI
      2. Если слов мало — читаем по горизонтальным полосам
      3. Возвращаем лучший результат
    """
    if roi is None or roi.size == 0:
        return {"text": "", "words": [], "confidence": 0.0}

    try:
        reader = get_ocr_reader()

        if not preprocess:
            words = _run_ocr(reader, roi, min_confidence)
            return _build_output(words)

        info = analyze_roi(roi)
        if debug:
            logger.info(
                f"ROI {info['size']} bright={info['brightness']} "
                f"contrast={info['contrast']} dark={info['is_dark']} "
                f"red_bg={info['is_red_bg']} small={info['is_small']}"
            )

        best = {"text": "", "words": [], "confidence": 0.0}

        # Шаг 1: целый ROI с разными предобработками
        for name, img_var in get_preprocessing_variants(roi, info):
            words = _run_ocr(reader, img_var, min_confidence)
            out = _build_output(words, info)

            if debug:
                logger.info(
                    f"  [{name:15s}] '{out['text']}' "
                    f"conf={out['confidence']:.2f} words={len(words)}"
                )

            if _score(out) > _score(best):
                best = out

        # Шаг 2: если мало слов — пробуем полосы
        if len(best.get("words", [])) < 3:
            if debug:
                logger.info("  → Мало слов, пробуем стратегию полос...")

            # Берём первый (лучший) вариант предобработки для полос
            best_variant_img = get_preprocessing_variants(roi, info)[0][1]
            big = _upscale(best_variant_img)

            strip_words = _read_with_strips(reader, big, min_confidence, debug=debug)
            strip_out = _build_output(strip_words, info)

            if debug:
                logger.info(f"  → Полосы: '{strip_out['text']}' words={len(strip_words)}")

            if _score(strip_out) > _score(best):
                best = strip_out

        return best

    except Exception as e:
        logger.error(f"OCR ошибка: {e}")
        return {"text": "", "words": [], "confidence": 0.0, "error": str(e)}
