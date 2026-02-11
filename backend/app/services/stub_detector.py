import time


class StubDetector:
    """
    Временный детектор до интеграции реальной YOLO-модели.
    Даст нам сделать тест
    """

    def predict(self, image_bytes: bytes) -> dict:
        start = time.time()

        # Фейковый результат
        boxes = [[50, 50, 200, 200]]
        scores = [0.9]
        classes = ["stub_object"]

        inference_time = round(time.time() - start, 4)

        return {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "inference_time": inference_time,
        }
