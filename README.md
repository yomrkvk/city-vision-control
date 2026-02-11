# City Vision Control

Система интеллектуального контроля городской среды на основе компьютерного зрения.

## Архитектура

Проект организован как монорепозиторий:

- frontend — веб-интерфейс загрузки изображений
- backend — FastAPI API
- ml — модели и обучение
- infra — Docker и окружение

## Быстрый запуск

### Через Docker

docker compose -f infra/docker-compose.yml up --build

API будет доступно:

http://localhost:8000/docs

### Локально

cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

## Endpoint

POST /detect/

Принимает изображение и возвращает:

{
  "boxes": [],
  "scores": [],
  "classes": [],
  "inference_time": 0.0
}

Сейчас используется StubDetector  
(временная заглушка до интеграции YOLOv8).

## Следующие шаги

- Реализация ml/detectors/yolo_detector.py
- Интеграция YOLO в backend
- Отрисовка bounding boxes на frontend
