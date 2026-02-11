InspectAI
===================

Проект: Интеллектуальная система контроля благоустройства городской среды на основе компьютерного зрения.
Цель: детектировать объекты (вывески, баннеры, рекламные конструкции), распознавать текст и классифицировать нарушения.

СОДЕРЖАНИЕ
----------

1. Архитектура проекта
2. Backend
3. ML-модуль
4. Frontend
5. Docker и запуск
6. API
7. Задачи команды
8. Структура репозитория
9. Дальнейшее развитие

ARCHITECTURA PROEKT
------------------
city-vision-control/
|
|-- frontend/              веб-интерфейс
|   |-- index.html
|   |-- css/
|   |-- js/
|
|-- backend/               FastAPI
|   |-- app/
|       |-- main.py       запуск сервиса
|       |-- routes/       endpoint’ы
|       |-- services/     StubDetector, YOLODetector
|   |-- requirements.txt
|
|-- ml/                    модели
|   |-- detectors/
|       |-- yolo_detector.py
|   |-- training/
|   |-- inference/
|   |-- requirements.txt
|
|-- data/                  датасеты (.gitkeep)
|-- weights/               веса моделей (.gitkeep)
|
|-- infra/                 Docker и окружение
|   |-- docker-compose.yml
|   |-- Dockerfile.backend
|
|-- API_CONTRACT.md        описание API
|-- README.md
|-- .gitignore

BACKEND
-------
- Используется FastAPI для приёма изображений и предоставления API.
- Заглушка StubDetector возвращает JSON с ключами: boxes, scores, classes, inference_time.
- Endpoint /detect готов к интеграции с YOLODetector после разработки ML-модуля.

Установка backend:
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

ML-МОДУЛЬ
----------
- Основной класс: YOLODetector в ml/detectors/yolo_detector.py
- Метод predict(image) возвращает JSON:
  {
    "boxes": [[x1, y1, x2, y2]],
    "scores": [0.95],
    "classes": ["sign"],
    "inference_time": 0.12
  }

Пример запуска:
if __name__ == "__main__":
    from detectors.yolo_detector import YOLODetector
    detector = YOLODetector("weights/yolo.pth")
    result = detector.predict("data/test.jpg")
    print(result)

Установка зависимостей ML:
cd ml
pip install -r requirements.txt

FRONTEND
--------
- Веб-интерфейс позволяет загружать изображения и получать JSON с результатами.
- Папки: css/, js/
- Подключение к backend: http://localhost:8000/detect/

DOCKER И ЗАПУСК
---------------
- Для воспроизводимости используется Docker и docker-compose.
- Команды:
docker compose -f infra/docker-compose.yml up --build
- После запуска сервис доступен на http://localhost:8000

API
---
Endpoint: /detect (POST)
Формат запроса: изображение (multipart/form-data)
Формат ответа:
{
  "boxes": [[x1, y1, x2, y2]],
  "scores": [0.95],
  "classes": ["sign"],
  "inference_time": 0.12
}

ЗАДАЧИ КОМАНДЫ
---------------
ML-инженер:
- Реализовать YOLODetector
- Поддерживать формат JSON, замер времени инференса, обработку ошибок
- Дедлайн: 16.02

Исследователь:
- Собирать нормативные документы по вывескам, баннерам, рекламным конструкциям
- Создавать структурированную таблицу для классификации нарушений
- Дедлайн первичная таблица: 15.02
- Поддержка ML и уточнение кейсов: до 20.02

СТРУКТУРА РЕПОЗИТОРИЯ
--------------------
- frontend/ — веб-интерфейс
- backend/app/ — FastAPI сервисы
- ml/ — модели, инференс, обучение
- data/ — исходные данные и тестовые изображения
- weights/ — сохранённые веса моделей
- infra/ — Docker и конфигурации окружения
- API_CONTRACT.md — контракт API между frontend, backend и ML

ДАЛЬНЕЙШЕЕ РАЗВИТИЕ
-------------------
- Подключение новых моделей (TrOCR, EfficientNet, SegFormer)
- Улучшение frontend с визуализацией сегментации и bounding boxes
- Проведение экспериментов по сравнению моделей по точности и скорости
- Автоматизация дообучения моделей на новых данных


