# VIDEO-20: Adaptive det_size for Face Detection

**Дата:** 2025-11-29
**Статус:** Завершено

---

## Проблема

InsightFace детектор работает на фиксированном разрешении (det_size).
При det_size=(640,640) мелкие лица на 4K видео не детектятся:
- Лицо 50px в 4K → 8px после ресайза до 640 → не видно детектору

---

## Решение

Адаптивный det_size на основе разрешения видео:
- Использовать оригинальное разрешение до max_det_size
- Для больших разрешений - пропорционально уменьшать
- Округлять до кратного 32 (требование InsightFace)

---

## Реализация

### Новая функция: get_optimal_det_size()

```python
def get_optimal_det_size(frame_width: int, frame_height: int, max_size: int = 1920) -> Tuple[int, int]:
    # Если меньше max_size - используем оригинал
    if frame_width <= max_size and frame_height <= max_size:
        det_w, det_h = frame_width, frame_height
    else:
        # Пропорционально уменьшаем
        scale = max_size / max(frame_width, frame_height)
        det_w = int(frame_width * scale)
        det_h = int(frame_height * scale)

    # Округление до 32
    det_w = (det_w // 32) * 32
    det_h = (det_h // 32) * 32

    # Минимум 128
    det_w = max(128, det_w)
    det_h = max(128, det_h)

    return (det_w, det_h)
```

### Модификация initialize_face_models()

```python
def initialize_face_models(det_size: Optional[Tuple[int, int]] = None):
    if det_size is None:
        det_size = (640, 640)

    # Пропуск реинициализации если тот же размер
    if _face_analyzer is not None and _current_det_size == det_size:
        return

    _face_analyzer.prepare(ctx_id=0, det_size=det_size)
```

---

## Примеры преобразований

| Входное разрешение | det_size (max=1920) |
|--------------------|---------------------|
| 640x480            | 640x480             |
| 1280x720           | 1280x704            |
| 1920x1080          | 1920x1056           |
| 3840x2160 (4K)     | 1920x1056           |
| 2560x1440          | 1920x1056           |

---

## Оценка VRAM

| det_size    | VRAM     |
|-------------|----------|
| 640x640     | ~1.5-2 GB|
| 1280x1280   | ~3-4 GB  |
| 1920x1080   | ~4-5 GB  |

RTX 4060 Ti (8-16GB) - Full HD работает без проблем.

---

## Результаты тестирования

**Видео:** news.mp4 (854x470)

| Метрика | 640x640 (было) | 832x448 (адаптивно) |
|---------|----------------|---------------------|
| Лиц     | 114            | 116 (+2)            |
| Персон  | 15             | 17 (+2)             |
| FPS     | 167            | 148                 |

---

## Изменённые файлы

1. `face_processor_v2.py`:
   - Добавлена `get_optimal_det_size()`
   - Модифицирована `initialize_face_models(det_size)`

2. `phase3_face_detection.py`:
   - Добавлен параметр `max_det_size` (default: 1920)
   - Автоопределение det_size из разрешения видео

---

**Путь:** E:\Projects\Quantum\Video_tagging_db\docs\251129_18_adaptive_det_size_spec.md
