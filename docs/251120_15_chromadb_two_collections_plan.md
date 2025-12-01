# ChromaDB Architecture: Two Collections Plan

**Дата:** 2025-11-20 15:00 (обновлено 2025-11-20 17:30)
**Задача:** Реализация двух коллекций ChromaDB для хранения статусов обработки и сегментов видео

---

## Ключевая идея архитектуры

### videos_config.json — источник правды
- Хранит список видео с `name`, `path`, `language`, `description`
- **НЕ удаляется и не модифицируется автоматически**
- Опционально поле `force_reprocess: true` для принудительной переобработки
- Используется как стартовая точка для Phase 1

### ChromaDB — центр управления обработкой
- **Коллекция `video_files`**: метаданные + статусы обработки (одна запись = одно видео)
- **Коллекция `video_segments`**: описания сегментов для поиска (много записей = много сегментов)

### Workflow: умные скрипты, которые можно запускать сколько угодно раз
- **Phase 1, 2, Load** можно запускать повторно — они сами определяют что нужно обработать
- **Автоматическое восстановление после сбоев** через timeout-based failure detection
- Таймаут на обработку = `video_duration * 2`

---

## Архитектура ChromaDB

### Коллекция 1: `video_files` (метаданные файлов + статусы)

**Назначение:** Одна запись = один видеофайл. Хранит статусы обработки и общую информацию.

**Структура документа:**
```json
{
  "id": "news",
  "document": "summary + keywords (для будущего поиска по всему видео)",
  "metadata": {
    "video_name": "news",
    "video_path": "video/news.mp4",
    "language": "kk",

    // Статусы обработки фаз
    "phase1_status": "completed",  // pending | processing | completed
    "phase1_start_time": "2025-11-20T14:30:00",
    "phase1_segments_created": 60,

    "phase2_status": "completed",  // pending | processing | completed
    "phase2_start_time": "2025-11-20T15:45:00",
    "phase2_segments_analyzed": 60,

    "db_load_status": "completed",  // pending | processing | completed
    "db_load_start_time": "2025-11-20T16:00:00",
    "db_segments_loaded": 60,

    // Метаданные видео
    "video_duration": 3000.0,  // используется для расчета таймаута
    "total_segments": 60,
    "width": 1920,
    "height": 1080,
    "fps": 25.0,

    // Сводка по всему видео (заполнит будущий обработчик Phase 4)
    "summary": "Общее описание всего ролика",
    "keywords": ["новости", "Казахстан", "погода"],
    "content_type": "новости",
    "created_at": "2025-11-20T14:00:00",
    "updated_at": "2025-11-20T16:00:00"
  }
}
```

**Возможные статусы:**
- `pending` - не начато
- `processing` - в процессе (проверяется таймаут)
- `completed` - завершено

**Убрали статус `failed`:** вместо него используется timeout-based detection:
- Если `status == "processing"` и `(current_time - start_time) > (video_duration * 2)` → считается failed
- Автоматически переобрабатывается при следующем запуске

---

### Коллекция 2: `video_segments` (сегменты видео)

**Назначение:** Много записей для каждого видео. Хранит описания сегментов для семантического поиска.

**Структура документа:**
```json
{
  "id": "news_seg_0",
  "document": "description + dialogue_translation + keywords (текст для эмбеддингов)",
  "metadata": {
    "video_name": "news",
    "video_path": "video/news.mp4",
    "segment_index": 0,
    "start_time": 0.0,
    "end_time": 50.0,
    "duration": 50.0,

    // Данные из Phase 2 VLLM analysis
    "description": "Ведущий новостей за столом...",
    "dialogue_translation": "Добрый вечер...",
    "keywords": ["новости", "ведущий", "студия"],
    "content_type": "новости",
    "mood_atmosphere": "официальная",
    "confidence": "высокая",
    "scene_change": false
  }
}
```

---

## Пайплайн обработки

### Phase 1: Whisper + Frame Extraction

**Что делает:**
1. Читает `videos_config.json`
2. **ДЛЯ КАЖДОГО ВИДЕО проверяет ChromaDB:**
   - Записи нет → создает с `phase1_status="processing"`, записывает `phase1_start_time` и `video_duration`
   - Запись есть, `phase1_status="completed"` → **пропускает**
   - Запись есть, `phase1_status="processing"` + таймаут истек → **переобрабатывает** (обновляет `phase1_start_time`)
3. Обрабатывает видео:
   - Извлекает аудио сегменты (50 сек с overlap 15 сек)
   - Транскрибирует через Whisper с указанным языком
   - Извлекает кадры (6 кадров на сегмент)
   - Сохраняет в `{video_name}_phase1_data.pkl`
4. Обновляет ChromaDB: `phase1_status="completed"`

**ChromaDB операции Phase 1:**
```python
# В начале process_video_phase1():

# Проверить существующий статус
video_status = db_manager.get_video_status(video_name)

if video_status:
    # Запись существует
    if video_status.get("phase1_status") == "completed":
        print(f"✓ Phase 1 already completed for {video_name}, skipping...")
        return

    if video_status.get("phase1_status") == "processing":
        # Проверить таймаут
        start_time = datetime.fromisoformat(video_status["phase1_start_time"])
        video_duration = video_status["video_duration"]
        timeout = video_duration * 2

        if (datetime.now() - start_time).total_seconds() < timeout:
            print(f"⚠️ Phase 1 still processing for {video_name}, skipping...")
            return
        else:
            print(f"🔄 Phase 1 timeout detected for {video_name}, reprocessing...")
else:
    # Первый запуск - создать запись
    print(f"➕ Creating new video_files record for {video_name}...")

# Обновить статус на "processing"
db_manager.create_or_update_video_file(
    video_name=video_name,
    video_path=video_path,
    language=language,
    video_duration=video_duration,
    total_segments=len(segments),
    width=video_width,
    height=video_height,
    fps=fps,
    phase1_status="processing",
    phase1_start_time=datetime.now().isoformat()
)

# ... обработка видео ...

# В конце process_video_phase1():
# Обновить статус на "completed"
db_manager.update_video_status(
    video_name=video_name,
    phase1_status="completed",
    phase1_segments_created=len(segment_data)
)
```

**Изменения в коде:**
- ✅ `phase1_whisper_frames.py` уже доработан (читает `videos_config.json`, поддерживает language)
- ❌ **TODO:** Добавить логику проверки статусов в начале `process_video_phase1()`
- ❌ **TODO:** Добавить вызов `db_manager.create_or_update_video_file()` для установки `"processing"`
- ❌ **TODO:** Добавить вызов `db_manager.update_video_status()` для установки `"completed"`

---

### Phase 2: VLLM Analysis

**Что делает:**
1. **НЕ читает videos_config.json**
2. **Читает ChromaDB:** ищет видео где `phase1_status="completed" AND phase2_status IN ("pending", "processing")`
3. Для каждого такого видео:
   - Проверяет таймаут если `phase2_status="processing"`
   - Обновляет статус на `"processing"`, записывает `phase2_start_time`
   - Загружает `{video_name}_phase1_data.pkl`
   - Анализирует сегменты через VLLM (Qwen 2.5 VL)
   - Сохраняет в `{video_name}_phase2_vllm_analysis.json`
   - Обновляет статус на `"completed"`

**ChromaDB операции Phase 2:**
```python
# В начале main():

# Получить список видео для обработки
videos_to_process = db_manager.list_videos(
    status_filter={
        "phase1_status": "completed",
        "phase2_status": ["pending", "processing"]
    }
)

for video_info in videos_to_process:
    video_name = video_info["video_name"]

    # Проверить таймаут если processing
    if video_info.get("phase2_status") == "processing":
        start_time = datetime.fromisoformat(video_info["phase2_start_time"])
        video_duration = video_info["video_duration"]
        timeout = video_duration * 2

        if (datetime.now() - start_time).total_seconds() < timeout:
            print(f"⚠️ Phase 2 still processing for {video_name}, skipping...")
            continue
        else:
            print(f"🔄 Phase 2 timeout detected for {video_name}, reprocessing...")

    # Установить статус "processing"
    db_manager.update_video_status(
        video_name=video_name,
        phase2_status="processing",
        phase2_start_time=datetime.now().isoformat()
    )

    # ... обработка через VLLM ...

    # Установить статус "completed"
    db_manager.update_video_status(
        video_name=video_name,
        phase2_status="completed",
        phase2_segments_analyzed=len(results)
    )
```

**Изменения в коде:**
- ❌ **TODO:** Изменить `main()` чтобы читал ChromaDB вместо videos_config.json
- ❌ **TODO:** Добавить логику проверки таймаутов
- ❌ **TODO:** Добавить вызов `db_manager.update_video_status()` в начале (processing) и конце (completed)

---

### Phase 3: Load Data to ChromaDB

**Что делает:**
1. **НЕ читает videos_config.json**
2. **Читает ChromaDB:** ищет видео где `phase2_status="completed" AND db_load_status IN ("pending", "processing")`
3. Для каждого такого видео:
   - Проверяет таймаут если `db_load_status="processing"`
   - Обновляет статус на `"processing"`, записывает `db_load_start_time`
   - Загружает `{video_name}_phase2_vllm_analysis.json`
   - Записывает сегменты в коллекцию `video_segments`
   - Обновляет статус на `"completed"`

**ChromaDB операции Phase 3:**
```python
# В начале main():

# Получить список видео для загрузки
videos_to_load = db_manager.list_videos(
    status_filter={
        "phase2_status": "completed",
        "db_load_status": ["pending", "processing"]
    }
)

for video_info in videos_to_load:
    video_name = video_info["video_name"]

    # Проверить таймаут если processing
    if video_info.get("db_load_status") == "processing":
        start_time = datetime.fromisoformat(video_info["db_load_start_time"])
        video_duration = video_info["video_duration"]
        timeout = video_duration * 2

        if (datetime.now() - start_time).total_seconds() < timeout:
            print(f"⚠️ DB load still processing for {video_name}, skipping...")
            continue
        else:
            print(f"🔄 DB load timeout detected for {video_name}, reprocessing...")

    # Установить статус "processing"
    db_manager.update_video_status(
        video_name=video_name,
        db_load_status="processing",
        db_load_start_time=datetime.now().isoformat()
    )

    # Загрузить сегменты в video_segments
    added_count = db_manager.add_segments(
        video_path=video_info["video_path"],
        segments=phase2_data['segments']
    )

    # Установить статус "completed"
    db_manager.update_video_status(
        video_name=video_name,
        db_load_status="completed",
        db_segments_loaded=added_count
    )
```

**Изменения в коде:**
- ❌ **TODO:** Изменить `main()` чтобы читал ChromaDB вместо videos_config.json
- ❌ **TODO:** Добавить логику проверки таймаутов
- ❌ **TODO:** Добавить вызов `db_manager.update_video_status()` в начале (processing) и конце (completed)

---

### Phase 4 (Будущая): Video Summary Generator

**Что будет делать:**
1. Читает все сегменты видео из ChromaDB
2. Создает общую сводку через LLM (суммаризация всех сегментов)
3. Генерирует keywords для всего видео
4. Обновляет поля `summary` и `keywords` в коллекции `video_files`

**Это позже!** Пока фокусируемся на Phase 1-3.

---

## Доработка db_manager.py

### Новые методы для работы с двумя коллекциями:

```python
class ChromaDBManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        # Две коллекции
        self.video_files_collection = self.client.get_or_create_collection(
            name="video_files",
            metadata={"description": "Video files metadata and processing status"}
        )

        self.video_segments_collection = self.client.get_or_create_collection(
            name="video_segments",
            metadata={"description": "Video segment descriptions from VLLM analysis"}
        )

    # === Методы для video_files ===

    def create_or_update_video_file(
        self,
        video_name: str,
        video_path: str,
        language: str,
        duration: float,
        total_segments: int,
        width: int,
        height: int,
        fps: float,
        **status_fields
    ) -> None:
        """
        Создать или обновить запись о видеофайле

        status_fields может включать:
        - phase1_status, phase1_date, phase1_segments_created
        - phase2_status, phase2_date, phase2_segments_analyzed
        - db_load_status, db_load_date, db_segments_loaded
        """

    def get_video_status(self, video_name: str) -> Dict[str, Any]:
        """Получить статус обработки видео"""

    def update_video_status(self, video_name: str, **status_fields) -> None:
        """Обновить статусы обработки видео"""

    def list_videos(self, status_filter: Dict = None) -> List[Dict]:
        """
        Получить список видео с фильтрацией

        Примеры:
        - status_filter={"phase1_status": "completed"} - только с завершенной Phase 1
        - status_filter={"db_load_status": "pending"} - не загруженные в БД
        """

    # === Методы для video_segments (уже есть) ===

    def add_segments(self, video_path: str, segments: List[Dict]) -> int:
        """Добавить сегменты видео (уже реализовано)"""

    def search(self, query: str, limit: int = 10, **filters) -> List[Dict]:
        """Поиск по сегментам (уже реализовано)"""

    def delete_video_segments(self, video_name: str) -> int:
        """Удалить все сегменты видео (уже реализовано)"""
```

---

## Текущий статус файлов

### ✅ Готовые файлы:

1. **videos_config.json** - конфигурация видео
```json
{
  "videos": [
    {"name": "news", "path": "video/news.mp4", "language": "kk"},
    {"name": "SovKz", "path": "video/SovKz.mp4", "language": "ru"},
    {"name": "AlmaAta", "path": "video/AlmaAta.mp4", "language": "ru"}
  ]
}
```

2. **phase1_whisper_frames.py** - ✅ доработан
   - Читает `videos_config.json`
   - Поддерживает параметр `language` для Whisper
   - Обрабатывает все 3 видео

3. **db_manager.py** - ✅ базовая версия есть
   - Работает с одной коллекцией `video_segments`
   - Есть методы: `add_segments()`, `search()`, `delete_video_segments()`

4. **load_data_to_chroma.py** - ✅ базовая версия есть
   - Загружает Phase 2 JSON в ChromaDB
   - Работает только с одной коллекцией

5. **test_search.py** - ✅ готов
   - Тестирует поиск по ChromaDB
   - Интерактивный режим

---

## TODO: Что нужно доработать

### 1. db_manager.py - добавить вторую коллекцию

- [ ] Инициализировать `video_files_collection`
- [ ] Реализовать `create_or_update_video_file()`
- [ ] Реализовать `get_video_status()`
- [ ] Реализовать `update_video_status()`
- [ ] Реализовать `list_videos()`
- [ ] Обновить `__init__` для работы с двумя коллекциями

### 2. phase1_whisper_frames.py - интеграция с ChromaDB

- [ ] Импортировать `ChromaDBManager`
- [ ] В конце `process_video_phase1()` добавить:
  ```python
  db_manager.create_or_update_video_file(
      video_name=video_name,
      video_path=video_path,
      language=language,
      duration=video_duration,
      total_segments=len(segments),
      width=video_width,
      height=video_height,
      fps=fps,
      phase1_status="completed",
      phase1_date=datetime.now().isoformat(),
      phase1_segments_created=len(segments)
  )
  ```

### 3. phase2_vllm_analysis.py - читать конфиг + ChromaDB

- [ ] Добавить `load_videos_config()` из Phase 1
- [ ] Изменить `main()` чтобы читал `videos_config.json` вместо хардкода `video_names = ["news"]`
- [ ] В конце `process_phase2_analysis()` добавить:
  ```python
  db_manager.update_video_status(
      video_name=video_name,
      phase2_status="completed",
      phase2_date=datetime.now().isoformat(),
      phase2_segments_analyzed=len(results)
  )
  ```

### 4. load_data_to_chroma.py - читать конфиг + проверка статусов

- [ ] Добавить `load_videos_config()`
- [ ] Изменить `main()` чтобы читал конфиг
- [ ] В `load_video_to_chroma()` добавить проверку:
  ```python
  # Проверить что Phase 2 завершена
  video_status = db_manager.get_video_status(video_name)
  if video_status.get("phase2_status") != "completed":
      logger.warning(f"Phase 2 not completed for {video_name}, skipping...")
      return False
  ```
- [ ] После успешной загрузки обновить статус:
  ```python
  db_manager.update_video_status(
      video_name=video_name,
      db_load_status="completed",
      db_load_date=datetime.now().isoformat(),
      db_segments_loaded=added_count
  )
  ```

---

## Порядок выполнения доработок

1. **Сначала:** Доработать `db_manager.py` (добавить вторую коллекцию и методы)
2. **Потом:** Доработать `phase1_whisper_frames.py` (интеграция с ChromaDB)
3. **Потом:** Доработать `phase2_vllm_analysis.py` (конфиг + статусы)
4. **Потом:** Доработать `load_data_to_chroma.py` (конфиг + проверка статусов)
5. **Тестирование:** Запустить полный пайплайн Phase 1 → Phase 2 → Load → Search

---

## Тестовый сценарий после доработок

```bash
# 1. Phase 1 - обработка всех видео
python phase1_whisper_frames.py
# Результат:
# - news_phase1_data.pkl, SovKz_phase1_data.pkl, AlmaAta_phase1_data.pkl
# - ChromaDB video_files: 3 записи со статусом phase1_status="completed"

# 2. Phase 2 - VLLM анализ всех видео
python phase2_vllm_analysis.py
# Результат:
# - news_phase2_vllm_analysis.json, SovKz_phase2_vllm_analysis.json, AlmaAta_phase2_vllm_analysis.json
# - ChromaDB video_files: обновлены статусы phase2_status="completed"

# 3. Load - загрузка в ChromaDB
python load_data_to_chroma.py
# Результат:
# - ChromaDB video_segments: ~180 сегментов (60 * 3 видео)
# - ChromaDB video_files: обновлены статусы db_load_status="completed"

# 4. Search - тестовый поиск
python test_search.py
# Результат: поиск работает по всем видео
```

---

## Проверка статусов через ChromaDB

```python
from db_manager import ChromaDBManager

db = ChromaDBManager()

# Получить все видео
videos = db.list_videos()
print(f"Total videos: {len(videos)}")

# Найти видео где Phase 2 не завершена
pending = db.list_videos(status_filter={"phase2_status": "pending"})
print(f"Videos pending Phase 2: {len(pending)}")

# Проверить конкретное видео
status = db.get_video_status("news")
print(f"news status: {status}")
```

---

## Будущие улучшения (Phase 4+)

1. **Video Summary Generator:**
   - Читает все сегменты видео
   - Суммаризация через LLM
   - Обновляет `summary` и `keywords` в `video_files`

2. **Веб-интерфейс (VIDEO-12, VIDEO-13):**
   - FastAPI backend
   - Поиск по `video_segments`
   - Показ метаданных из `video_files`

3. **Статистика и мониторинг:**
   - Dashboard со статусами обработки
   - Сколько видео в каждой фазе
   - Ошибки обработки

---

**Путь к документу:** E:\Projects\Quantum\Video_tagging_db\docs\251120_15_chromadb_two_collections_plan.md
