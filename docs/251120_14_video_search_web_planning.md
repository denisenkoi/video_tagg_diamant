# Video Search Web Interface - Planning

**Дата:** 2025-11-20 14:00
**Проект:** VIDEO (Video_tagging)
**Цель:** Создать веб-интерфейс для семантического поиска видео по описаниям от VLLM

---

## Структура эпиков

### EPIC 1: VIDEO-1 - Векторная БД + поиск
**Цель:** Хранение и семантический поиск описаний видео через ChromaDB

**Задачи:**
- ✅ VIDEO-4: Design vector database structure
- ✅ VIDEO-5: Implement data loading into vector database
- VIDEO-3: Create full video tagging pipeline
- VIDEO-10: Implement ChromaDB integration module
- VIDEO-11: Create semantic search API module

**Технологии:**
- ChromaDB (векторная БД)
- sentence-transformers (embeddings: paraphrase-multilingual-MiniLM-L12-v2)
- Python integration modules

---

### EPIC 2: VIDEO-9 - Веб-интерфейс
**Цель:** Веб-морда для поиска и просмотра видео с таймингами

**Задачи:**
- VIDEO-12: Create FastAPI backend for video search
- VIDEO-13: Create web UI for video search
- VIDEO-14: Create Ansible deployment for video search web
- VIDEO-15: Test deployment and create user documentation

**Технологии:**
- FastAPI (API backend)
- Nginx + systemd (деплой)
- HTML/JS + Jinja2 (frontend)
- Ansible (автоматизация деплоя)
- rsync (синхронизация видео на VPS)

---

## Архитектура системы

```
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 1: Whisper + Frames                 │
│  phase1_whisper_frames.py → news_phase1_data.pkl            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 2: VLLM Analysis                    │
│  phase2_vllm_analysis.py → news_phase2_vllm_analysis.json   │
│  { segments: [{description, keywords, timestamps}] }        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   ChromaDB Vector Storage                   │
│  db_manager.py: загрузка описаний в векторную БД           │
│  - Embeddings: paraphrase-multilingual-MiniLM-L12-v2        │
│  - Metadata: video_name, start_time, end_time, keywords     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Semantic Search Engine                    │
│  search_engine.py: поиск по текстовым запросам              │
│  - Векторный поиск по similarity                            │
│  - Фильтры: дата, длительность, confidence                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                        │
│  GET /api/search?q=авария на дороге                         │
│  GET /api/video/{name}/segments                             │
│  GET /api/video/{name}/download                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Web Interface                          │
│  - Форма поиска                                             │
│  - Результаты с карточками (описание + ключевые слова)     │
│  - Видеоплеер с навигацией по таймингам                    │
│  - Подсветка найденных сегментов                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    VPS Deployment (Ansible)                 │
│  - Nginx reverse proxy (80/443)                             │
│  - systemd service (video-web)                              │
│  - rsync для синхронизации видео                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Детали компонентов

### 1. ChromaDB Integration (VIDEO-10)

**Файл:** `db_manager.py`

**Класс:** `ChromaDBManager`

**Методы:**
```python
class ChromaDBManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Инициализация ChromaDB клиента"""

    def add_segments(self, video_name: str, segments: List[Dict]) -> int:
        """
        Загрузить сегменты видео в ChromaDB

        Args:
            video_name: имя видео (без расширения)
            segments: список сегментов из phase2_vllm_analysis.json

        Returns:
            количество добавленных сегментов
        """

    def search_segments(self, query: str, limit: int = 10,
                       filters: Dict = None) -> List[Dict]:
        """
        Поиск сегментов по текстовому запросу

        Args:
            query: текст запроса (русский/английский)
            limit: максимум результатов
            filters: {video_name, start_time_min, start_time_max, confidence}

        Returns:
            список найденных сегментов с метаданными
        """

    def get_segment_by_id(self, segment_id: str) -> Dict:
        """Получить сегмент по ID"""

    def delete_video_segments(self, video_name: str) -> int:
        """Удалить все сегменты видео"""
```

**Данные для хранения:**
```python
# Из phase2_vllm_analysis.json
segment = {
    "segment_index": 0,
    "start_time": 0.0,
    "end_time": 50.0,
    "duration": 50.0,
    "transcript": "текст из Whisper",
    "analysis": {
        "description": "подробное описание",
        "dialogue_translation": "перевод речи",
        "keywords": ["ключевые", "слова"],
        "content_type": "тип контента",
        "mood_atmosphere": "настроение",
        "scene_change": true,
        "confidence": "высокая"
    }
}

# ChromaDB запись
{
    "id": f"{video_name}_seg_{segment_index}",
    "document": f"{description} {dialogue_translation} {' '.join(keywords)}",
    "metadata": {
        "video_name": "news",
        "segment_index": 0,
        "start_time": 0.0,
        "end_time": 50.0,
        "duration": 50.0,
        "keywords": ["ключевые", "слова"],
        "confidence": "высокая",
        "content_type": "тип контента"
    }
}
```

---

### 2. Semantic Search Engine (VIDEO-11)

**Файл:** `search_engine.py`

**Функции:**
```python
def search_videos(query: str,
                 limit: int = 10,
                 video_filter: str = None,
                 confidence_filter: str = None) -> List[Dict]:
    """
    Семантический поиск видео

    Args:
        query: текст запроса
        limit: количество результатов
        video_filter: фильтр по имени видео
        confidence_filter: "высокая" | "средняя" | "низкая"

    Returns:
        [
            {
                "video_name": "news",
                "segment_index": 5,
                "start_time": 250.0,
                "end_time": 300.0,
                "description": "описание сегмента",
                "keywords": ["ключ1", "ключ2"],
                "confidence": "высокая",
                "relevance_score": 0.89
            }
        ]
    """

def get_video_timeline(video_name: str) -> List[Dict]:
    """Получить все сегменты видео по таймлайну"""

def highlight_search_terms(text: str, query: str) -> str:
    """Подсветить найденные термины в тексте"""
```

---

### 3. FastAPI Backend (VIDEO-12)

**Структура:**
```
web_app/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings from .env
│   ├── api/
│   │   ├── __init__.py
│   │   ├── search.py        # Search endpoints
│   │   ├── videos.py        # Video endpoints
│   │   └── health.py        # Health check
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html       # Search page
│   │   └── video.html       # Video player
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           ├── search.js
│           └── player.js
├── .env                     # Config
├── requirements.txt
└── README.md
```

**API Endpoints:**

```python
# GET /api/search
# Query params: q, limit, video_filter, confidence_filter
# Response:
{
    "query": "авария на дороге",
    "total_results": 5,
    "results": [
        {
            "video_name": "news",
            "segment_index": 12,
            "start_time": 600.0,
            "end_time": 650.0,
            "description": "описание...",
            "keywords": ["авария", "дорога"],
            "confidence": "высокая",
            "relevance_score": 0.92
        }
    ]
}

# GET /api/video/{video_name}/segments
# Response: список всех сегментов видео

# GET /api/video/{video_name}/download
# Response: прямая ссылка на скачивание файла

# GET /health
# Response: {"status": "healthy", "chromadb": "connected"}
```

---

### 4. Web UI (VIDEO-13)

**Главная страница (index.html):**
- Форма поиска с текстовым полем
- Фильтры: имя видео, confidence
- Кнопка "Поиск"

**Результаты поиска:**
```html
<div class="search-results">
    <div class="result-card">
        <h3>news - Сегмент 12 (10:00 - 10:50)</h3>
        <p class="description">Описание происходящего в сегменте...</p>
        <div class="keywords">
            <span class="badge">авария</span>
            <span class="badge">дорога</span>
        </div>
        <div class="meta">
            <span>Релевантность: 92%</span>
            <span>Confidence: высокая</span>
        </div>
        <button onclick="openVideo('news', 600)">Просмотр</button>
    </div>
</div>
```

**Видеоплеер (video.html):**
- HTML5 `<video>` элемент
- Таймлайн с отметками сегментов
- Навигация по сегментам
- Кнопка "Скачать видео"

**JavaScript функциональность:**
```javascript
// search.js
async function performSearch(query, filters) {
    const response = await fetch(`/api/search?q=${query}&...`);
    const data = await response.json();
    renderResults(data.results);
}

// player.js
function openVideo(videoName, startTime) {
    // Открыть видеоплеер и перемотать на startTime
    videoElement.currentTime = startTime;
    videoElement.play();
}

function highlightSegments(segments) {
    // Подсветить найденные сегменты на таймлайне
}
```

---

### 5. Ansible Deployment (VIDEO-14)

**Базис:** `anpr_deploy/` (FastAPI + Nginx + systemd)

**Структура:**
```
video_web_deploy/
├── inventory.ini            # VPS credentials
├── init_video_web.yml       # Инфраструктура (один раз)
├── deploy_video_web.yml     # Деплой приложения
├── sync_videos.sh           # Синхронизация видео на VPS
├── templates/
│   ├── .env.j2
│   ├── video-web.service.j2
│   ├── nginx-video.conf.j2
│   └── chroma_db.conf.j2
└── files/
    └── web_app/             # FastAPI приложение
```

**init_video_web.yml:**
1. Обновление системы
2. Установка: Python 3.12, Nginx, rsync
3. Создание `/opt/video_web/`
4. Настройка firewall (UFW)

**deploy_video_web.yml:**
1. Копирование FastAPI приложения
2. Создание venv + установка зависимостей
3. Настройка ChromaDB persistence directory
4. Конфигурация systemd service
5. Настройка Nginx reverse proxy
6. Запуск и проверка

**sync_videos.sh:**
```bash
#!/bin/bash
# Синхронизация видео файлов на VPS
rsync -avz --progress \
    /mnt/e/Projects/Quantum/Video_tagging_db/video/ \
    root@VPS_IP:/opt/video_web/videos/
```

**Nginx конфигурация:**
```nginx
server {
    listen 80;
    server_name video.example.com;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
    }

    location /videos/ {
        alias /opt/video_web/videos/;
        autoindex on;
    }
}
```

---

## Зависимости и порядок выполнения

### EPIC 1: Векторная БД + поиск

1. **VIDEO-4**: Design vector database structure ✅
   - Определить схему данных для ChromaDB
   - Выбрать модель embeddings
   - Спроектировать metadata structure

2. **VIDEO-10**: Implement ChromaDB integration
   - Создать `db_manager.py`
   - Реализовать add_segments, search_segments
   - Тесты с Phase 2 данными

3. **VIDEO-5**: Implement data loading into vector database
   - Скрипт для загрузки Phase 2 JSON в ChromaDB
   - Валидация данных
   - Логирование процесса

4. **VIDEO-11**: Create semantic search API module
   - Создать `search_engine.py`
   - Реализовать фильтры и ранжирование
   - Тестовый скрипт для проверки

5. **VIDEO-3**: Create full video tagging pipeline
   - Интеграция Phase 1 + Phase 2 + ChromaDB
   - Единый скрипт для обработки видео от начала до конца

---

### EPIC 2: Веб-интерфейс

6. **VIDEO-12**: Create FastAPI backend
   - Создать FastAPI приложение
   - API endpoints для поиска и видео
   - Интеграция с search_engine.py

7. **VIDEO-13**: Create web UI
   - HTML/CSS/JS интерфейс
   - Форма поиска + результаты
   - Видеоплеер с таймингами

8. **VIDEO-14**: Create Ansible deployment
   - init_video_web.yml
   - deploy_video_web.yml
   - Nginx + systemd конфигурация

9. **VIDEO-15**: Test deployment
   - Деплой на VPS
   - Тестирование всех функций
   - Документация пользователя

---

## Технологический стек

### Backend:
- **Python 3.12**
- **FastAPI** - веб-фреймворк
- **ChromaDB** - векторная БД
- **sentence-transformers** - embeddings
- **uvicorn** - ASGI сервер

### Frontend:
- **HTML5** + **CSS3**
- **JavaScript** (vanilla)
- **Jinja2** - templating
- **Video.js** или HTML5 video

### Deployment:
- **Ansible** - автоматизация
- **Nginx** - reverse proxy
- **systemd** - управление сервисами
- **rsync** - синхронизация файлов
- **UFW** - firewall

### Monitoring:
- **Health endpoints** - статус сервисов
- **Логирование** - Python logging
- **systemd журналы** - journalctl

---

## Requirements.txt

```txt
# FastAPI
fastapi==0.104.1
uvicorn[standard]==0.24.0

# ChromaDB
chromadb==0.4.18
sentence-transformers==2.2.2

# Utilities
python-dotenv==1.0.0
aiofiles==23.2.1
jinja2==3.1.2
python-multipart==0.0.6

# Optional
pydantic==2.5.0
pydantic-settings==2.1.0
```

---

## Конфигурация (.env)

```bash
# Application
APP_NAME=video_search_web
APP_HOST=0.0.0.0
APP_PORT=8001
DEBUG=false

# ChromaDB
CHROMA_PERSIST_DIR=/opt/video_web/chroma_db
CHROMA_COLLECTION_NAME=video_segments

# Embeddings
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# Videos
VIDEO_DIR=/opt/video_web/videos
PHASE2_DATA_DIR=/opt/video_web/data/phase2

# Security
WEB_USERNAME=video_admin
WEB_PASSWORD=Video2025!Web

# Logging
LOG_LEVEL=INFO
LOG_FILE=/opt/video_web/logs/app.log
```

---

## Тестирование

### Локальное тестирование:

1. **ChromaDB интеграция:**
```bash
cd /mnt/e/Projects/Quantum/Video_tagging_db
python db_manager.py --test
```

2. **Semantic search:**
```bash
python search_engine.py --query "авария на дороге" --limit 5
```

3. **FastAPI приложение:**
```bash
cd web_app
uvicorn app.main:app --reload --port 8001
```

4. **Проверка API:**
```bash
curl "http://localhost:8001/api/search?q=авария&limit=5"
```

### Тестирование на VPS:

1. **Health check:**
```bash
curl http://VPS_IP/health
```

2. **Поиск:**
```bash
curl "http://VPS_IP/api/search?q=новости"
```

3. **Видео endpoints:**
```bash
curl http://VPS_IP/api/video/news/segments
```

---

## Метрики успеха

### EPIC 1 (Векторная БД):
- ✅ ChromaDB успешно хранит >100 сегментов
- ✅ Поиск работает на русском и английском
- ✅ Relevance score >0.7 для точных совпадений
- ✅ Время поиска <1 секунды

### EPIC 2 (Веб-интерфейс):
- ✅ Деплой на VPS за <10 минут через Ansible
- ✅ API response time <500ms
- ✅ Видео загружается и играет корректно
- ✅ Навигация по таймингам работает
- ✅ Responsive дизайн на mobile

---

## Риски и митигация

### Риск 1: ChromaDB performance на больших данных
- **Митигация:** Индексирование, pagination, caching

### Риск 2: Embeddings модель занимает много RAM
- **Митигация:** Использовать mini-модель (L12 вместо L24)

### Риск 3: Видео файлы слишком большие для VPS
- **Митигация:** Конвертация в H.264 720p, rsync синхронизация

### Риск 4: Долгий деплой из-за медленного интернета
- **Митигация:** Использовать rsync --compress, deploy_app_only.yml

---

## Расписание (оптимистичное)

| День | Задачи | Время |
|------|--------|-------|
| День 1 | VIDEO-4, VIDEO-10 | 4-6 ч |
| День 2 | VIDEO-5, VIDEO-11 | 4-6 ч |
| День 3 | VIDEO-12, VIDEO-13 | 6-8 ч |
| День 4 | VIDEO-14, VIDEO-15 | 4-6 ч |

**Итого:** 4 дня интенсивной работы

---

## Следующие шаги

1. ✅ Создать второй эпик VIDEO-9 (веб-интерфейс)
2. ✅ Создать все задачи и привязать к эпикам
3. Начать реализацию с VIDEO-10 (ChromaDB integration)
4. Тестировать каждый компонент отдельно
5. Интеграционное тестирование
6. Деплой на VPS

---

## Ссылки на задачи

### EPIC 1: VIDEO-1 - Vector DB
- https://denisenkovano01.atlassian.net/browse/VIDEO-1
- VIDEO-4: https://denisenkovano01.atlassian.net/browse/VIDEO-4
- VIDEO-5: https://denisenkovano01.atlassian.net/browse/VIDEO-5
- VIDEO-10: https://denisenkovano01.atlassian.net/browse/VIDEO-10
- VIDEO-11: https://denisenkovano01.atlassian.net/browse/VIDEO-11
- VIDEO-3: https://denisenkovano01.atlassian.net/browse/VIDEO-3

### EPIC 2: VIDEO-9 - Web Interface
- https://denisenkovano01.atlassian.net/browse/VIDEO-9
- VIDEO-12: https://denisenkovano01.atlassian.net/browse/VIDEO-12
- VIDEO-13: https://denisenkovano01.atlassian.net/browse/VIDEO-13
- VIDEO-14: https://denisenkovano01.atlassian.net/browse/VIDEO-14
- VIDEO-15: https://denisenkovano01.atlassian.net/browse/VIDEO-15

---

**Путь к документу:** E:\Projects\Quantum\Video_tagging_db\docs\251120_14_video_search_web_planning.md
