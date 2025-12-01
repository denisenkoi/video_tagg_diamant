# PostgreSQL Migration: ChromaDB → PostgreSQL + pgvector

**Дата:** 2025-11-21 01:30
**Статус:** В процессе установки

---

## Причина миграции

ChromaDB показал **высокую хрупкость при сбоях**:
- При краше приложения (FileNotFoundError на сохранении) база данных получила **коррупцию**
- Ошибка: `range start index 10 out of range for slice of length 9` в Rust-коде
- Это произошло при **первом же использовании**

**Решение:** Перейти на PostgreSQL - проверенную ACID-совместимую СУБД.

---

## Что установлено

### 1. PostgreSQL 18.1
```bash
sudo apt install -y postgresql-common
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
sudo apt install -y postgresql
sudo apt install -y postgresql-18-pgvector
```

**Статус:** ✅ Установлено и запущено

### 2. База данных
```sql
CREATE USER video_user WITH PASSWORD 'video_pass_2024';
CREATE DATABASE video_tagging_db OWNER video_user;
CREATE EXTENSION vector;
GRANT ALL PRIVILEGES ON DATABASE video_tagging_db TO video_user;
GRANT ALL ON SCHEMA public TO video_user;
```

**Статус:** ✅ Создано

### 3. Таблицы

#### video_files (метаданные + статусы)
```sql
CREATE TABLE video_files (
    video_name VARCHAR(255) PRIMARY KEY,
    video_path TEXT NOT NULL,
    language VARCHAR(10) NOT NULL,
    video_duration FLOAT NOT NULL,
    total_segments INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    fps FLOAT NOT NULL,

    -- Phase statuses
    phase1_status VARCHAR(20) DEFAULT 'pending',
    phase1_start_time TIMESTAMP,
    phase1_segments_created INTEGER,

    phase2_status VARCHAR(20) DEFAULT 'pending',
    phase2_start_time TIMESTAMP,
    phase2_segments_analyzed INTEGER,

    db_load_status VARCHAR(20) DEFAULT 'pending',
    db_load_start_time TIMESTAMP,
    db_segments_loaded INTEGER,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### video_segments (сегменты для поиска)
```sql
CREATE TABLE video_segments (
    id VARCHAR(255) PRIMARY KEY,
    video_name VARCHAR(255) NOT NULL REFERENCES video_files(video_name) ON DELETE CASCADE,
    video_path TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    duration FLOAT NOT NULL,

    description TEXT,
    keywords TEXT,
    confidence VARCHAR(20),
    content_type VARCHAR(50),

    -- Vector embedding для семантического поиска
    embedding vector(384),

    created_at TIMESTAMP DEFAULT NOW()
);

-- Индексы
CREATE INDEX video_segments_embedding_idx ON video_segments
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX video_segments_video_name_idx ON video_segments(video_name);
CREATE INDEX video_segments_confidence_idx ON video_segments(confidence);
CREATE INDEX video_files_phase1_status_idx ON video_files(phase1_status);
CREATE INDEX video_files_phase2_status_idx ON video_files(phase2_status);
CREATE INDEX video_files_db_load_status_idx ON video_files(db_load_status);
```

**Статус:** ✅ Созданы

---

## Что разработано

### db_manager_postgres.py ✅
Новый менеджер базы данных:
- Подключение к PostgreSQL
- Автоматическая генерация эмбеддингов через `sentence-transformers`
- Все методы API совместимы с ChromaDB версией
- Использует `psycopg2` + `pgvector`

**Особенность:** Встроенный класс `EmbeddingModel` (Singleton) для генерации векторов.

---

## Что осталось сделать

### ❌ Шаг 1: Установить Python-зависимости в conda окружение
```bash
conda activate auto_prog
pip install psycopg2-binary pgvector sentence-transformers
```

### ❌ Шаг 2: Протестировать db_manager_postgres.py
```bash
python db_manager_postgres.py
```
Ожидается вывод:
```
=== PostgreSQL DB Manager Test ===
INFO:db_manager:Connecting to PostgreSQL at localhost:5432/video_tagging_db
INFO:db_manager:Loading sentence-transformers model...
INFO:db_manager:✓ Model loaded
INFO:db_manager:✓ PostgreSQL connected:
INFO:db_manager:  - video_files: 0 records
INFO:db_manager:  - video_segments: 0 segments

Stats:
  Video files: 0
  Video segments: 0
  Database: video_tagging_db

✓ PostgreSQL DB Manager is working!
```

### ❌ Шаг 3: Заменить db_manager.py
```bash
# Сделать backup старого
mv db_manager.py db_manager_chromadb_backup.py

# Использовать PostgreSQL версию
cp db_manager_postgres.py db_manager.py
```

### ❌ Шаг 4: Удалить поврежденную ChromaDB базу
```bash
rm -rf chroma_db/
```

### ❌ Шаг 5: Протестировать Phase 1
```bash
python phase1_whisper_frames.py
```

Должно:
- Подключиться к PostgreSQL
- Обработать 3 видео (news, SovKz, AlmaAta)
- Записать статусы в `video_files`
- Сохранить результаты в `output/`

---

## Параметры подключения к PostgreSQL

### Основные параметры
```python
host = "localhost"
port = 5432
database = "video_tagging_db"
user = "video_user"
password = "video_pass_2024"
```

### Connection string (для других инструментов)
```
postgresql://video_user:video_pass_2024@localhost:5432/video_tagging_db
```

### Проверка подключения из командной строки
```bash
PGPASSWORD=video_pass_2024 psql -U video_user -d video_tagging_db -h localhost -c "SELECT 1;"
```

### Управление сервисом PostgreSQL на WSL
```bash
# Запуск
sudo service postgresql start

# Остановка
sudo service postgresql stop

# Статус
sudo service postgresql status

# Перезапуск
sudo service postgresql restart
```

---

## Создание новой базы для другого проекта

### Шаг 1: Создать базу и пользователя
```bash
# Подключиться как postgres
sudo -u postgres psql

# В psql выполнить:
CREATE USER assistant_user WITH PASSWORD 'assistant_pass_2024';
CREATE DATABASE assistant_db OWNER assistant_user;

# Подключиться к новой базе
\c assistant_db

# Включить расширения (если нужны)
CREATE EXTENSION vector;  -- Для векторного поиска
CREATE EXTENSION pg_trgm;  -- Для текстового поиска

# Дать права
GRANT ALL PRIVILEGES ON DATABASE assistant_db TO assistant_user;
GRANT ALL ON SCHEMA public TO assistant_user;

# Выйти
\q
```

### Шаг 2: Создать таблицы
```sql
-- Пример структуры для ассистента
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    response TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE user_context (
    user_id VARCHAR(255) PRIMARY KEY,
    context_data JSONB,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Если нужен векторный поиск
CREATE TABLE knowledge_base (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX knowledge_base_embedding_idx ON knowledge_base
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### Шаг 3: Подключение из Python
```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="assistant_db",
    user="assistant_user",
    password="assistant_pass_2024"
)

# Или использовать connection string
conn = psycopg2.connect(
    "postgresql://assistant_user:assistant_pass_2024@localhost:5432/assistant_db"
)
```

### Шаг 4: (Опционально) Создать отдельного db_manager для ассистента
Скопировать `db_manager_postgres.py` и адаптировать под структуру таблиц ассистента.

---

## Преимущества PostgreSQL над ChromaDB

| Параметр | ChromaDB | PostgreSQL + pgvector |
|----------|----------|----------------------|
| **ACID транзакции** | ❌ Нет | ✅ Да |
| **Надежность при сбоях** | ❌ Низкая (коррупция при краше) | ✅ Высокая |
| **SQL запросы** | ❌ Ограниченное API | ✅ Полный SQL |
| **Масштабирование** | ❌ Только embedded | ✅ Client-server, репликация |
| **Backup/restore** | ⚠️ Копирование файлов | ✅ pg_dump/pg_restore |
| **Мониторинг** | ❌ Минимальный | ✅ Огромная экосистема |
| **Эмбеддинги** | ✅ Автоматические | ⚠️ Нужно считать вручную |
| **Скорость разработки** | ✅ Быстрее (меньше кода) | ⚠️ Больше boilerplate |

---

## Worksheet - Текущий статус

**Текущая задача:** Установка зависимостей в conda окружение `auto_prog`

**Команда для выполнения:**
```bash
conda activate auto_prog
pip install psycopg2-binary pgvector sentence-transformers
python db_manager_postgres.py
```

**После успешного теста:**
1. Заменить `db_manager.py` на PostgreSQL версию
2. Удалить `chroma_db/`
3. Запустить Phase 1 повторно
4. Проверить что статусы пишутся в PostgreSQL

---

**Путь к документу:** E:\Projects\Quantum\Video_tagging_db\docs\251121_01_postgresql_migration.md
