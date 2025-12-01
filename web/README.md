# Video Tagging Web Interface

Веб-интерфейс для поиска по видео и просмотра галереи лиц.

## Быстрый старт

```bash
cd /mnt/e/Projects/Quantum/Video_tagging_db/web
./run.sh
```

Или:
```bash
cd /mnt/e/Projects/Quantum/Video_tagging_db/web
/home/vano/anaconda3/envs/vllm/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Открыть в браузере: **http://localhost:8000**

## Функции

### Поиск по видео
- Семантический поиск по описаниям сегментов
- Переход к видео с нужным таймстампом

### Галерея персон
- Просмотр всех лиц из видео
- Два режима кластеризации:
  - **On-the-fly** — кластеризация при обработке
  - **Post-clustering** — Agglomerative clustering по всем эмбеддингам

### Видеоплеер
- HTML5 плеер с поддержкой seeking
- Навигация по таймстампу из URL

## API Endpoints

| Endpoint | Описание |
|----------|----------|
| `GET /api/search?q=текст` | Семантический поиск |
| `GET /api/videos` | Список видео |
| `GET /api/persons?video=X&clustering=Y` | Персоны из видео |
| `GET /api/thumbnail/{id}` | Миниатюра лица |
| `GET /api/video/{name}/stream` | Стриминг видео |
| `GET /api/video/{name}/frame?t=X` | Кадр по времени |
| `GET /health` | Статус сервера |

## Требования

- Python 3.10+ (anaconda vllm env)
- PostgreSQL с данными video_tagging_db
- Видеофайлы в `video/`

## Для выставки

1. Запустить PostgreSQL (если не запущен)
2. Запустить веб-сервер: `./run.sh`
3. Открыть http://localhost:8000 в браузере

Порт 8000 можно пробросить на Windows через WSL.
