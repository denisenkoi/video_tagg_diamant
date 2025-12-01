# Agent Instructions: Улучшение индексации и поиска

## ОБЩИЙ ПОДХОД - ОБЯЗАТЕЛЬНО К ИСПОЛНЕНИЮ

1. **Читай `docs/worksheet.md`** - там текущий прогресс и заметки
2. **Выполняй чекпоинты строго по порядку**
3. **После каждого чекпоинта:**
   - Обновляй worksheet.md (статус + заметки + timestamp)
   - Отмечай чекпоинт как выполненный здесь
4. **При ошибке** - записывай в worksheet и ОСТАНАВЛИВАЙСЯ
5. **ЗАПРЕЩЕНО:**
   - Упрощать функциональность
   - Добавлять "улучшения" не по заданию
   - Менять существующий код без необходимости
   - Ломать pipeline обработки видео

---

## ОКРУЖЕНИЕ

- Python: `/home/vano/anaconda3/envs/vllm/bin/python`
- PostgreSQL: `localhost:5432/video_tagging_db` (user: video_user, pass: video_pass_2024)
- Рабочая директория: `/mnt/e/Projects/Quantum/Video_tagging_db`
- Видео: `video/` (относительные пути в БД)

---

## ТЕКУЩАЯ ЗАДАЧА: VIDEO-30 (Epic: VIDEO-28)

**Цель:** LLM Loop Detection + интеграция с оркестратором для автоперезапуска

### Проблема:
LLM (Qwen3-32B-AWQ) зацикливается на генерации повторяющихся слов:
```json
"keywords": ["айтып", "айтып", "айтып", ... x500]
```

### Что уже сделано:
- [x] Детекция loop в phase2_vllm_analysis.py (keyword_loop, keyword_flood, search_terms_loop)
- [x] Оркестратор скопирован в проект (model_orchestrator_v2.sh)
- [x] Документация: docs/251130_14_llm_loop_detection_and_orchestrator.md
- [x] Jira задача VIDEO-30 создана

---

## ЧЕКПОИНТЫ

### CP1 - Исправить cron для оркестратора [X] DONE
- Cron запускает из /home/vano, нужен cd в нужную директорию
- `*/1 * * * * cd /mnt/e/Projects/Quantum/Video_tagging_db && ./model_orchestrator_v2.sh cron`

### CP2 - Добавить автоперезапуск LLM при loop [X] DONE
- При обнаружении keyword_loop/search_terms_loop → создать restart_vllm.flag
- Ожидать перезапуска LLM (check API health)
- Retry текущего сегмента
- Исправлены баги: JSON парсинг через Python, проверка PID > 0, grace period

### CP3 - Запустить Phase2 с новой детекцией [X] DONE
- Перезапустить LLM через оркестратор (если зациклилась)
- Запустить phase2_vllm_analysis.py
- AlmaAta: ✅ DONE (32/32 segments)
- news, SovKz: processing

### CP4 - Загрузка в БД [X] DONE
- После успешного Phase2 запустить load_data_to_db.py
- Загружено: AlmaAta (32), news (60), SovKz (35) сегментов

### CP5 - Тестирование поиска [X] DONE
- ✅ "ракета" → SovKz:1 (relevance=0.488) - найден "космические корабли, запуск ракеты"
- ✅ "электричество" → news:22, SovKz:16-18
- ✅ "печь" → news:35, SovKz:1

---

## ФАЙЛЫ ДЛЯ ИЗМЕНЕНИЯ

| Файл | Изменения |
|------|-----------|
| phase2_vllm_analysis.py | Авторестарт при loop + ожидание готовности API |
| model_orchestrator_v2.sh | Уже скопирован, работает |
| crontab | Исправлен (cd в директорию проекта) |

---

## КРИТИЧЕСКИЕ ТОЧКИ ОСТАНОВКИ

1. **Phase2 долго работает** - нормально, GPU-intensive
2. **Ошибки в JSON** - проверить output/*.json вручную
3. **search_terms пустые** - проверить промпт Phase2

---

## СВЯЗАННЫЕ ЗАДАЧИ JIRA

| Задача | Описание | Статус |
|--------|----------|--------|
| VIDEO-28 | Epic: Улучшение качества индексации и поиска | In Progress |
| VIDEO-29 | Переиндексация Phase2 с search_terms | In Progress |

---

## КЛЮЧЕВЫЕ ФАЙЛЫ

- `videos_config.json` - конфиг видео с force флагом
- `phase1_whisper_frames.py` - читает force, вызывает очистку
- `phase2_vllm_analysis.py` - генерация описаний и search_terms
- `load_data_to_db.py` - загрузка JSON в PostgreSQL
- `db_manager_postgres.py` - методы БД (delete_video_data, add_segments, search)

---

**Путь:** E:\Projects\Quantum\Video_tagging_db\docs\AGENT_INSTRUCTIONS.md
