# VIDEO-19: Post-clustering Analysis

**Дата:** 2025-11-29
**Статус:** Анализ завершен

---

## Цель

Сравнить качество кластеризации лиц:
- On-the-fly (при обработке видео)
- Post-clustering (на всех эмбеддингах после обработки)

---

## Методы кластеризации

### 1. On-the-fly (текущий)
- Алгоритм: последовательное сравнение с известными персонами
- Threshold: cosine similarity >= 0.7
- Результат: 17 персон

### 2. HDBSCAN
- min_cluster_size=2, min_samples=1
- metric: precomputed (cosine distance)
- Результат: 26 кластеров + 5 noise
- **Проблема:** слишком много кластеров (over-clustering)

### 3. Agglomerative Clustering (лучший)
- distance_threshold=0.3 (эквивалент similarity=0.7)
- linkage='average'
- metric: precomputed (cosine distance)
- Результат: 16 кластеров

---

## Результаты сравнения

### Тестовые данные
- Видео: news.mp4 (854x470, 25 fps)
- Детекций: 116 лиц
- Эмбеддинги: 512D (ArcFace)

### Сравнение алгоритмов

| Threshold (sim) | On-the-fly | Agglomerative |
|-----------------|------------|---------------|
| 0.75            | -          | 20 кластеров  |
| 0.70            | 17 персон  | 16 кластеров  |
| 0.65            | -          | 12 кластеров  |
| 0.60            | -          | 10 кластеров  |

### Найденные расхождения

**Объединения (on-the-fly ошибочно разделил):**
```
person_06 (6 лиц) + person_16 (2 лица) → post_01
person_05 (1 лицо) + person_14 (3 лица) → post_09
```

**Разбиения (on-the-fly ошибочно объединил):**
```
person_05 → post_09 (1) + post_14 (4)
```

---

## Структура БД

```sql
ALTER TABLE face_detections ADD COLUMN person_id_postclustering VARCHAR(50);
ALTER TABLE face_detections ADD COLUMN person_id_alt VARCHAR(50);

-- Колонки:
-- person_id                - on-the-fly кластеризация
-- person_id_postclustering - Agglomerative clustering
-- person_id_alt            - альтернативная (HDBSCAN и др.)
```

---

## Код для пост-кластеризации

```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

# Получить эмбеддинги из БД
embeddings = np.array([...], dtype=np.float64)

# Нормализация и расстояния
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized = embeddings / norms
distances = cosine_distances(normalized)

# Кластеризация
clusterer = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.3,  # similarity = 0.7
    metric='precomputed',
    linkage='average'
)
labels = clusterer.fit_predict(distances)
```

---

## Выводы

1. Пост-кластеризация даёт более точные результаты
2. Agglomerative лучше HDBSCAN для этой задачи
3. Рекомендуется использовать как финальный этап после детекции
4. Позволяет перекластеризовать при добавлении новых видео

---

**Путь:** E:\Projects\Quantum\Video_tagging_db\docs\251129_18_post_clustering_analysis.md
