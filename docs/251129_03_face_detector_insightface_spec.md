# Specification: Face Detector Migration to InsightFace

**Date:** 2025-11-29 03:00
**Task:** VIDEO-17 (TBD)
**Status:** Draft

---

## Goal

Migrate `face_processor.py` from DeepFace + TensorFlow to InsightFace + FER for:
- Higher accuracy (99.86% vs 99.65% on LFW)
- Better performance (7-10x speedup)
- Cleaner dependencies (no TensorFlow)

---

## Current Implementation (DeepFace)

```
Detection: OpenCV Haar Cascade (outdated, low accuracy)
Recognition: DeepFace -> ArcFace (sequential, slow)
Age/Gender: DeepFace VGG models (sequential)
Emotion: DeepFace (sequential)
Race: DeepFace (TO BE REMOVED)
```

**Performance:** ~300ms per face

---

## Target Implementation (InsightFace)

```
Detection: SCRFD (InsightFace) - SOTA detector
Recognition: ArcFace buffalo_l (512D embeddings)
Age/Gender: genderage.onnx (included in buffalo_l)
Emotion: FER library (separate)
Race: REMOVED
```

**Expected Performance:** ~40ms per face

---

## Dependencies

### Install
```bash
pip install insightface
pip install onnxruntime-gpu  # GPU inference
pip install fer              # Emotion detection
```

### Remove
- TensorFlow (via DeepFace)
- deepface package (optional, can keep for fallback)

### Models (auto-download)
Path: `~/.insightface/models/buffalo_l/`

| File | Purpose |
|------|---------|
| `det_10g.onnx` | SCRFD face detector (10GF) |
| `w600k_r50.onnx` | ArcFace ResNet50 (embeddings 512D) |
| `genderage.onnx` | Age and gender detection |
| `1k3d68.onnx` | 3D facial landmarks (68 points) |
| `2d106det.onnx` | 2D facial landmarks (106 points) |

Manual download if needed:
- https://github.com/deepinsight/insightface/releases
- https://huggingface.co/immich-app/buffalo_l

---

## Code Changes

### 1. Initialize Models

**DELETE:**
```python
from deepface import DeepFace
_arcface_model = None
_age_model = None
_gender_model = None
_emotion_model = None
_race_model = None
_use_deepface_directly = False
```

**ADD:**
```python
from insightface.app import FaceAnalysis
from fer import FER

_face_analyzer = None
_emotion_detector = None
```

### 2. initialize_face_models()

```python
def initialize_face_models():
    global _face_analyzer, _emotion_detector

    print("Initializing InsightFace models...")

    _face_analyzer = FaceAnalysis(
        name='buffalo_l',
        root='~/.insightface',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    # ctx_id: 0 = GPU, -1 = CPU
    # det_size: detection resolution
    _face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    print(f"InsightFace initialized with {len(_face_analyzer.models)} models")

    # Emotion detector (FER)
    _emotion_detector = FER(mtcnn=False)
    print("FER emotion detector initialized")
```

### 3. detect_faces_single_frame()

```python
def detect_faces_single_frame(frame: np.ndarray, min_face_size: int) -> List[Dict[str, Any]]:
    """Detect faces using InsightFace SCRFD detector"""

    assert frame.shape[2] == 3, 'Frame must be BGR format'
    assert min_face_size > 0, 'Minimum face size must be positive'

    # InsightFace expects BGR (OpenCV format) - OK
    faces = _face_analyzer.get(frame)

    detected_faces = []
    for i, face in enumerate(faces):
        # face.bbox = [x1, y1, x2, y2]
        x1, y1, x2, y2 = face.bbox.astype(int)
        w = x2 - x1
        h = y2 - y1

        # Size filter
        if w >= min_face_size and h >= min_face_size:
            face_data = {
                'coordinates': (x1, y1, w, h),
                'region': frame[y1:y2, x1:x2],
                'size': (w, h),
                'face_id': f'face_{i:03d}',
                'center_x': x1 + w // 2,
                'center_y': y1 + h // 2,
                # InsightFace provides all attributes!
                'embedding': face.normed_embedding,  # 512D normalized
                'age': int(face.age),
                'gender': 'Man' if face.gender == 1 else 'Woman',
                'det_score': float(face.det_score),
                'landmarks': face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None
            }
            detected_faces.append(face_data)

    return detected_faces
```

### 4. add_emotion_to_faces()

```python
def add_emotion_to_faces(faces_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add emotion detection using FER"""
    global _emotion_detector

    for face in faces_data:
        try:
            result = _emotion_detector.detect_emotions(face['region'])
            if result:
                emotions = result[0]['emotions']
                face['emotion'] = max(emotions, key=emotions.get)
            else:
                face['emotion'] = 'neutral'
        except Exception:
            face['emotion'] = 'neutral'

    return faces_data
```

### 5. Functions to DELETE

- `process_faces_with_deepface()` - not needed
- `process_faces_with_tf_models()` - not needed
- `preprocess_face_for_tf()` - not needed
- All `race` references - remove completely

### 6. Functions to UPDATE

- `process_faces_batch_tf()` - simplify, just call add_emotion_to_faces()
- `cluster_faces_to_persons()` - remove race field
- `group_faces_by_timestamp()` - remove race field

---

## InsightFace Face Object Structure

```python
face.bbox              # [x1, y1, x2, y2] - bounding box
face.det_score         # float - detection confidence
face.landmark_3d_68    # 68 3D landmarks
face.landmark_2d_106   # 106 2D landmarks
face.embedding         # 512D raw embedding
face.normed_embedding  # 512D L2-normalized (USE THIS!)
face.gender            # 0 = Female, 1 = Male
face.age               # int - age
```

---

## Performance Comparison

| Metric | DeepFace (old) | InsightFace (new) |
|--------|----------------|-------------------|
| Detection | ~100ms | ~10ms |
| Embedding | ~50ms/face | included in det |
| Age/Gender | ~100ms/face | included in det |
| Emotion | ~50ms/face | +30ms/face (FER) |
| **Total** | ~300ms/face | ~40ms/face |

**Speedup: 7-10x**

---

## Testing Checklist

1. [ ] Face detection works (SCRFD instead of Haar)
2. [ ] Embeddings generated (512D normalized)
3. [ ] Age/Gender detected correctly
4. [ ] Emotion detection works (FER)
5. [ ] Clustering works (cosine similarity)
6. [ ] GPU used (check nvidia-smi)
7. [ ] No race field in output

### GPU Check
```python
import onnxruntime
print(onnxruntime.get_available_providers())
# Expected: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## License

**InsightFace code**: MIT License
**buffalo_l models**: Non-commercial research only

For exhibition/demo - OK.
For commercial product - license required from InsightFace.

---

**Document Path (Windows):** E:\Projects\Quantum\Video_tagging_db\docs\251129_03_face_detector_insightface_spec.md
