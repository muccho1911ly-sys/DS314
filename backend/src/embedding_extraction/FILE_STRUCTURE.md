# File Structure - Embedding Extraction Module

## Tổng quan cấu trúc

```
backend/src/embedding_extraction/
├── README.md              # Tài liệu tổng quan module
├── MODELS.md              # Danh sách models được hỗ trợ
├── API_REFERENCE.md       # API reference chi tiết
├── FILE_STRUCTURE.md      # File này - cấu trúc và mô tả files
│
├── models/                # Core model implementations
│   ├── __init__.py
│   ├── embedder.py        # API chính: load model & encode images
│   ├── encoder.py         # API encode video frames
│   ├── frame_loader.py    # DataLoader cho video frames
│   ├── registry.py        # Model registry và factory pattern
│   │
│   └── backends/          # Backend implementations
│       ├── __init__.py
│       ├── base.py                    # Abstract base class
│       ├── open_clip.py               # OpenCLIP backend
│       ├── blip2.py                   # BLIP-2 backend
│       ├── perception_encoder.py      # PE-Core backend
│       └── test_perception_encoder.py # Tests
│
├── pipeline/              # Processing pipelines
│   ├── __init__.py
│   └── extract_embeddings.py  # Main extraction pipeline
│
└── utils/                 # Utilities
    ├── __init__.py
    └── similarity.py      # Similarity computations
```

## Chi tiết từng file

### Documentation Files

#### README.md
- Tổng quan về module
- Hướng dẫn sử dụng cơ bản
- Các use cases chính
- Dependencies

#### MODELS.md
- Danh sách tất cả models được hỗ trợ
- So sánh performance các models
- Hướng dẫn chọn model phù hợp
- Config examples cho từng model

#### API_REFERENCE.md
- Chi tiết API của tất cả functions
- Parameters và return values
- Code examples
- Error handling

### models/

#### embedder.py (158 lines)
**Chức năng:** API chính để load embedding models và encode keyframe images

**Key functions:**
- `load_embedding_model()`: Load model với backend tùy chọn
- `encode_keyframe_images()`: Batch encoding cho images
- `list_image_paths()`: Scan image directory
- `load_embedding()`: Load embeddings đã lưu

**Dependencies:**
- `registry.py`: Load models
- `backends/base.py`: LoadedModel type
- torch, numpy, PIL, cv2

#### encoder.py (83 lines)
**Chức năng:** API để encode video frames thành embeddings

**Key functions:**
- `load_clip_model()`: Load CLIP model
- `encode_video_frames()`: Full video encoding pipeline
- `load_embeddings()`: Load video embeddings từ pickle

**Features:**
- Mixed precision (FP16) support
- Automatic memory cleanup
- Progress tracking với tqdm
- Pickle serialization

#### frame_loader.py
**Chức năng:** Efficient DataLoader cho video frames

**Key components:**
- `FrameDataset`: PyTorch Dataset cho video
- `build_frame_loader()`: Factory function
- `iter_frame_batches()`: Batch iterator

#### registry.py
**Chức năng:** Model registry và loading system

**Supported backends:**
- open_clip
- blip2
- perception_encoder

### models/backends/

#### base.py
**Chức năng:** Abstract base class cho tất cả backends

**Key classes:**
- `LoadedModel`: Return type (model, preprocess, device, precision)
- `BackendBase`: Abstract backend interface

#### open_clip.py
**Chức năng:** OpenCLIP backend implementation

**Supported models:**
- All OpenCLIP models (CLIP, SigLIP, LongCLIP, DFN, etc.)

#### blip2.py
**Chức năng:** BLIP-2 vision-language model backend

**Supported models:**
- Salesforce/blip2-opt-2.7b
- Salesforce/blip2-flan-t5-xl

**Features:**
- Multiple pooling strategies (cls, mean, last)

#### perception_encoder.py
**Chức năng:** Perception Encoder (PE-Core) backend

### pipeline/

#### extract_embeddings.py
**Chức năng:** Main pipeline để batch extract embeddings

**Key classes:**
- `ExtractEmbeddingPipeline`: Single-model extraction
- `run_multi_model()`: Multi-model extraction

### utils/

#### similarity.py
**Chức năng:** Tính toán similarity giữa embeddings

## Dependencies Graph

```
extract_embeddings.py
    └─> embedder.py
        └─> registry.py
            ├─> backends/open_clip.py
            ├─> backends/blip2.py
            └─> backends/perception_encoder.py
                └─> backends/base.py

encoder.py
    ├─> registry.py
    └─> frame_loader.py
```

## External Dependencies

**Core:**
- torch, numpy, Pillow, opencv-python, tqdm

**Backend-specific:**
- open_clip_torch (OpenCLIP)
- transformers (BLIP-2)
- decord (video loading)

## Entry Points

```python
# Load và encode images
from src.embedding_extraction.models.embedder import (
    load_embedding_model, encode_keyframe_images
)

# Encode videos
from src.embedding_extraction.models.encoder import (
    load_clip_model, encode_video_frames
)

# Run pipeline
from src.embedding_extraction.pipeline.extract_embeddings import (
    ExtractEmbeddingPipeline, run_multi_model
)
```
