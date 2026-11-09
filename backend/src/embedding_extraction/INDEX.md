# Embedding Extraction Module - Index

> Module trích xuất embeddings từ images và videos sử dụng các mô hình deep learning.

## 📚 Tài liệu

- **[README.md](README.md)** - Tổng quan module, hướng dẫn cài đặt và sử dụng cơ bản
- **[API_REFERENCE.md](API_REFERENCE.md)** - API reference chi tiết cho developers
- **[MODELS.md](MODELS.md)** - Danh sách models được hỗ trợ và hướng dẫn chọn model
- **[FILE_STRUCTURE.md](FILE_STRUCTURE.md)** - Cấu trúc files và dependencies

## 🚀 Quick Start

### 1. Load model và encode images

```python
from src.embedding_extraction.models.embedder import (
    load_embedding_model,
    encode_keyframe_images,
    list_image_paths
)

# Load model
model, preprocess, device, precision = load_embedding_model(
    model_name="ViT-L-14",
    pretrained="openai",
    precision="amp",
    device_name="cuda",
    backend="open_clip"
)

# List images
image_paths = list_image_paths(Path("keyframes/video_001"))

# Encode
embeddings, valid_paths = encode_keyframe_images(
    model=model,
    preprocess=preprocess,
    device=device,
    image_paths=image_paths,
    batch_size=32,
    precision=precision,
    normalize=True
)
```

### 2. Encode video

```python
from src.embedding_extraction.models.encoder import (
    load_clip_model,
    encode_video_frames
)
from src.utils.config import FrameLoaderConfig

# Load model
model, preprocess, device = load_clip_model(
    model_name="ViT-L-14",
    pretrained="openai",
    device_name="cuda"
)

# Config
loader_cfg = FrameLoaderConfig(
    backend="decord",
    num_workers=4,
    pin_memory=True
)

# Encode video
embeddings = encode_video_frames(
    model=model,
    preprocess=preprocess,
    device=device,
    video_path=Path("videos/video.mp4"),
    output_path=Path("embeddings/video.pkl"),
    batch_size=32,
    loader_cfg=loader_cfg
)
```

### 3. Run pipeline

```python
from src.embedding_extraction.pipeline.extract_embeddings import (
    ExtractEmbeddingPipeline
)

config = {
    "model": {
        "name": "ViT-L-14",
        "backend": "open_clip",
        "pretrained": "openai",
        "batch_size": 32,
        "normalize": True
    },
    "input_keyframes_root": "data/keyframes",
    "output_embeddings_root": "data/embeddings",
    "save": {"overwrite": True}
}

pipeline = ExtractEmbeddingPipeline(config)
pipeline.run()
```

## 📦 Cấu trúc Module

```
embedding_extraction/
├── models/
│   ├── embedder.py         # Image encoding API
│   ├── encoder.py          # Video encoding API
│   ├── frame_loader.py     # DataLoader
│   ├── registry.py         # Model registry
│   └── backends/
│       ├── open_clip.py    # OpenCLIP backend
│       ├── blip2.py        # BLIP-2 backend
│       └── perception_encoder.py
├── pipeline/
│   └── extract_embeddings.py
└── utils/
    └── similarity.py
```

## 🎯 Use Cases

1. **Keyframe Extraction + Embedding**
   - Extract keyframes từ videos
   - Encode keyframes thành embeddings
   - Index vào FAISS/Milvus cho retrieval

2. **Semantic Search**
   - Encode text queries
   - Tìm similar images/videos
   - Reranking results

3. **Video Understanding**
   - Temporal analysis
   - Scene detection
   - Content-based retrieval

4. **Multi-modal Learning**
   - Vision-language models (BLIP-2)
   - Cross-modal retrieval
   - Image captioning

## 🔧 Models Được Hỗ Trợ

### OpenCLIP
- ViT-L-14 (OpenAI)
- ViT-SO400M-14-SigLIP-384 (WebLI)
- LongCLIP-L
- DFN5B-CLIP-ViT-H-14-384

### BLIP-2
- Salesforce/blip2-opt-2.7b
- Salesforce/blip2-flan-t5-xl

### Perception Encoder
- PE-Core models

👉 Xem chi tiết trong [MODELS.md](MODELS.md)

## 📖 API Documentation

### Core APIs
- `load_embedding_model()` - Load bất kỳ model nào
- `encode_keyframe_images()` - Batch encode images
- `encode_video_frames()` - Encode full video
- `load_embeddings()` - Load embeddings đã lưu

### Pipeline APIs
- `ExtractEmbeddingPipeline` - Single-model pipeline
- `run_multi_model()` - Multi-model extraction

### Utils
- `cosine_similarity()` - Tính similarity
- `batch_similarity()` - Efficient batch processing

👉 Xem chi tiết trong [API_REFERENCE.md](API_REFERENCE.md)

## 💡 Tips

### Performance
- Sử dụng `precision="amp"` trên GPU để tăng tốc
- Tăng `batch_size` nếu có đủ VRAM
- Set `num_workers=4-8` cho DataLoader
- Enable `pin_memory=True` trên CUDA

### Memory Management
- Pipeline tự động cleanup memory sau mỗi video
- Với large-scale processing, process theo batches
- Monitor GPU memory với `torch.cuda.memory_summary()`

### Model Selection
- **Accuracy priority**: SigLIP-SO400M
- **Speed priority**: ViT-B-16
- **Balanced**: ViT-L-14 (OpenAI)
- **Vietnamese content**: PE-Core

## 🔗 Related Modules

- `src/retrieval/` - Retrieval systems using embeddings
- `src/keyframe_extraction/` - Keyframe extraction
- `src/color_retrieval/` - Color-based retrieval
- `src/ocr_extraction/` - OCR text extraction

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra [FILE_STRUCTURE.md](FILE_STRUCTURE.md) để hiểu dependencies
2. Xem [API_REFERENCE.md](API_REFERENCE.md) cho usage details
3. Thử các examples trong [README.md](README.md)
4. Kiểm tra logs để debug

## 🎓 Examples

Xem thêm examples trong:
- `scripts/embedding_extraction/` - Batch processing scripts
- `tests/` - Unit tests và integration tests
- Config files trong `backend/configs/embedding_extraction/`

---

**Last updated:** 2026-09-14
**Module version:** 2.0
**Python version:** 3.12+
