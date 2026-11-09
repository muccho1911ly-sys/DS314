# Embedding Extraction Module

Module này chịu trách nhiệm trích xuất embeddings từ hình ảnh và video sử dụng các mô hình deep learning.

## Cấu trúc thư mục

```
embedding_extraction/
├── models/                 # Các mô hình embedding
│   ├── backends/          # Các backend implementation
│   │   ├── base.py       # Base class cho backends
│   │   ├── open_clip.py  # OpenCLIP backend
│   │   ├── blip2.py      # BLIP-2 backend
│   │   └── perception_encoder.py  # Perception Encoder backend
│   ├── embedder.py        # API chính để load model và encode images
│   ├── encoder.py         # API để encode video frames
│   ├── frame_loader.py    # DataLoader cho video frames
│   └── registry.py        # Model registry và factory
├── pipeline/              # Pipeline xử lý
│   └── extract_embeddings.py  # Pipeline trích xuất embeddings
└── utils/                 # Tiện ích
    └── similarity.py      # Tính toán độ tương đồng
```

## Chức năng chính

### 1. Models (models/)
- **embedder.py**: API chính để load embedding models và encode keyframe images
  - `load_embedding_model()`: Load bất kỳ model nào (OpenCLIP, BLIP-2, PE-Core)
  - `encode_keyframe_images()`: Encode batch images thành embeddings
  
- **encoder.py**: API để encode video frames
  - `load_clip_model()`: Load CLIP model
  - `encode_video_frames()`: Encode toàn bộ video thành embeddings
  - `load_embeddings()`: Load embeddings đã lưu từ file

- **registry.py**: Quản lý registry của các models
  - Đăng ký và load models theo backend
  - Hỗ trợ nhiều backends khác nhau

- **frame_loader.py**: DataLoader tối ưu cho video frames
  - Batch processing cho video frames
  - Multi-worker support

### 2. Backends (models/backends/)
- **base.py**: Abstract base class cho tất cả backends
- **open_clip.py**: OpenCLIP implementation (SigLIP, CLIP, etc.)
- **blip2.py**: BLIP-2 vision-language model
- **perception_encoder.py**: Perception Encoder (PE-Core)

### 3. Pipeline (pipeline/)
- **extract_embeddings.py**: 
  - `ExtractEmbeddingPipeline`: Pipeline chính để trích xuất embeddings
  - `run_multi_model()`: Chạy nhiều models cùng lúc

### 4. Utils (utils/)
- **similarity.py**: Các hàm tính toán độ tương đồng giữa embeddings

## Sử dụng

### Load và encode images
```python
from src.embedding_extraction.models.embedder import (
    load_embedding_model,
    encode_keyframe_images
)

# Load model
model, preprocess, device, precision = load_embedding_model(
    model_name="ViT-L-14",
    pretrained="openai",
    precision="fp32",
    device_name="cuda",
    backend="open_clip"
)

# Encode images
embeddings, valid_paths = encode_keyframe_images(
    model=model,
    preprocess=preprocess,
    device=device,
    image_paths=image_paths,
    batch_size=32,
    precision=precision
)
```

### Encode video
```python
from src.embedding_extraction.models.encoder import (
    load_clip_model,
    encode_video_frames
)

# Load model
model, preprocess, device = load_clip_model(
    model_name="ViT-L-14",
    pretrained="openai",
    device_name="cuda"
)

# Encode video
embeddings = encode_video_frames(
    model=model,
    preprocess=preprocess,
    device=device,
    video_path=video_path,
    output_path=output_path,
    batch_size=32,
    loader_cfg=loader_config
)
```

### Sử dụng Pipeline
```python
from src.embedding_extraction.pipeline.extract_embeddings import (
    ExtractEmbeddingPipeline
)

# Config
config = {
    "model": {
        "name": "ViT-L-14",
        "backend": "open_clip",
        "pretrained": "openai",
        "batch_size": 32,
        "normalize": True
    },
    "input_keyframes_root": "/path/to/keyframes",
    "output_embeddings_root": "/path/to/embeddings",
    "save": {
        "overwrite": True,
        "extension": ".npy"
    }
}

# Run pipeline
pipeline = ExtractEmbeddingPipeline(config)
pipeline.run()
```

## Models hỗ trợ

### OpenCLIP Backend
- ViT-B-16, ViT-B-32
- ViT-L-14, ViT-L-16
- SigLIP variants
- LongCLIP variants
- DFN (Diffusion-based Feature Network)

### BLIP-2 Backend
- Salesforce/blip2-opt-2.7b
- Salesforce/blip2-flan-t5-xl
- Custom pooling strategies

### Perception Encoder Backend
- PE-Core models
- Custom perception encoders

## Dependencies
- torch
- numpy
- opencv-python
- Pillow
- open_clip_torch
- transformers (cho BLIP-2)
- tqdm

## Notes
- Tất cả embeddings được normalize về unit vectors (L2 norm = 1)
- Hỗ trợ mixed precision (FP16) trên GPU để tăng tốc
- Multi-worker DataLoader cho throughput cao
- Automatic memory cleanup sau mỗi video
