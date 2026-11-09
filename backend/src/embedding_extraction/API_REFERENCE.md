# API Reference - Embedding Extraction Module

## Core APIs

### embedder.py

#### load_embedding_model()
Load bất kỳ embedding model nào (OpenCLIP, BLIP-2, PE-Core).

**Parameters:**
- `model_name` (str): Tên model (vd: "ViT-L-14", "Salesforce/blip2-opt-2.7b")
- `pretrained` (str | None): Pretrained weights (vd: "openai", "webli")
- `precision` (str): Độ chính xác ("fp32", "fp16", "amp")
- `device_name` (str): Device ("cuda", "cpu", "auto")
- `backend` (str | None): Backend name ("open_clip", "blip2", "perception_encoder")
- `logger` (logging.Logger | None): Logger instance
- `**extra`: Backend-specific parameters

**Returns:**
- `tuple`: (model, preprocess, device, precision)

**Example:**
```python
model, preprocess, device, precision = load_embedding_model(
    model_name="ViT-L-14",
    pretrained="openai",
    precision="amp",
    device_name="cuda",
    backend="open_clip"
)
```

#### encode_keyframe_images()
Encode batch của keyframe images thành embeddings.

**Parameters:**
- `model`: Loaded model
- `preprocess`: Preprocessing function
- `device` (torch.device): Device
- `image_paths` (list[Path]): Danh sách đường dẫn ảnh
- `batch_size` (int): Batch size
- `precision` (str): Độ chính xác
- `normalize` (bool): Normalize embeddings (default: True)
- `logger` (logging.Logger | None): Logger

**Returns:**
- `tuple`: (embeddings: np.ndarray, valid_paths: list[Path])

### encoder.py

#### encode_video_frames()
Encode toàn bộ video frames thành embeddings.

**Parameters:**
- `model`: Loaded model
- `preprocess`: Preprocessing function
- `device` (torch.device): Device
- `video_path` (Path): Đường dẫn video
- `output_path` (Path): Output path để lưu embeddings
- `batch_size` (int): Batch size
- `loader_cfg` (FrameLoaderConfig): Frame loader configuration
- `logger` (logging.Logger | None): Logger

**Returns:**
- `np.ndarray`: Video embeddings

## Pipeline API

### ExtractEmbeddingPipeline

**Config structure:**
```python
config = {
    "model": {
        "name": "ViT-L-14",
        "backend": "open_clip",
        "pretrained": "openai",
        "batch_size": 32,
        "normalize": True
    },
    "input_keyframes_root": "/path/to/keyframes",
    "output_embeddings_root": "/path/to/output"
}
```
