# 📘 Hướng dẫn tích hợp Model Embedding mới

Hướng dẫn đầy đủ để thêm và sử dụng model embedding mới vào hệ thống.

---

## 🎯 Tổng quan

Bạn có 3 model mới đã được thêm vào config:
1. **SigLIP-G** (`siglip-g`)
2. **PE-Core-G** (`pe-core-g`)  
3. **TIPSv2-G** (`tipsv2-g`)

Hiện tại các model này:
- ✅ Đã có embeddings trong `data/embeddings/embeddings_<model>/`
- ⏳ Chưa có FAISS index
- ⏳ Chưa được enable trong config

---

## 📋 BƯỚC 1: Build FAISS Index

### Kiểm tra embeddings có sẵn

```bash
cd D:/new_ERS/event-retrieval-system
ls data/embeddings/
```

Bạn sẽ thấy:
- `embeddings_siglip_g/`
- `embeddings_pe_core_g/`
- `embeddings_tipsv2_g/`

### Build index cho TẤT CẢ 3 models

```bash
cd backend
python scripts/build_all_indexes.py
```

**Hoặc** build từng model riêng:

```bash
# Chỉ build SigLIP-G
python scripts/build_all_indexes.py --models siglip_g

# Chỉ build PE-Core-G
python scripts/build_all_indexes.py --models pe_core_g

# Chỉ build TIPSv2-G
python scripts/build_all_indexes.py --models tipsv2_g

# Build 2 models cùng lúc
python scripts/build_all_indexes.py --models siglip_g pe_core_g
```

### Kết quả sau khi build

Sẽ tạo ra các file trong `data/database/`:
```
data/database/
├── faiss_hnsw_siglip_g/
│   ├── keyframes.faiss
│   ├── metadata.csv
│   └── vectors_fp32.npy (cần tạo riêng, xem bước 1.1)
├── faiss_hnsw_pe_core_g/
│   ├── keyframes.faiss
│   ├── metadata.csv
│   └── vectors_fp32.npy
└── faiss_hnsw_tipsv2_g/
    ├── keyframes.faiss
    ├── metadata.csv
    └── vectors_fp32.npy
```

### Bước 1.1: Tạo vector cache (Optional nhưng khuyến khích)

Vector cache giúp temporal search nhanh hơn nhiều:

```bash
cd backend
python src/retrieval/indexer/faiss/vector_cache.py \
  --index-path ../data/database/faiss_hnsw_siglip_g/keyframes.faiss \
  --output-path ../data/database/faiss_hnsw_siglip_g/vectors_fp32.npy \
  --dtype float32

# Tương tự cho các model khác
python src/retrieval/indexer/faiss/vector_cache.py \
  --index-path ../data/database/faiss_hnsw_pe_core_g/keyframes.faiss \
  --output-path ../data/database/faiss_hnsw_pe_core_g/vectors_fp32.npy \
  --dtype float32

python src/retrieval/indexer/faiss/vector_cache.py \
  --index-path ../data/database/faiss_hnsw_tipsv2_g/keyframes.faiss \
  --output-path ../data/database/faiss_hnsw_tipsv2_g/vectors_fp32.npy \
  --dtype float32
```

---

## 📋 BƯỚC 2: Bật model trong config

Mở file `backend/configs/app.yaml` và tìm model bạn muốn dùng:

### Ví dụ: Bật SigLIP-G

```yaml
    # --- SigLIP-G - MODEL MỚI 1 ---
    - model_key: siglip-g
      enabled: true  # ← ĐỔI TỪ false → true
      backend: open_clip
      model_name: ViT-SO400M-14-SigLIP-384
      # ... rest of config
```

### Ví dụ: Bật PE-Core-G

```yaml
    # --- PE-Core-G14-448 - MODEL MỚI 2 ---
    - model_key: pe-core-g
      enabled: true  # ← ĐỔI TỪ false → true
      backend: perception_encoder
      model_name: PE-Core-G14-448
      # ... rest of config
```

### Ví dụ: Bật TIPSv2-G

```yaml
    # --- TIPSv2-G - MODEL MỚI 3 ---
    - model_key: tipsv2-g
      enabled: true  # ← ĐỔI TỪ false → true
      backend: open_clip
      model_name: ViT-bigG-14
      # ... rest of config
```

### Đặt làm model mặc định (optional)

Nếu muốn model mới làm mặc định cho tất cả searches:

```yaml
semantic:
  default_model_key: siglip-g  # ← ĐỔI tên model_key vào đây
  models:
    # ...
```

---

## 📋 BƯỚC 3: Cập nhật fusion config (nếu dùng fusion search)

Nếu bạn muốn dùng model mới trong fusion search, cập nhật:

```yaml
fusion:
  local_semantic_model: pe-core-g     # ← Model cho fine-grained
  global_semantic_model: siglip-g     # ← Model cho global context
  default_weights:
    semantic: 0.8
    ocr: 0.1
    asr: 0.1
```

**Lưu ý:** Model phải có `enabled: true` mới dùng được trong fusion.

---

## 📋 BƯỚC 4: Kiểm tra model name và backend

### ⚠️ QUAN TRỌNG: Verify model name

Các model name trong config **PHẢI khớp** với:
- Model name trong `backend/src/embedding_extraction/models/registry.py` (MODEL_PRESETS)
- Hoặc model name mà backend hỗ trợ

### Kiểm tra SigLIP-G

Mở file `registry.py` và tìm preset cho SigLIP-G. **Nếu không có**, bạn cần:

1. Thêm vào `MODEL_PRESETS`:
```python
"siglip-g": dict(
    backend=BACKEND_OPEN_CLIP,
    pretrained="webli",  # hoặc tag chính xác
    model_name="ViT-bigG-14-SigLIP-384"  # tên chính xác
),
```

2. Hoặc update config với đúng tên model open_clip hỗ trợ

### Kiểm tra TIPSv2-G

TIPSv2 có thể cần backend riêng hoặc config đặc biệt. Nếu backend hiện tại không support, bạn sẽ cần:

1. Tạo backend mới trong `backend/src/embedding_extraction/models/backends/tipsv2.py`
2. Đăng ký trong `registry.py`

---

## 📋 BƯỚC 5: Restart backend và test

### 5.1. Restart backend

```bash
cd backend
python -m uvicorn src.api.main:app --reload --port 8000
```

### 5.2. Check logs

Xem log khi backend khởi động để verify model load thành công:

```
[VECTOR CACHE] model=siglip-g | mode=memmap | ...
✓ Loaded 3 model(s)
✓ Available models: siglip2-so400m, siglip-g, pe-core-g
```

### 5.3. Test API

```bash
# Test search với model cụ thể
curl -X POST "http://localhost:8000/api/v2/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "người đàn ông mặc áo đỏ",
    "model_key": "siglip-g",
    "top_k": 10
  }'
```

---

## 🎮 Sử dụng Model

### Chọn model khi search

**API Request:**
```json
{
  "query": "query của bạn",
  "model_key": "siglip-g",  // Chọn model
  "top_k": 20
}
```

**Không chỉ định model_key** → dùng `default_model_key` trong config

### Bật/tắt model nhanh

Muốn thử model khác? Chỉ cần:

1. Mở `backend/configs/app.yaml`
2. Tìm model cần bật/tắt
3. Đổi `enabled: true/false`
4. Restart backend

**Không cần build lại index!** Index đã build 1 lần là dùng mãi.

---

## 🔧 Troubleshooting

### Lỗi: "Model not found in registry"

**Nguyên nhân:** Model name trong config không khớp với registry

**Giải pháp:**
1. Check `backend/src/embedding_extraction/models/registry.py`
2. Thêm preset cho model
3. Hoặc sửa `model_name` trong config

### Lỗi: "Index file not found"

**Nguyên nhân:** Chưa build FAISS index

**Giải pháp:** Chạy lại bước 1

### Lỗi: "Vector cache not found"

**Nguyên nhân:** Chưa tạo vector cache

**Giải pháp:** 
- Chạy bước 1.1
- Hoặc set `vector_cache_mode: none` trong config (temporal search sẽ chậm hơn)

### Lỗi: "Backend not supported"

**Nguyên nhân:** Backend chưa được implement

**Giải pháp:**
1. Tạo backend mới trong `backends/<model>.py`
2. Đăng ký trong `registry.py`
3. Hoặc dùng backend tương tự (ví dụ: dùng `open_clip` cho các CLIP-based models)

---

## 📊 So sánh Models

Để test model nào tốt hơn:

1. Enable nhiều models cùng lúc
2. Gọi API với từng model_key
3. So sánh kết quả

Ví dụ:
```bash
# Test với SigLIP2-SO400M
curl ... -d '{"query": "...", "model_key": "siglip2-so400m"}'

# Test với SigLIP-G
curl ... -d '{"query": "...", "model_key": "siglip-g"}'

# Test với PE-Core-G
curl ... -d '{"query": "...", "model_key": "pe-core-g"}'
```

---

## 📝 Checklist tích hợp model mới

- [ ] Có embeddings trong `data/embeddings/embeddings_<model>/`
- [ ] Đã build FAISS index (`python scripts/build_all_indexes.py`)
- [ ] Đã tạo vector cache (optional nhưng nên có)
- [ ] Model name đã được thêm vào `registry.py` hoặc là preset có sẵn
- [ ] Backend được hỗ trợ trong `backends/`
- [ ] Đã set `enabled: true` trong `app.yaml`
- [ ] Đã restart backend
- [ ] Test search thành công

---

## 🚀 Quick Start (TL;DR)

```bash
# 1. Build indexes
cd backend
python scripts/build_all_indexes.py

# 2. Tạo vector cache (optional)
python src/retrieval/indexer/faiss/vector_cache.py \
  --index-path ../data/database/faiss_hnsw_siglip_g/keyframes.faiss \
  --output-path ../data/database/faiss_hnsw_siglip_g/vectors_fp32.npy

# 3. Edit config
# Mở backend/configs/app.yaml
# Đổi enabled: false → true cho model muốn dùng

# 4. Restart backend
python -m uvicorn src.api.main:app --reload --port 8000

# 5. Test
curl -X POST "http://localhost:8000/api/v2/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "model_key": "siglip-g"}'
```

---

## 📚 Tài liệu tham khảo

- `backend/src/embedding_extraction/models/registry.py` - Model registry
- `backend/src/retrieval/index/faiss_index.py` - FAISS index loader
- `backend/configs/build_indexes_multi.yaml` - Build config
- `backend/scripts/build_all_indexes.py` - Build script

---

**Cập nhật:** 2026-09-17
