# 🎯 TÓM TẮT: Tích hợp 3 Model Embedding Mới

## ✅ Đã hoàn thành

### 1. **Config Files**
- ✅ `backend/configs/app.yaml` - Đã thêm 3 models mới với cấu hình đầy đủ
- ✅ `backend/configs/build_indexes_multi.yaml` - Config để build FAISS indexes

### 2. **Scripts**
- ✅ `backend/scripts/build_all_indexes.py` - Build indexes cho nhiều models
- ✅ `backend/scripts/create_all_vector_caches.py` - Tạo vector caches

### 3. **Documentation**
- ✅ `backend/HƯỚNG_DẪN_THÊM_MODEL.md` - Hướng dẫn chi tiết đầy đủ

## 🆕 3 Models đã thêm

| Model | Model Key | Backend | Status | Index Path |
|-------|-----------|---------|--------|------------|
| **SigLIP-G** | `siglip-g` | open_clip | `enabled: false` | `data/database/faiss_hnsw_siglip_g/` |
| **PE-Core-G** | `pe-core-g` | perception_encoder | `enabled: false` | `data/database/faiss_hnsw_pe_core_g/` |
| **TIPSv2-G** | `tipsv2-g` | open_clip | `enabled: false` | `data/database/faiss_hnsw_tipsv2_g/` |

## 📋 CÁC BƯỚC CẦN LÀM TIẾP

### Bước 1: Build FAISS Indexes (BẮT BUỘC)

```bash
cd backend
python scripts/build_all_indexes.py
```

Hoặc build từng model:
```bash
python scripts/build_all_indexes.py --models siglip_g
python scripts/build_all_indexes.py --models pe_core_g  
python scripts/build_all_indexes.py --models tipsv2_g
```

### Bước 2: Tạo Vector Caches (Khuyến khích)

```bash
cd backend
python scripts/create_all_vector_caches.py
```

### Bước 3: Verify Model Names (QUAN TRỌNG)

⚠️ **Cần kiểm tra model names có đúng không!**

Các model names trong `app.yaml` hiện tại:
- `siglip-g`: `ViT-SO400M-14-SigLIP-384` 
- `pe-core-g`: `PE-Core-G14-448`
- `tipsv2-g`: `ViT-bigG-14`

**Cần verify:**
1. Mở `backend/src/embedding_extraction/models/registry.py`
2. Check xem các model names này có trong `MODEL_PRESETS` không
3. Nếu KHÔNG có, cần thêm preset hoặc sửa lại model name

### Bước 4: Bật Model muốn dùng

Mở `backend/configs/app.yaml` và đổi `enabled: false → true`:

```yaml
# Ví dụ bật SigLIP-G
- model_key: siglip-g
  enabled: true  # ← Đổi thành true
  # ... rest
```

### Bước 5: Update Fusion Config (Optional)

Nếu dùng fusion search, update:

```yaml
fusion:
  local_semantic_model: pe-core-g      # ← Đổi model key
  global_semantic_model: siglip-g      # ← Đổi model key
```

### Bước 6: Restart Backend và Test

```bash
cd backend
python -m uvicorn src.api.main:app --reload --port 8000
```

Test search:
```bash
curl -X POST "http://localhost:8000/api/v2/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "người đàn ông mặc áo đỏ", "model_key": "siglip-g"}'
```

## 🎮 Sử dụng

### Bật/Tắt model nhanh

```yaml
# File: backend/configs/app.yaml

# BẬT model
- model_key: siglip-g
  enabled: true   # ← Chỉ cần đổi dòng này

# TẮT model  
- model_key: pe-core-g
  enabled: false  # ← Đổi về false
```

Sau đó restart backend là xong!

### Chọn model khi search

```json
{
  "query": "your query",
  "model_key": "siglip-g",  // Chọn model
  "top_k": 20
}
```

Không chỉ định `model_key` → dùng `default_model_key` trong config

## 📁 Cấu trúc Files

```
backend/
├── configs/
│   ├── app.yaml                        # ← Main config (đã update)
│   └── build_indexes_multi.yaml        # ← Build config (mới)
├── scripts/
│   ├── build_all_indexes.py           # ← Build script (mới)
│   └── create_all_vector_caches.py    # ← Cache script (mới)
└── HƯỚNG_DẪN_THÊM_MODEL.md           # ← Documentation (mới)

data/
├── embeddings/                         # ← Embeddings có sẵn
│   ├── embeddings_siglip_g/
│   ├── embeddings_pe_core_g/
│   └── embeddings_tipsv2_g/
└── database/                           # ← Sẽ tạo sau khi build
    ├── faiss_hnsw_siglip_g/           # ← Cần build
    ├── faiss_hnsw_pe_core_g/          # ← Cần build
    └── faiss_hnsw_tipsv2_g/           # ← Cần build
```

## ⚠️ Lưu ý quan trọng

1. **Model Names**: PHẢI verify model names trong `registry.py` trước khi chạy
2. **Backends**: Nếu TIPSv2 cần backend riêng, cần implement trước
3. **Index Files**: Build index CHỈ 1 LẦN, sau đó dùng mãi
4. **Vector Cache**: Optional nhưng rất nên có cho temporal search
5. **Fusion**: Model phải `enabled: true` mới dùng được trong fusion

## 🔗 Tài liệu tham khảo

- Chi tiết đầy đủ: `backend/HƯỚNG_DẪN_THÊM_MODEL.md`
- Model registry: `backend/src/embedding_extraction/models/registry.py`
- FAISS index: `backend/src/retrieval/index/faiss_index.py`

## ✨ Quick Commands

```bash
# Build tất cả indexes
cd backend && python scripts/build_all_indexes.py

# Tạo tất cả vector caches
python scripts/create_all_vector_caches.py

# Restart backend
python -m uvicorn src.api.main:app --reload --port 8000

# Test
curl -X POST "http://localhost:8000/api/v2/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "model_key": "siglip-g"}'
```

---

**Tạo bởi:** Claude Code
**Ngày:** 2026-09-17
