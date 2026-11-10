# event-retrieval-system
```
event-retrieval-system/
├── backend/
│   ├── configs/
│   │   ├── app.yaml
│   │   ├── embeddings.yaml
│   │   ├── indexing.yaml
│   │   └── kf_extraction.yaml
│   ├── data/
│   │   ├── database/
│   │   │   └── faiss_clip_vitb16_quickgelu/
│   │   ├── embeddings/
│   │   ├── processed/
│   │   │   ├── embeddings/
│   │   │   │   └── L21_a/
│   │   │   ├── embeddings_clip_vitb16_quickgelu/
│   │   │   │   ├── L21/
│   │   │   │   │   ├── L21_V001/
│   │   │   ├── keyframes/
│   │   │   │   ├── L21/
│   │   │   │   │   ├── L21_V001/
│   │   │   ├── map_keyframes/
│   │   │   │   ├── L21/
│   │   │   └── scenes/
│   │   │       └── L21_a/
│   │   │           └── L21_V002.scenes.txt
│   │   └── raw/
│   │       └── videos/
│   │           └── L21/
│   ├── external/
│   │   ├── temp/
│   │   └── TransNetV2/
│   ├── logs/
│   ├── src/
│   │   ├── core/
│   │   │   ├── clustering.py
│   │   │   └── selector.py
│   │   ├── dataset/
│   │   │   └── frame_loader.py
│   │   ├── index/
│   │   │   ├── __init__.py
│   │   │   ├── faiss_index.py
│   │   │   └── retrieval_system.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── detector.py
│   │   │   ├── embedder.py
│   │   │   └── encoder.py
│   │   ├── pipelines/
│   │   │   ├── __init__.py
│   │   │   ├── build_faiss.py
│   │   │   ├── extract_embeddings.py
│   │   │   └── extract_keyframes.py
│   │   ├── ui/
│   │   │   ├── __init__.py
│   │   │   ├── frame_context.py
│   │   │   ├── query_parser.py
│   │   │   └── scoring.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── config.py
│   │       ├── device.py
│   │       ├── logger.py
│   │       ├── seed.py
│   │       └── video_io.py
│   ├── weights/
│   ├── __init__.py
│   ├── app.py
│   ├── cli.py
│   ├── extract_kf.py
│   ├── main.py
│   ├── padding.py
│   ├── README.md
│   ├── requirements.txt
│   ├── src_embeddings.py
│   ├── src_faiss.py
│   └── src_system.py
├── frontend/
│   ├── public/
│   │   ├── favicon.svg
│   │   └── icons.svg
│   ├── src/
│   │   ├── api/
│   │   │   └── retrievalAPI.js
│   │   ├── assets/
│   │   │   ├── react.svg
│   │   │   └── vite.svg
│   │   ├── components/
│   │   │   ├── DetailPanel.jsx
│   │   │   ├── GroupedResults.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   ├── ResultGrid.jsx
│   │   │   ├── ResultToolbar.jsx
│   │   │   ├── SearchBar.jsx
│   │   │   ├── SettingsPanel.jsx
│   │   │   ├── Sidebar.jsx
│   │   │   └── VideoModal.jsx
│   │   ├── hooks/
│   │   │   └── useRetrievalSearch.js
│   │   ├── utils/
│   │   │   ├── frameUtils.js
│   │   │   ├── groupByVideo.js
│   │   │   └── surroundingFrames.js
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── eslint.config.js
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── README.md
│   └── vite.config.js
└── scan.py
```


## Chạy DS314 trên Windows

Dùng Python 3.13 (64 bit) và Node.js 24. Máy chạy Qwen3-VL 4-bit cần GPU NVIDIA có CUDA hoạt động.

```powershell
git clone https://github.com/muccho1911ly-sys/DS314.git
cd DS314
```

Đặt bộ dữ liệu được cung cấp riêng vào `data/DS314/` ở gốc repo. Cần thư mục
`keyframes/`, `map-keyframes/` và `database/`. Thêm `videos/` để phát video.
Với cấu hình mặc định, `database/` cần các thư mục `faiss_hnsw_siglip2_so400m`,
`faiss_hnsw_siglip2_giant`, `faiss_hnsw_qwen3_vl_2b` và `faiss_hnsw_jina_clip_v2`.
Mỗi thư mục cần `keyframes.faiss`, `metadata.csv` và `vectors_fp32.npy`.
Thư mục `data/DS314/experiments/` chỉ dùng khi cần đối chiếu hoặc import lại embedding.
Khi đã có database, bạn có thể chạy backend, frontend và test truy vấn với các thư mục ở trên.

### Backend

Từ gốc repo, chạy trong PowerShell:

```powershell
cd backend
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r ..\requirements.txt
$env:HF_HOME = "$PWD\.cache\huggingface"
$env:HF_HUB_DISABLE_XET = "1"
.\venv\Scripts\python.exe scripts\setup_ds314_large_models.py
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Script setup chuẩn bị runtime và tải weights cho hai model SigLIP2 và Qwen3-VL.
Chạy setup lần đầu hoặc khi cần tải lại cache. Jina CLIP tải weights ở truy vấn
đầu tiên dùng model đó. Kiểm tra [API health](http://127.0.0.1:8000/api/health)
trả về trạng thái `ok`.

### Frontend

Mở PowerShell thứ hai tại gốc repo:

```powershell
cd frontend
'VITE_API_BASE_URL=/' | Set-Content -Encoding ascii .env.local
npm ci
npm run dev
```

Mở [frontend](http://127.0.0.1:5173/). Vite chuyển tiếp `/api` và `/static` đến backend cổng 8000.


Lưu ý
```
Auto split sẽ giúp cho truy vấn các chi tiết siêu nhỏ tốt hơn, nhưng sẽ lâu và tốn thời gian hơn (tắt đi thì search bằng keyword , , vẫn bình thường)

Semantic Search : search ngữ nghĩa --> nhập query bất kì kh có dấu . hay ;

Temporal Search : cách các event bở dấu ; hoặc . và search keyword trong từng event vẫn oke --> đặc biệt chinh3 duration khác trừ 1 giúp khoanh vùng móc thời gian chính xác hơn

Auto Search : đơn gian là kết hợp semantic và temporal nếu không có bất kì dấu : hoặc ; thì tự động semantic còn có event thì nhảy qua temporal mặc định là -1 (search toàn bộ vid)

```
