import os
os.environ["OMP_NUM_THREADS"] = "8"
import sys
import faiss
import numpy as np
import time

def build_cache(name, index_path, out_path):
    print(f"\n[{name}] Loading {index_path} ...", flush=True)
    t0 = time.time()
    idx = faiss.read_index(index_path)
    print(f"[{name}] ntotal={idx.ntotal:,} d={idx.d}  ({time.time()-t0:.1f}s)", flush=True)
    
    print(f"[{name}] Reconstructing vectors...", flush=True)
    t1 = time.time()
    vectors = np.empty((idx.ntotal, idx.d), dtype=np.float32)
    idx.reconstruct_n(0, idx.ntotal, vectors)
    print(f"[{name}] Reconstruct done ({time.time()-t1:.1f}s)", flush=True)
    
    print(f"[{name}] Normalizing...", flush=True)
    faiss.normalize_L2(vectors)
    
    print(f"[{name}] Saving {out_path} ...", flush=True)
    np.save(out_path, vectors)
    mb = os.path.getsize(out_path) / 1024**2
    print(f"[{name}] Done! {mb:.1f} MB  total={time.time()-t0:.1f}s", flush=True)

base = "D:/new_ERS/event-retrieval-system/data"
build_cache(
    "AIC",
    f"{base}/AIC/database/faiss_hnsw_siglip2_gopt16_384/keyframes.faiss",
    f"{base}/AIC/database/faiss_hnsw_siglip2_gopt16_384/vectors_fp32.npy"
)
build_cache(
    "CAM",
    f"{base}/CAM/database/faiss_hnsw_siglip2_gopt16_384/keyframes.faiss",
    f"{base}/CAM/database/faiss_hnsw_siglip2_gopt16_384/vectors_fp32.npy"
)
print("\nAll done!", flush=True)
