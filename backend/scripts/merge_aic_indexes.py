"""Merge AIC FAISS indexes: M-series (siglip2_gopt16_384) + L-series (siglip_g).

Uses existing vectors_fp32.npy from both indexes — much faster than re-reading
241k individual .npy files.

Usage:
    cd backend
    python scripts/merge_aic_indexes.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import sys
import csv
import time
from pathlib import Path

import faiss
import numpy as np

BASE = Path("D:/new_ERS/event-retrieval-system/data/AIC/database")
GOPT_DIR  = BASE / "faiss_hnsw_siglip2_gopt16_384"   # M-series, correct paths
SIGLIPG_DIR = BASE / "faiss_hnsw_siglip_g"            # L-series, needs path fix
OUTPUT_DIR  = GOPT_DIR                                  # overwrite in-place

AIC_KEYFRAMES_ROOT = "D:\\new_ERS\\event-retrieval-system\\data\\AIC\\keyframes"
AIC_MAP_ROOT       = "D:\\new_ERS\\event-retrieval-system\\data\\AIC\\map-keyframes"
OLD_KF_PREFIX  = "D:\\new_ERS\\event-retrieval-system\\data\\processed\\keyframes\\"
OLD_MAP_PREFIX = "D:\\new_ERS\\event-retrieval-system\\data\\processed\\map_keyframes\\"


def fix_siglipg_row(row: dict) -> dict:
    """Convert siglip_g metadata row to gopt16_384 format."""
    kf = row.get("keyframe_path", "").replace("/", "\\")
    mp = row.get("map_path", "").replace("/", "\\")

    # Fix keyframe_path: old prefix → AIC path, .jpg → .webp
    if kf.startswith(OLD_KF_PREFIX):
        rel = kf[len(OLD_KF_PREFIX):]
    else:
        rel = kf
    rel_webp = Path(rel).with_suffix(".webp").as_posix().replace("/", "\\")
    new_kf = AIC_KEYFRAMES_ROOT + "\\" + rel_webp

    # Fix map_path
    if mp.startswith(OLD_MAP_PREFIX):
        rel_map = mp[len(OLD_MAP_PREFIX):]
    else:
        rel_map = mp
    new_map = AIC_MAP_ROOT + "\\" + rel_map

    kid = row.get("keyframe_id", "0")
    try:
        kid_int = int(kid)
        kid_str = str(kid_int).zfill(6)
    except ValueError:
        kid_int = 0
        kid_str = str(kid).zfill(6)

    return {
        "collection": "aic",
        "dataset": row.get("dataset", ""),
        "video_id": row.get("video_id", ""),
        "keyframe_id": kid_str,
        "keyframe_id_int": str(kid_int),
        "source_name": kid_str,
        "frame_idx": row.get("frame_idx", ""),
        "timestamp_sec": row.get("timestamp_sec", ""),
        "fps": row.get("fps", ""),
        "keyframe_path": new_kf,
        "embedding_path": row.get("embedding_path", ""),
        "map_path": new_map,
    }


def main():
    t0 = time.time()

    # ── 1. Load M-series vcache ──────────────────────────────────────────────
    print("Loading M-series vectors_fp32.npy ...")
    m_vcache = np.load(str(GOPT_DIR / "vectors_fp32.npy"))
    print(f"  M-series: {m_vcache.shape}")  # (125851, 1536)

    # ── 2. Load L-series vcache ──────────────────────────────────────────────
    print("Loading L-series vectors_fp32.npy ...")
    l_vcache = np.load(str(SIGLIPG_DIR / "vectors_fp32.npy"))
    print(f"  L-series: {l_vcache.shape}")  # (115151, 1536)

    if m_vcache.shape[1] != l_vcache.shape[1]:
        print(f"ERROR: dimension mismatch {m_vcache.shape[1]} vs {l_vcache.shape[1]}")
        sys.exit(1)

    # ── 3. Combine vectors ───────────────────────────────────────────────────
    print("Stacking vectors ...")
    combined = np.vstack([m_vcache, l_vcache]).astype(np.float32)
    print(f"  Combined: {combined.shape}")
    del m_vcache, l_vcache

    # ── 4. Load M-series metadata ────────────────────────────────────────────
    print("Loading M-series metadata ...")
    with open(GOPT_DIR / "metadata.csv", newline="", encoding="utf-8-sig") as f:
        m_reader = csv.DictReader(f)
        m_fields = m_reader.fieldnames
        m_rows = list(m_reader)
    print(f"  M-series rows: {len(m_rows)}")

    # ── 5. Load + fix L-series metadata ─────────────────────────────────────
    print("Loading + fixing L-series metadata ...")
    with open(SIGLIPG_DIR / "metadata.csv", newline="", encoding="utf-8-sig") as f:
        l_reader = csv.DictReader(f)
        l_rows = [fix_siglipg_row(row) for row in l_reader]
    print(f"  L-series rows: {len(l_rows)}")

    if len(m_rows) + len(l_rows) != combined.shape[0]:
        print(f"WARNING: metadata count {len(m_rows)+len(l_rows)} != vectors {combined.shape[0]}")

    # ── 6. Build HNSW ────────────────────────────────────────────────────────
    d = combined.shape[1]
    M, ef_c, ef_s = 32, 400, 399
    print(f"Building HNSW (M={M}, ef_construction={ef_c}, ntotal={combined.shape[0]}) ...")
    faiss.normalize_L2(combined)

    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_c
    index.hnsw.efSearch = ef_s
    index.add(combined)
    print(f"  Built: ntotal={index.ntotal}")
    del combined

    # ── 7. Save index ────────────────────────────────────────────────────────
    faiss_out = str(OUTPUT_DIR / "keyframes.faiss")
    print(f"Saving FAISS index → {faiss_out}")
    faiss.write_index(index, faiss_out)
    del index

    # ── 8. Save combined metadata ────────────────────────────────────────────
    meta_out = str(OUTPUT_DIR / "metadata.csv")
    print(f"Saving metadata.csv → {meta_out}")
    all_rows = m_rows + l_rows
    with open(meta_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=m_fields)
        writer.writeheader()
        for i, row in enumerate(all_rows):
            row["_faiss_id"] = str(i)
            # Only write fields present in m_fields
            writer.writerow({k: row.get(k, "") for k in m_fields})
    print(f"  Saved {len(all_rows)} rows")

    # ── 9. Build new vcache ──────────────────────────────────────────────────
    print("Rebuilding vectors_fp32.npy from merged index ...")
    idx2 = faiss.read_index(faiss_out)
    ntotal = idx2.ntotal
    vcache = np.zeros((ntotal, d), dtype=np.float32)
    idx2.reconstruct_n(0, ntotal, vcache)
    vcache_out = str(OUTPUT_DIR / "vectors_fp32.npy")
    np.save(vcache_out, vcache)
    print(f"  vcache saved: {vcache.shape}, {Path(vcache_out).stat().st_size/1e6:.1f} MB")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. Index has {ntotal} frames (M+L series).")


if __name__ == "__main__":
    main()
