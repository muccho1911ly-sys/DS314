"""Patch L-series rows in merged metadata.csv with correct frame_idx/timestamp_sec/fps
from map-keyframe CSVs. Runs in-place — no HNSW rebuild needed.

Usage:
    cd backend
    python scripts/patch_aic_metadata_timestamps.py
"""
import csv
import time
from pathlib import Path
from collections import defaultdict

META_PATH   = Path("D:/new_ERS/event-retrieval-system/data/AIC/database/faiss_hnsw_siglip2_gopt16_384/metadata.csv")
MAP_ROOT    = Path("D:/new_ERS/event-retrieval-system/data/AIC/map-keyframes")

def load_map_cache():
    """Build lookup: (video_id, keyframe_id_int) → {frame_idx, timestamp_sec, fps}"""
    cache = {}
    csvs = list(MAP_ROOT.rglob("*.csv"))
    print(f"Loading {len(csvs)} map-keyframe CSVs ...")
    for csv_path in csvs:
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row.get("video_id", "").strip()
                    kid = row.get("keyframe_id", "").strip()
                    try:
                        kid_int = int(kid)
                    except ValueError:
                        continue
                    cache[(vid, kid_int)] = {
                        "frame_idx": row.get("frame_idx", ""),
                        "timestamp_sec": row.get("timestamp_sec", ""),
                        "fps": row.get("fps", ""),
                    }
        except Exception as e:
            print(f"  Warning: {csv_path.name}: {e}")
    print(f"Loaded {len(cache)} keyframe entries")
    return cache


def main():
    t0 = time.time()
    map_cache = load_map_cache()

    print(f"Reading metadata: {META_PATH}")
    with open(META_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    print(f"  Total rows: {len(rows)}")

    patched = 0
    missing = 0
    for row in rows:
        dataset = row.get("dataset", "")
        if not dataset.upper().startswith("L"):
            continue  # skip M/S series — already correct
        vid = row.get("video_id", "")
        try:
            kid_int = int(row.get("keyframe_id_int") or row.get("keyframe_id", "0"))
        except ValueError:
            kid_int = 0

        info = map_cache.get((vid, kid_int))
        if info:
            row["frame_idx"]     = info["frame_idx"]
            row["timestamp_sec"] = info["timestamp_sec"]
            row["fps"]           = info["fps"]
            patched += 1
        else:
            missing += 1

    print(f"  Patched: {patched}, missing in map-cache: {missing}")

    tmp = META_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(META_PATH)
    print(f"  Saved → {META_PATH}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
