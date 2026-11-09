"""Rename S-series video_id from S01_Vxxx → S01-Vxxx across:
  1. keyframes folders
  2. map-keyframes folders + CSV files inside
  3. metadata.csv in the FAISS index

Videos are already named S01-V001.mp4 (dashes), so skip.

Usage: python scripts/rename_s_series.py
"""
import csv
import os
import time
from pathlib import Path

ROOT = Path("D:/new_ERS/event-retrieval-system/data/AIC")
META = Path("D:/new_ERS/event-retrieval-system/data/AIC/database/faiss_hnsw_siglip2_gopt16_384/metadata.csv")


def rename_s_folders(base_dir: Path, label: str):
    s_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.upper().startswith("S")]
    for s_dir in sorted(s_dirs):
        vid_dirs = sorted(s_dir.iterdir())
        renamed = 0
        for vid in vid_dirs:
            if "_V" in vid.name:
                new_name = vid.name.replace("_V", "-V", 1)
                new_path = vid.parent / new_name
                if not new_path.exists():
                    vid.rename(new_path)
                    renamed += 1
                    # Also rename CSVs/files inside (for map-keyframes)
                    if new_path.is_dir():
                        for f in new_path.iterdir():
                            if "_V" in f.name:
                                f.rename(f.parent / f.name.replace("_V", "-V", 1))
        print(f"  [{label}] {s_dir.name}: renamed {renamed} items")


def patch_metadata():
    print(f"Reading metadata: {META}")
    with open(META, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    print(f"  Total rows: {len(rows)}")

    patched = 0
    for row in rows:
        if row.get("dataset", "").upper().startswith("S"):
            vid = row.get("video_id", "")
            if "_V" in vid:
                row["video_id"] = vid.replace("_V", "-V", 1)
                patched += 1
            kp = row.get("keyframe_path", "")
            if "_V" in kp:
                row["keyframe_path"] = kp.replace("_V", "-V", 1)

    print(f"  Patched video_id: {patched}")
    tmp = META.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(META)
    print(f"  Saved → {META}")


def main():
    t0 = time.time()

    print("1. Renaming keyframes folders...")
    rename_s_folders(ROOT / "keyframes", "keyframes")

    print("2. Renaming map-keyframes folders + CSVs...")
    rename_s_folders(ROOT / "map-keyframes", "map-keyframes")

    print("3. Patching metadata.csv...")
    patch_metadata()

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
