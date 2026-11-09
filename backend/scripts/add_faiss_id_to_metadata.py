"""Script add _faiss_id to metadata.csv of FAISS indexes."""
from pathlib import Path
import pandas as pd
import sys

# List of models
models = [
    ("siglip-g", "D:/new_ERS/event-retrieval-system/data/database/faiss_hnsw_siglip_g"),
    ("pe-core-g", "D:/new_ERS/event-retrieval-system/data/database/faiss_hnsw_pe_core_g"),
    ("tipsv2-g", "D:/new_ERS/event-retrieval-system/data/database/faiss_hnsw_tipsv2_g"),
]

for model_name, db_dir in models:
    metadata_path = Path(db_dir) / "metadata.csv"

    if not metadata_path.exists():
        print("[X] {}: metadata.csv not found at {}".format(model_name, metadata_path))
        continue

    # Read metadata
    df = pd.read_csv(metadata_path)

    # Check if _faiss_id already exists
    if "_faiss_id" in df.columns:
        print("[OK] {}: _faiss_id already exists".format(model_name))
        continue

    # Add _faiss_id column (index 0, 1, 2, ...)
    df["_faiss_id"] = range(len(df))

    # Save back
    df.to_csv(metadata_path, index=False)

    print("[OK] {}: added _faiss_id column ({} rows)".format(model_name, len(df)))

print("\n[OK] Done!")
