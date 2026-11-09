#!/usr/bin/env python3
"""Extract vectors from FAISS index to .npy file for vector cache."""

import sys
import argparse
from pathlib import Path
import numpy as np
import faiss

def extract_vectors(index_path: Path, output_path: Path, dtype: str = "float32"):
    """Extract all vectors from FAISS index and save to .npy file."""
    print(f"Loading FAISS index from: {index_path}")
    index = faiss.read_index(str(index_path))

    n_vectors = index.ntotal
    print(f"Index contains {n_vectors} vectors")

    # Reconstruct all vectors
    print("Extracting vectors...")
    vectors = np.zeros((n_vectors, index.d), dtype=np.float32)
    for i in range(n_vectors):
        vectors[i] = index.reconstruct(i)

    # Convert dtype if needed
    if dtype != "float32":
        print(f"Converting to {dtype}...")
        vectors = vectors.astype(dtype)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {output_path}")
    np.save(output_path, vectors)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved {n_vectors} vectors ({vectors.shape[1]}D) as {dtype}")
    print(f"  File size: {file_size_mb:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="Extract vectors from FAISS index")
    parser.add_argument("index_path", help="Path to FAISS index file")
    parser.add_argument("output_path", help="Output .npy file path")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                       help="Output dtype (default: float32)")
    args = parser.parse_args()

    index_path = Path(args.index_path)
    output_path = Path(args.output_path)

    if not index_path.exists():
        print(f"Error: Index file not found: {index_path}")
        return 1

    extract_vectors(index_path, output_path, args.dtype)
    return 0

if __name__ == "__main__":
    sys.exit(main())
