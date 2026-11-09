"""Script tạo vector cache cho tất cả models đã build index.

Vector cache giúp temporal search nhanh hơn rất nhiều.

Usage:
    cd backend
    python scripts/create_all_vector_caches.py
"""
import sys
import logging
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def create_vector_cache(index_path: Path, output_path: Path, dtype: str, logger: logging.Logger):
    """Tạo vector cache từ FAISS index."""
    import faiss
    import numpy as np

    if not index_path.exists():
        logger.error(f"Index not found: {index_path}")
        return False

    if output_path.exists():
        logger.warning(f"Vector cache already exists, skipping: {output_path}")
        return True

    try:
        logger.info(f"Loading index: {index_path}")
        index = faiss.read_index(str(index_path))

        logger.info(f"Reconstructing {index.ntotal:,} vectors (dim={index.d})...")
        vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
        index.reconstruct_n(0, index.ntotal, vectors)

        # Normalize
        logger.info("Normalizing vectors...")
        faiss.normalize_L2(vectors)

        # Convert dtype if needed
        if dtype == "float16":
            logger.info("Converting to float16...")
            vectors = vectors.astype(np.float16)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to: {output_path}")
        np.save(str(output_path), vectors)

        size_mb = output_path.stat().st_size / (1024 ** 2)
        logger.info(f"✓ Created vector cache: {size_mb:.2f} MB")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to create vector cache: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Create vector caches for all built indexes")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Vector dtype (default: float32)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing vector caches"
    )
    args = parser.parse_args()

    logger = setup_logging()

    # Danh sách models cần tạo vector cache
    # Format: (model_name, index_path, output_path)
    base_dir = Path(__file__).parent.parent.parent / "data" / "database"

    models = [
        ("siglip-g", "faiss_hnsw_siglip_g"),
        ("pe-core-g", "faiss_hnsw_pe_core_g"),
        ("tipsv2-g", "faiss_hnsw_tipsv2_g"),
        ("siglip2-so400m", "faiss_hnsw_siglip2_so400m_384"),
        ("dfn-vit-h14", "faiss_hnsw_dfn_vit_h14_378"),
    ]

    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Found {len(models)} model(s) to process")
    logger.info("=" * 80)

    results = []
    for name, folder in models:
        logger.info(f"\nProcessing: {name}")
        logger.info("-" * 80)

        index_path = base_dir / folder / "keyframes.faiss"
        output_path = base_dir / folder / f"vectors_{args.dtype}.npy"

        if not index_path.exists():
            logger.warning(f"Index not found, skipping: {index_path}")
            results.append((name, False))
            continue

        if output_path.exists() and not args.force:
            logger.info(f"✓ Vector cache already exists: {output_path}")
            results.append((name, True))
            continue

        success = create_vector_cache(index_path, output_path, args.dtype, logger)
        results.append((name, success))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{status}: {name}")

    total = len(results)
    success_count = sum(1 for _, s in results if s)
    logger.info(f"\nTotal: {success_count}/{total} successful")

    return 0 if success_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
