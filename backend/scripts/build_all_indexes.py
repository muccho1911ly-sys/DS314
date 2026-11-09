"""Script build FAISS indexes cho nhiều models từ config.

Usage:
    cd backend
    python scripts/build_all_indexes.py [--config configs/build_indexes_multi.yaml]
"""
import sys
import logging
from pathlib import Path
import argparse
import yaml

# Add backend dir to path so `from src.retrieval...` works
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.indexer.faiss.build_pipeline import BuildFaissIndexPipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def build_index_for_model(model_cfg: dict, common_cfg: dict, logger: logging.Logger):
    """Build FAISS index cho 1 model."""
    name = model_cfg.get("name", "unknown")
    logger.info("=" * 80)
    logger.info(f"Building index for: {name}")
    logger.info("=" * 80)

    # Merge common settings với model-specific settings
    index_defaults = common_cfg.get("index_defaults", {})

    cfg = {
        "embeddings_root": model_cfg["embeddings_root"],
        "keyframes_root": model_cfg.get("keyframes_root") or common_cfg["keyframes_root"],
        "map_keyframes_root": model_cfg.get("map_keyframes_root") or common_cfg["map_keyframes_root"],
        "collection": model_cfg.get("collection"),
        "datasets": model_cfg.get("datasets"),
        "index": {
            "output_dir": model_cfg["output_dir"],
            "index_name": model_cfg.get("index_name", "keyframes.faiss"),
            "metadata_name": model_cfg.get("metadata_name", "metadata.csv"),
            "type": model_cfg.get("type") or index_defaults.get("type", "hnsw"),
            "metric": model_cfg.get("metric") or index_defaults.get("metric", "cosine"),
            "hnsw_m": model_cfg.get("hnsw_m") or index_defaults.get("hnsw_m", 32),
            "ef_construction": model_cfg.get("ef_construction") or index_defaults.get("ef_construction", 400),
            "ef_search": model_cfg.get("ef_search") or index_defaults.get("ef_search", 399),
        }
    }

    # Create output dir if not exists
    output_dir = Path(cfg["index"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Build index
    try:
        pipeline = BuildFaissIndexPipeline(cfg=cfg, logger=logger)
        pipeline.run()
        logger.info(f"✓ Successfully built index for {name}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to build index for {name}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indexes for multiple models")
    parser.add_argument(
        "--config",
        default="configs/build_indexes_multi.yaml",
        help="Path to config file (default: configs/build_indexes_multi.yaml)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Build only specific models by name (default: all)"
    )
    args = parser.parse_args()

    logger = setup_logging()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models = config.get("models", [])
    if not models:
        logger.error("No models found in config")
        return 1

    # Filter models if specified
    if args.models:
        models = [m for m in models if m.get("name") in args.models]
        if not models:
            logger.error(f"No models matched: {args.models}")
            return 1

    logger.info(f"Found {len(models)} model(s) to build")

    # Build indexes
    results = []
    for i, model_cfg in enumerate(models, 1):
        name = model_cfg.get("name", f"model_{i}")
        logger.info(f"\n[{i}/{len(models)}] Processing: {name}")
        success = build_index_for_model(model_cfg, config, logger)
        results.append((name, success))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("BUILD SUMMARY")
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
