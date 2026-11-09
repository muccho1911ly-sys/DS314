"""Debug script to test search and see full error traceback"""
import sys
import traceback
import yaml
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def test_search():
    print("Loading config...")
    with open("configs/app.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Building retrieval system...")
    from src.retrieval.system import build_system
    system = build_system(config)

    print(f"Available models: {system.available_models()}")

    print("\nTesting semantic search with siglip-g...")
    try:
        results, plan = system.search_semantic(
            query="test",
            top_k=10,
            model_key="siglip-g"
        )
        print(f"Success! Got {len(results)} results")
        print(f"Query plan: {plan}")
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_search()
