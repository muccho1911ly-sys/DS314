"""Test script to debug siglip-g loading issue."""
import sys
import time
print("Starting test...", flush=True)

try:
    print("Importing open_clip...", flush=True)
    import open_clip

    print("Creating model...", flush=True)
    start = time.time()

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-gopt-16-SigLIP2-384",
        pretrained="webli",
        precision="fp32",
    )

    elapsed = time.time() - start
    print(f"Model created in {elapsed:.2f}s", flush=True)

    print("Moving to CPU...", flush=True)
    model = model.to("cpu").eval()
    print("Success! Model loaded.", flush=True)

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
