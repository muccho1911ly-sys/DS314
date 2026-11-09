"""Test loading siglip2-so400m to see if it's specific to siglip-g."""
import sys
import time
print("Testing SigLIP2-SO400M...", flush=True)

try:
    print("Importing open_clip...", flush=True)
    import open_clip

    print("Creating model ViT-SO400M-16-SigLIP2-384...", flush=True)
    start = time.time()

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-SO400M-16-SigLIP2-384",
        pretrained="webli",
        precision="fp32",
    )

    elapsed = time.time() - start
    print(f"Model created in {elapsed:.2f}s", flush=True)

    print("Moving to CPU...", flush=True)
    model = model.to("cpu").eval()
    print("SUCCESS! SigLIP2-SO400M loaded fine.", flush=True)

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
