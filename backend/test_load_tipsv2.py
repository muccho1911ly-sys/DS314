#!/usr/bin/env python3
"""Test TIPSv2-G model loading."""

import sys
import torch
import open_clip

print("Testing TIPSv2-G model loading...")
print(f"OpenCLIP version: {open_clip.__version__}")
print(f"PyTorch version: {torch.__version__}")

model_name = "ViT-bigG-14"
pretrained = "laion2b_s39b_b160k"

print(f"\nLoading model: {model_name}")
print(f"Pretrained: {pretrained}")

try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision="fp32",
    )
    print("OK Model created")

    device = torch.device("cpu")
    model = model.to(device).eval()
    print(f"OK Model on {device}")

    print("\nSUCCESS - TIPSv2-G loaded!")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
