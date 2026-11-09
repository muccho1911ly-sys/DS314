"""TIPSv2 backend for google/tipsv2-g14.

TIPSv2-G is a Hugging Face Transformers model, NOT an OpenCLIP model.
Official model card: https://huggingface.co/google/tipsv2-g14

This backend loads the model using AutoModel.from_pretrained with trust_remote_code=True,
following the official implementation pattern.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModel, AutoProcessor

from src.embedding_extraction.models.backends.base import EncoderWrapper, LoadedModel
from src.utils.device import resolve_device


def load(
    model_name: str,
    pretrained: str,
    precision: str = "fp32",
    device_name: str = "auto",
    logger: logging.Logger | None = None,
    **_: object,
) -> LoadedModel:
    """Load TIPSv2-G model from google/tipsv2-g14.

    Args:
        model_name: Model identifier (e.g., "tipsv2-g14")
        pretrained: HuggingFace model ID (e.g., "google/tipsv2-g14")
        precision: One of "fp32", "fp16", "bf16", "amp"
        device_name: Device to load model on ("auto", "cpu", "cuda", etc.)
        logger: Optional logger for diagnostics

    Returns:
        LoadedModel with model, preprocessor, and metadata
    """
    device = resolve_device(device_name)

    # Force fp32 on CPU
    if device.type == "cpu" and precision in {"fp16", "amp", "bf16"}:
        if logger:
            logger.warning(
                "[tipsv2] precision=%s not supported on CPU, falling back to fp32",
                precision,
            )
        precision = "fp32"

    # Determine torch dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "amp": torch.float16,  # Will use autocast
    }
    torch_dtype = dtype_map.get(precision, torch.float32)

    if logger:
        logger.info(
            "[tipsv2] Loading TIPSv2-G from %s on %s precision=%s dtype=%s",
            pretrained, device, precision, torch_dtype,
        )

    # Load model and processor
    model = AutoModel.from_pretrained(
        pretrained,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model = model.to(device).eval()

    processor = AutoProcessor.from_pretrained(
        pretrained,
        trust_remote_code=True,
    )

    # Get embedding dimension
    embedding_dim = getattr(model.config, "projection_dim", None) or \
                   getattr(model.config, "hidden_size", None) or \
                   getattr(model, "text_embed_dim", None) or \
                   1536  # TIPSv2-G default (actual dimension is 1536, not 1664)

    if logger:
        logger.info(
            "[tipsv2] Model loaded successfully:"
        )
        logger.info("  - Model class: %s", model.__class__.__name__)
        logger.info("  - Device: %s", device)
        logger.info("  - Dtype: %s", torch_dtype)
        logger.info("  - Embedding dimension: %s", embedding_dim)

    # Setup autocast for mixed precision
    use_autocast = device.type == "cuda" and precision in {"fp16", "amp", "bf16"}
    autocast_dtype = torch.float16 if precision in {"fp16", "amp"} else torch.bfloat16

    @torch.inference_mode()
    def _encode_image(batch: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings.

        Args:
            batch: Tensor of shape (B, C, H, W) with preprocessed images

        Returns:
            Tensor of shape (B, embedding_dim) with normalized embeddings
        """
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            # TIPSv2 returns TIPSv2ImageOutput with cls_token, register_tokens, patch_tokens
            outputs = model.encode_image(pixel_values=batch)
            # Use cls_token as the image embedding: (B, 1, D) -> (B, D)
            embeddings = outputs.cls_token.squeeze(1)
            # Normalize for cosine similarity
            return torch.nn.functional.normalize(embeddings, dim=-1)

    @torch.inference_mode()
    def _encode_text(texts: list[str]) -> torch.Tensor:
        """Encode text queries to embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            Tensor of shape (len(texts), embedding_dim) with normalized embeddings
        """
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            # TIPSv2 encode_text accepts strings directly and handles tokenization internally
            outputs = model.encode_text(texts)
            # Normalize for cosine similarity
            return torch.nn.functional.normalize(outputs, dim=-1)

    # Create preprocessing function compatible with the pipeline
    def preprocess(image):
        """Preprocess PIL image for TIPSv2.

        Args:
            image: PIL Image

        Returns:
            Preprocessed tensor ready for the model
        """
        inputs = processor(images=image, return_tensors="pt")
        # Return just the pixel_values tensor, already on CPU
        return inputs["pixel_values"].squeeze(0)

    return LoadedModel(
        model=EncoderWrapper(_encode_image, _encode_text, backend_label="tipsv2"),
        preprocess=preprocess,
        device=device,
        precision=precision,
        embedding_dim=embedding_dim,
        supports_text=True,
    )
