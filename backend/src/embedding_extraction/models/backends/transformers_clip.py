"""Hugging Face CLIP/SigLIP checkpoints, shared by text and image search."""
from pathlib import Path

import torch

from src.embedding_extraction.models.backends.base import EncoderWrapper, LoadedModel
from src.utils.device import resolve_device


def feature_tensor(output):
    # Transformers 4 returns a tensor; Transformers 5 returns a model output.
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


def load(model_name, pretrained=None, precision='fp32', device_name='auto', logger=None,
         cache_dir=None, revision=None, **_):
    from transformers import AutoModel, AutoProcessor

    checkpoint = pretrained or model_name
    device = resolve_device(device_name)
    cache_dir = cache_dir or str(Path(__file__).resolve().parents[4] / '.cache/huggingface')
    options = dict(cache_dir=cache_dir, revision=revision)
    model = AutoModel.from_pretrained(checkpoint, **options).to(device).eval()
    processor = AutoProcessor.from_pretrained(checkpoint, **options)
    is_siglip = model.config.model_type in {'siglip', 'siglip2'}
    dim = model.config.text_config.hidden_size if is_siglip else model.config.projection_dim
    max_length = model.config.text_config.max_position_embeddings

    def preprocess(image):
        return processor.image_processor(images=image, return_tensors='pt')['pixel_values'][0]

    @torch.inference_mode()
    def encode_image(batch):
        return feature_tensor(model.get_image_features(pixel_values=batch.to(device)))

    @torch.inference_mode()
    def encode_text(texts):
        tokens = processor.tokenizer(list(texts), padding='max_length', truncation=True,
                                     max_length=max_length, return_tensors='pt').to(device)
        return feature_tensor(model.get_text_features(**tokens))

    return LoadedModel(model=EncoderWrapper(encode_image, encode_text, backend_label='transformers_clip'),
                       preprocess=preprocess, device=device, precision='fp32',
                       embedding_dim=dim, supports_text=True)
