"""Jina CLIP v2's official text and path-based image encoders."""
from pathlib import Path

import numpy as np
import torch

from src.embedding_extraction.models.backends.base import LoadedModel
from src.utils.device import resolve_device


def load(model_name, pretrained=None, precision='fp32', device_name='auto', logger=None,
         cache_dir=None, revision=None, trust_remote_code=False, text_task='retrieval.query', **_):
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

    if not trust_remote_code:
        raise ValueError('Jina CLIP requires explicit trust_remote_code: true.')
    device = resolve_device(device_name)
    dtype = torch.float32
    if device.type == 'cuda':
        if precision == 'bf16' and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif precision in {'fp16', 'amp'}:
            dtype = torch.float16
    cache_dir = cache_dir or str(Path(__file__).resolve().parents[4] / '.cache/huggingface')
    model = AutoModel.from_pretrained(pretrained or model_name, trust_remote_code=True,
                                      revision=revision, cache_dir=cache_dir,
                                      torch_dtype=str(dtype).removeprefix('torch.'),
                                      use_text_flash_attn=False, use_vision_xformers=False).to(device=device, dtype=dtype).eval()
    # Jina otherwise downloads its tokenizer lazily from the default cache on
    # the first query, ignoring the model's explicit cache_dir.
    model.tokenizer = AutoTokenizer.from_pretrained(
        pretrained or model_name, cache_dir=cache_dir, revision=revision, trust_remote_code=True,
    )
    model.preprocess = AutoImageProcessor.from_pretrained(
        pretrained or model_name, cache_dir=cache_dir, revision=revision, trust_remote_code=True,
    )

    class JinaEncoder:
        @torch.inference_mode()
        def encode_text(self, texts):
            vectors = model.encode_text(list(texts), task=text_task, truncate_dim=None)
            return torch.as_tensor(vectors, device=device, dtype=torch.float32)

        @torch.inference_mode()
        def encode_image_paths(self, paths):
            vectors = model.encode_image([str(p) for p in paths], truncate_dim=None)
            if isinstance(vectors, torch.Tensor):
                vectors = vectors.float().cpu().numpy()
            return np.asarray(vectors, dtype=np.float32)

    return LoadedModel(model=JinaEncoder(), preprocess=None, device=device,
                       precision=str(dtype).removeprefix('torch.'), embedding_dim=1024, supports_text=True)
