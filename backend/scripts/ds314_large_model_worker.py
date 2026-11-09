"""JSON-lines inference worker. Diagnostics go to stderr, vectors to stdout."""
import contextlib
import json
from pathlib import Path
import sys
import traceback
import faulthandler

faulthandler.enable()
print(f'Worker starting: Python {sys.version.split()[0]}', file=sys.stderr, flush=True)
import torch
from PIL import Image, ImageOps


def create_encoder(options):
    checkpoint = options['checkpoint']
    device = 'cuda' if options['device'] == 'auto' and torch.cuda.is_available() else options['device']
    if device == 'auto':
        device = 'cpu'
    dtype = torch.float32
    if device.startswith('cuda') and options['precision'] != 'fp32':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    cache = str(Path(__file__).resolve().parents[1] / '.cache/huggingface')
    if options['kind'] == 'jina_clip':
        from src.embedding_extraction.models.backends.jina_clip import load
        loaded = load(checkpoint, precision=options['precision'], device_name=device,
            cache_dir=cache, trust_remote_code=True, text_task=options.get('text_task', 'retrieval.query'))
        def encode(mode, values):
            if mode == 'text':
                return loaded.model.encode_text(values)
            return torch.from_numpy(loaded.model.encode_image_paths([Path(p) for p in values]))
        return encode
    from huggingface_hub import snapshot_download
    checkpoint = snapshot_download(checkpoint, cache_dir=cache, local_files_only=True)
    from transformers import AutoModel, AutoProcessor, AutoImageProcessor
    if options['kind'] == 'siglip':
        processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=cache,
            use_fast=not options.get('slow_tokenizer', False))
        processor.image_processor = AutoImageProcessor.from_pretrained(checkpoint, cache_dir=cache, use_fast=False)
        model = AutoModel.from_pretrained(checkpoint, cache_dir=cache, torch_dtype=dtype,
            low_cpu_mem_usage=True, device_map={'': device}, attn_implementation='sdpa').eval()

        def encode(mode, values):
            if mode == 'text':
                tokens = processor.tokenizer(values, padding='max_length', truncation=True,
                    max_length=model.config.text_config.max_position_embeddings, return_tensors='pt').to(device)
                result = model.get_text_features(**tokens)
            else:
                images = []
                for path in values:
                    with Image.open(path) as image:
                        images.append(ImageOps.exif_transpose(image).convert('RGB'))
                pixels = processor(images=images, return_tensors='pt')['pixel_values'].to(device=device, dtype=dtype)
                result = model.get_image_features(pixel_values=pixels)
            return result if isinstance(result, torch.Tensor) else result.pooler_output
        return encode

    import importlib.util
    from transformers import BitsAndBytesConfig
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from qwen_vl_utils.vision_process import process_vision_info
    source = Path(__file__).resolve().parents[1] / '.cache/qwen3_vl_embedding.py'
    spec = importlib.util.spec_from_file_location('ds314_official_qwen_embedding', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not device.startswith('cuda'):
        raise RuntimeError('Qwen NF4 requires CUDA in this runtime.')
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=options.get('double_quant', True))
    helper = module.Qwen3VLEmbedder.__new__(module.Qwen3VLEmbedder)
    helper.max_length = options.get('max_length', 1024)
    helper.min_pixels = 4 * 32 * 32
    helper.max_pixels = options.get('max_pixels', 147456)
    helper.total_pixels = 7864320
    helper.fps = 1.0
    helper.max_frames = 64
    helper.default_instruction = options.get('instruction', "Represent the user's input.")
    helper.model = module.Qwen3VLForEmbedding.from_pretrained(checkpoint, cache_dir=cache,
        quantization_config=quantization, torch_dtype=dtype, device_map={'': device},
        attn_implementation='sdpa', low_cpu_mem_usage=True, trust_remote_code=True).eval()
    helper.processor = Qwen3VLProcessor.from_pretrained(checkpoint, cache_dir=cache, padding_side='right')

    def encode(mode, values):
        if mode == 'text':
            inputs = [{'text': value} for value in values]
        else:
            inputs = []
            for path in values:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert('RGB')
                conversation = helper.format_model_input(image=image)
                checked, _, _ = process_vision_info([conversation], image_patch_size=16,
                    return_video_metadata=True, return_video_kwargs=True)
                if not checked:
                    raise ValueError('Qwen vision processing produced no images')
                inputs.append({'image': image})
        return helper.process(inputs, normalize=True)
    return encode


def main():
    options = json.loads(sys.argv[1])
    with contextlib.redirect_stdout(sys.stderr):
        encoder = create_encoder(options)
    print(f"Encoder ready: {options['checkpoint']}", file=sys.stderr, flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                vectors = encoder(request['mode'], request['values']).float()
                vectors = torch.nn.functional.normalize(vectors, dim=-1).cpu().tolist()
            print(json.dumps({'vectors': vectors}), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}), flush=True)


if __name__ == '__main__':
    main()
