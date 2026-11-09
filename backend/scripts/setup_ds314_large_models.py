"""Install an isolated worker runtime and pre-cache DS314 large checkpoints."""
import os
from pathlib import Path
import subprocess
import sys
import time

BACKEND = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(BACKEND / '.cache/huggingface'))
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'


def main():
    target = BACKEND / '.cache/large_model_runtime'
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--target', str(target),
        '--upgrade', '--no-deps', 'transformers==4.57.3', 'bitsandbytes==0.48.2', 'qwen-vl-utils==0.0.14',
        'accelerate==1.12.0', 'psutil==7.1.3'])
    import requests
    import hashlib
    source_path = BACKEND / '.cache/qwen3_vl_embedding.py'
    if not source_path.exists():
        commit = requests.get('https://api.github.com/repos/QwenLM/Qwen3-VL-Embedding/commits',
            params={'path': 'src/models/qwen3_vl_embedding.py', 'per_page': 1}, timeout=30)
        commit.raise_for_status()
        sha = commit.json()[0]['sha']
        url = f'https://raw.githubusercontent.com/QwenLM/Qwen3-VL-Embedding/{sha}/src/models/qwen3_vl_embedding.py'
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        source_path.write_bytes(response.content)
        (source_path.with_suffix('.source.txt')).write_text(
            f'Source: {url}\nLicense: Apache-2.0\nSHA256: {hashlib.sha256(response.content).hexdigest()}\n', encoding='utf-8')
    if '--runtime-only' in sys.argv:
        return
    from huggingface_hub import snapshot_download
    for checkpoint in ['google/siglip2-so400m-patch14-384',
                       'google/siglip2-giant-opt-patch16-384', 'Qwen/Qwen3-VL-Embedding-2B']:
        for attempt in range(3):
            try:
                print(f'Caching {checkpoint}, attempt {attempt+1}', flush=True)
                path = snapshot_download(checkpoint, cache_dir=str(BACKEND / '.cache/huggingface'),
                    allow_patterns=['*.json', '*.txt', '*.model', '*.safetensors', '*.jinja'], max_workers=2)
                print(f'Cached {checkpoint}: {path}', flush=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)


if __name__ == '__main__':
    main()
