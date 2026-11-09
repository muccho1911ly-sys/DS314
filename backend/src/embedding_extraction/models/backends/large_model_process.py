"""Lazy isolated encoders: one model process owns GPU memory at a time.

The main API keeps Transformers 4.56 for Jina. A worker imports a separately
installed Transformers 4.57 runtime, needed by Qwen3-VL. Switching large
models stops the previous worker and releases its CUDA allocations.
"""
import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock

import numpy as np
import torch
from src.embedding_extraction.models.backends.base import LoadedModel

BACKEND = Path(__file__).resolve().parents[4]
_lock = RLock()
_process = None
_signature = None
_log = None


def close_worker():
    global _process, _signature, _log
    with _lock:
        if _process is not None:
            _process.terminate()
            try:
                _process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                _process.kill(); _process.wait()
            _process = None
        if _log is not None:
            _log.close(); _log = None
        _signature = None


atexit.register(close_worker)


def _request(options, mode, values):
    global _process, _signature, _log
    signature = json.dumps(options, sort_keys=True)
    with _lock:
        if _signature != signature or _process is None or _process.poll() is not None:
            close_worker()
            runtime = BACKEND / '.cache/large_model_runtime'
            is_jina = options['kind'] == 'jina_clip'
            if not is_jina and not (runtime / 'transformers').is_dir():
                raise RuntimeError('Missing isolated runtime. Run scripts/setup_ds314_large_models.py first.')
            env = os.environ.copy()
            env['PYTHONPATH'] = str(BACKEND) if is_jina else str(runtime) + os.pathsep + str(BACKEND)
            env['HF_HOME'] = str(BACKEND / '.cache/huggingface')
            env['HF_HUB_DISABLE_XET'] = '1'
            env['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
            env['PYTHONFAULTHANDLER'] = '1'
            (BACKEND / 'logs').mkdir(exist_ok=True)
            _log = (BACKEND / 'logs/large_model_worker.log').open('a', encoding='utf-8')
            _process = subprocess.Popen(
                [sys.executable, '-u', str(BACKEND / 'scripts/ds314_large_model_worker.py'), signature],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=_log,
                text=True, encoding='utf-8', env=env, cwd=str(BACKEND),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            _signature = signature
        _process.stdin.write(json.dumps({'mode': mode, 'values': values}) + '\n')
        _process.stdin.flush()
        line = _process.stdout.readline()
        if not line:
            try:
                code = _process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                code = None
            close_worker()
            raise RuntimeError(f'Large model worker exited (code={code}). See backend/logs/large_model_worker.log')
        response = json.loads(line)
        if 'error' in response:
            raise RuntimeError(response['error'])
        vectors = np.asarray(response['vectors'], dtype=np.float32)
        if vectors.shape != (len(values), options['dimension']) or not np.isfinite(vectors).all():
            raise RuntimeError(f'Invalid worker embedding shape: {vectors.shape}')
        return vectors


def load(model_name, pretrained=None, precision='bf16', device_name='auto', logger=None,
         dimension=None, worker_kind='siglip', **extra):
    if dimension is None:
        raise ValueError('Large model configuration must declare dimension')
    options = dict(checkpoint=pretrained or model_name, dimension=int(dimension),
                   kind=worker_kind, precision=precision, device=device_name, **extra)

    class Encoder:
        def encode_text(self, texts):
            return torch.from_numpy(_request(options, 'text', list(texts)))

        def encode_image_paths(self, paths):
            return _request(options, 'image', [str(Path(p).resolve()) for p in paths])

    return LoadedModel(model=Encoder(), preprocess=None, device=torch.device('cpu'),
                       precision=precision, embedding_dim=int(dimension), supports_text=True)
