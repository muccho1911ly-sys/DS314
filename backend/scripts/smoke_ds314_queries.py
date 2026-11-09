"""Exercise real DS314 text/image encoders and both HTTP search contracts."""
import os
from pathlib import Path
import json
import sys
import argparse

BACKEND = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(BACKEND / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
sys.path.insert(0, str(BACKEND))

import numpy as np
from fastapi.testclient import TestClient
from fastapi.staticfiles import StaticFiles
from src.api.main import app, load_config, _dataset_runtime_config
from src.api.legacy.paths import LegacyPaths
from src.retrieval.system import build_system


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['clip_b32', 'siglip2_384', 'jina_clip_v2',
                        'siglip2_so400m', 'siglip2_giant', 'qwen3_vl_2b'],
                        default=['clip_b32', 'siglip2_384', 'jina_clip_v2'])
    expected = set(parser.parse_args().models)
    base = load_config('configs/app.yaml')
    config = _dataset_runtime_config(base, 'ds314', base['datasets']['available']['ds314'])
    config['semantic']['models'] = [m for m in config['semantic']['models'] if m['model_key'] in expected]
    config['semantic']['default_model_key'] = config['semantic']['models'][0]['model_key']
    system = build_system(config)
    assert set(system.available_models()) == expected, system.available_models()
    paths = LegacyPaths(BACKEND, BACKEND / 'configs/app.yaml', config)
    app.state.cfg = base
    app.state.retrieval_system = system
    app.state.retrieval_systems = {'ds314': system}
    app.state.dataset_configs = {'ds314': config}
    app.state.dataset_paths = {'ds314': paths}
    app.state.dataset_errors = {'ds314': None}
    app.state.legacy_paths = paths
    app.state.should_profile = False
    app.mount('/static/ds314/keyframes', StaticFiles(directory=str(paths.keyframes_root)), name='smoke_ds314')
    client = TestClient(app)  # No context: use the explicitly configured DS314 system.
    assert set(client.get('/api/search/models?dataset=ds314').json()['models']) == expected
    public = client.get('/api/config?dataset=ds314')
    assert public.status_code == 200, public.text
    assert public.json()['model']['key'] == config['semantic']['default_model_key']
    assert public.json()['semantic_model_roles'] == {
        'local': 'clip_b32' if 'clip_b32' in expected else None,
        'global': 'jina_clip_v2' if 'jina_clip_v2' in expected else None}
    reports = []
    for key in sorted(expected):
        payload = dict(dataset='ds314', model_key=key, query='a person riding a motorbike',
                       top_k=3, translate=False, use_split=False)
        response = client.post('/api/search/semantic', json=payload)
        assert response.status_code == 200, response.text
        assert len(response.json()['results']) == 3, response.text
        legacy = client.post('/api/search', json=dict(dataset='ds314', model_key=key,
                            query=payload['query'], top_k=3, use_translate=False, search_mode='semantic'))
        assert legacy.status_code == 200, legacy.text
        results = legacy.json()['results']
        assert len(results) == 3 and all(r['collection'] == 'ds314' for r in results)
        image = client.get(results[0]['image_url'])
        assert image.status_code == 200 and image.headers['content-type'].startswith('image/')
        index = system.orchestrator.semantic_search._resolve(key)
        row = index.metadata_records[0]
        vector = index.encode_image(row['keyframe_path'])
        assert vector.shape == (1, index.index.d) and np.isfinite(vector).all()
        cached = index.index.reconstruct(0)
        cosine = float(vector[0] @ cached / (np.linalg.norm(vector[0]) * np.linalg.norm(cached)))
        assert cosine > 0.98, (key, cosine)
        reports.append(dict(model=key, text_query_status=200, image_status=image.status_code,
                            image_embedding_cosine=cosine,
                            top_results=[{'video_id': r['video_id'], 'frame_id': r['frame_id']} for r in results]))
        print(json.dumps(reports[-1]), flush=True)
    out = BACKEND.parent / 'data/DS314/database/query_smoke_report.json'
    previous = json.loads(out.read_text()) if out.exists() else []
    combined = {r['model']: r for r in previous}
    combined.update({r['model']: r for r in reports})
    out.write_text(json.dumps(list(combined.values()), indent=2), encoding='utf-8')
    print(f'All DS314 query checks passed: {out}', flush=True)


if __name__ == '__main__':
    main()
