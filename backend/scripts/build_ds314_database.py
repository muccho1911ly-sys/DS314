"""Build DS314 indexes and float32 caches from converted per-frame embeddings.

Run from any directory: python backend/scripts/build_ds314_database.py
Uses the metadata exported and verified by convert_ds314.py, avoiding repeated
map CSV reads for every keyframe. Canonical vector files are read afresh.
"""
from pathlib import Path
import json
import sys
import argparse

import faiss
import numpy as np
import pandas as pd
import yaml

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from src.retrieval.indexer.faiss.index_builder import create_faiss_index, save_faiss_index
from src.retrieval.indexer.faiss.vector_cache import build_vector_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+')
    args = parser.parse_args()
    cfg = yaml.safe_load((BACKEND / 'configs/build_indexes_ds314.yaml').read_text(encoding='utf-8'))
    dimensions = {'clip_b32': 512, 'siglip2_384': 768, 'jina_clip_v2': 1024}
    faiss.omp_set_num_threads(8)
    reports = []
    for model in cfg['models']:
        name = model['name']
        if args.models and name not in args.models:
            continue
        dimension = model.get('dimension', dimensions.get(name))
        folder = (BACKEND / model['embeddings_root']).resolve()
        output = (BACKEND / model['output_dir']).resolve()
        root = (BACKEND / '../data/DS314').resolve()
        assert folder.is_relative_to(root) and output.is_relative_to(root / 'database')
        print(f'[{name}] Loading canonical vectors and metadata', flush=True)
        string_columns = {'keyframe_id': str, 'video_id': str, 'dataset': str, 'source_name': str}
        metadata = pd.read_csv(folder / 'metadata.csv', dtype=string_columns)
        assert len(metadata) == 5000
        assert metadata['collection'].eq('ds314').all()
        assert not metadata.duplicated(['dataset', 'video_id', 'keyframe_id']).any()
        paths = [Path(p).resolve() for p in metadata['embedding_path']]
        assert len(set(paths)) == len(paths)
        assert set(paths) == {p.resolve() for p in folder.rglob('*.npy')}
        vectors = []
        for path, record in zip(paths, metadata.to_dict('records')):
            relative = Path(record['dataset']) / record['video_id'] / record['keyframe_id']
            assert path == (folder / relative.with_suffix('.npy')).resolve()
            assert Path(record['keyframe_path']).resolve() == root / 'keyframes' / relative.with_suffix('.jpg')
            assert Path(record['keyframe_path']).is_file() and Path(record['map_path']).is_file()
            vector = np.load(path, allow_pickle=False)
            assert vector.shape == (dimension,) and vector.dtype == np.float32
            vectors.append(vector)
        matrix = np.stack(vectors)
        assert np.isfinite(matrix).all()
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5)
        print(f'[{name}] Building HNSW cosine index: {matrix.shape}', flush=True)
        settings = cfg['index_defaults']
        index = create_faiss_index(matrix, metric=settings['metric'], index_type=settings['type'],
                                   hnsw_m=settings['hnsw_m'], ef_construction=settings['ef_construction'],
                                   ef_search=settings['ef_search'])
        index_path, metadata_path = save_faiss_index(index, metadata, output, 'keyframes.faiss', 'metadata.csv')
        cache_path = output / 'vectors_fp32.npy'
        build_vector_cache(index_path, cache_path, np.dtype(np.float32), batch_size=5000, normalize=True)
        # Verify persisted files, row alignment and actual search behavior.
        loaded = faiss.read_index(str(index_path))
        cache = np.load(cache_path, mmap_mode='r', allow_pickle=False)
        saved_metadata = pd.read_csv(metadata_path, dtype=string_columns)
        assert loaded.ntotal == len(saved_metadata) == matrix.shape[0]
        assert loaded.d == matrix.shape[1] and cache.shape == matrix.shape
        assert cache.dtype == np.float32 and np.isfinite(cache).all()
        assert np.allclose(cache, matrix, atol=1e-6)
        assert np.allclose(loaded.reconstruct_n(0, loaded.ntotal), matrix, atol=1e-6)
        assert saved_metadata['embedding_path'].tolist() == metadata['embedding_path'].tolist()
        assert saved_metadata['source_name'].tolist() == metadata['source_name'].tolist()
        scores, ids = loaded.search(matrix, 1)
        assert (ids >= 0).all() and np.allclose(scores[:, 0], 1, atol=1e-5)
        report = dict(model=name, vectors=loaded.ntotal, dimension=loaded.d,
                      index_type=type(loaded).__name__, metric='cosine', cache_dtype=str(cache.dtype),
                      self_search_queries=len(matrix), self_search_passed=True,
                      files={p.name: p.stat().st_size for p in (index_path, metadata_path, cache_path)})
        reports.append(report)
        print(json.dumps(report), flush=True)
    report_path = root / 'database/build_report.json'
    previous = json.loads(report_path.read_text()) if report_path.exists() else []
    combined = {r['model']: r for r in previous}
    combined.update({r['model']: r for r in reports})
    report_path.write_text(json.dumps(list(combined.values()), indent=2), encoding='utf-8')
    print(f'Completed: {report_path}', flush=True)


if __name__ == '__main__':
    main()
