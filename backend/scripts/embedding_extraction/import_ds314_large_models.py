"""Validate/import the second DS314 matrix export, preserving existing identities."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3] / 'data/DS314'
MODELS = {'siglip2_so400m': 1152, 'siglip2_giant': 1536, 'qwen3_vl_2b': 2048}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    root = ROOT.resolve()
    source = root / 'embeddings/BATCH_3_large_models_embeddings'
    archive = root / 'experiments/large_models_embedding_import'
    if archive.exists():
        raise RuntimeError('Import archive already exists; refusing to overwrite.')
    rows = read_csv(source / 'metadata.csv')
    canonical = read_csv(root / 'embeddings/metadata.csv')
    identities = {r['original_filename']: r for r in canonical}
    base_records = read_csv(root / 'embeddings/embeddings_clip_b32/metadata.csv')
    base_lookup = {(r['dataset'], r['video_id'], r['keyframe_id']): r for r in base_records}
    assert len(rows) == len(identities) == 5000
    assert [int(r['row_id']) for r in rows] == list(range(5000))
    assert set(r['filename'] for r in rows) == set(identities)
    mapped = []
    for r in rows:
        identity = identities[r['filename']]
        assert identity['dataset'].lower() == r['batch'].lower()
        image = root / identity['image_path']
        assert image.is_file() and image.stat().st_size == int(r['file_size'])
        mapped.append(base_lookup[(identity['dataset'], identity['video_id'], identity['keyframe_id'])].copy())
    matrices = {}
    for model, dim in MODELS.items():
        folder = root / 'embeddings' / model
        progress = json.loads((folder / 'progress.json').read_text())
        matrix = np.load(folder / 'embeddings.npy', allow_pickle=False)
        mask = np.load(folder / 'valid_mask.npy', allow_pickle=False)
        assert matrix.shape == (5000, dim) and matrix.dtype == np.float32
        assert mask.shape == (5000,) and mask.dtype == bool and mask.all()
        assert np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5)
        for batch in sorted({r['batch'] for r in rows}):
            ids = [i for i,r in enumerate(rows) if r['batch'] == batch]
            assert np.array_equal(np.load(folder / f'{batch}.npy'), matrix[ids])
            assert np.array_equal(np.load(folder / f'{batch}_valid_mask.npy'), mask[ids])
        matrices[model] = (matrix, progress)
    # Map IDs must agree with the existing image schema; never invent timing.
    map_groups = {}
    for record in mapped:
        map_groups.setdefault(Path(record['map_path']), []).append(record)
    for map_path, records in map_groups.items():
        expected = {(r['video_id'], int(r['keyframe_id'])) for r in records}
        if map_path.exists():
            existing = read_csv(map_path)
            assert {(r['video_id'], int(r['keyframe_id'])) for r in existing} == expected
        else:
            map_path.parent.mkdir(parents=True, exist_ok=True)
            with map_path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['video_id','keyframe_id','frame_idx','timestamp_sec','fps'])
                writer.writeheader()
                writer.writerows(dict(video_id=r['video_id'],keyframe_id=int(r['keyframe_id']),frame_idx='',timestamp_sec='',fps='') for r in records)
    reports = []
    for model, (matrix, progress) in matrices.items():
        output = root / 'embeddings' / f'embeddings_{model}'
        if output.exists():
            raise RuntimeError(f'Output already exists: {output}')
        records = []
        for i, template in enumerate(mapped):
            record = template.copy()
            path = output / record['dataset'] / record['video_id'] / f"{record['keyframe_id']}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix[i], allow_pickle=False)
            assert np.array_equal(np.load(path, allow_pickle=False), matrix[i])
            record['embedding_path'] = str(path)
            records.append(record)
        with (output / 'metadata.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        (output / 'provenance.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
        reports.append(dict(model=model, dimension=matrix.shape[1], vectors=len(records), map_groups=len(map_groups), verified=True))
        print(json.dumps(reports[-1]), flush=True)
    archive.mkdir(parents=True)
    for name in [*MODELS, source.name]:
        original = (root / 'embeddings' / name).resolve()
        assert original.is_relative_to(root / 'embeddings')
        original.rename(archive / name)
    (root / 'embeddings/large_models_import_report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
