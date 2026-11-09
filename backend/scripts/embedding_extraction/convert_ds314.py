"""Convert the DS314 matrix export to the project's per-keyframe schema.

Run from any directory: python backend/scripts/embedding_extraction/convert_ds314.py
Original inputs are archived under DS314/experiments/embedding_import_original.
Video IDs are derived from filename prefixes, not verified against source videos.
No frame positions, FPS, or timestamps are inferred from filenames.
"""
from pathlib import Path
import csv
import json
import re
import shutil
import sys
from functools import lru_cache

import numpy as np


ROOT = Path(__file__).resolve().parents[3] / 'data' / 'DS314'
MODELS = {'clip_b32': 512, 'siglip2_384': 768, 'jina_clip_v2': 1024}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = ROOT.resolve()
    archive = root / 'experiments' / 'embedding_import_original'
    stage = root / '.schema_stage'
    if '--resume' in sys.argv or '--verify' in sys.argv:
        # Continue a fully staged import after an interrupted directory rename.
        assert archive.is_dir()
        if '--resume' in sys.argv:
            assert stage.is_dir()
        for name in ('keyframes', 'embeddings', 'map-keyframes'):
            if (stage / name).is_dir():
                if (root / name).exists():
                    assert not (archive / name).exists()
                    (root / name).rename(archive / name)
                assert (archive / name).is_dir()
                (stage / name).rename(root / name)
        if stage.exists():
            stage.rmdir()
        with (root / 'embeddings' / 'metadata.csv').open(encoding='utf-8', newline='') as stream:
            manifest = list(csv.DictReader(stream))
        matrices = {m: np.load(archive / 'embeddings' / m / 'embeddings.npy', allow_pickle=False) for m in MODELS}
        verify(root, manifest, matrices)
        return
    if archive.exists() or stage.exists():
        raise RuntimeError('Archive/staging already exists; refusing to overwrite an import.')
    with (root / 'embeddings' / 'metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5000
    assert [int(row['row_id']) for row in rows] == list(range(len(rows)))
    images = {}
    for image in (root / 'keyframes').rglob('*'):
        if image.is_file() and image.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
            if image.name in images:
                raise ValueError(f'Ambiguous image: {image.name}')
            images[image.name] = image
    manifest = []
    destinations = set()
    for row in rows:
        match = re.fullmatch(r'(\d+)_(\d+)\.jpg', row['filename'])
        if not match or not re.fullmatch(r'batch\d+', row['batch']):
            raise ValueError(f'Unexpected naming: {row}')
        video, key = match.groups()
        dataset = row['batch'].capitalize()
        video_id = f'DS314_V{video}'
        key_id = f'{int(key):06d}'
        rel = Path(dataset) / video_id / key_id
        if rel in destinations:
            raise ValueError(f'Duplicate destination: {rel}')
        destinations.add(rel)
        image = images[row['filename']]
        assert image.stat().st_size == int(row['uncompressed_bytes'])
        manifest.append(dict(row_id=row['row_id'], collection='ds314', dataset=dataset,
                             video_id=video_id, keyframe_id=key_id,
                             image_path=f'keyframes/{rel.as_posix()}.jpg',
                             original_filename=row['filename'], original_image_path=row['image_path'],
                             original_local_path=image.relative_to(root).as_posix()))
    matrices = {}
    for model, dim in MODELS.items():
        folder = root / 'embeddings' / model
        matrix = np.load(folder / 'embeddings.npy', allow_pickle=False)
        mask = np.load(folder / 'valid_mask.npy', allow_pickle=False)
        assert matrix.shape == (len(rows), dim) and matrix.dtype == np.float32
        assert mask.shape == (len(rows),) and mask.dtype == bool and mask.all()
        assert np.isfinite(matrix).all()
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5)
        for batch in sorted({row['batch'] for row in rows}):
            ids = [i for i, row in enumerate(rows) if row['batch'] == batch]
            assert np.array_equal(np.load(folder / f'{batch}.npy', allow_pickle=False), matrix[ids])
            assert np.array_equal(np.load(folder / f'{batch}_valid.npy', allow_pickle=False), mask[ids])
        matrices[model] = matrix
    for record in manifest:
        relative = Path(record['dataset']) / record['video_id'] / record['keyframe_id']
        image_out = stage / record['image_path']
        image_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / record['original_local_path'], image_out)
        for model, matrix in matrices.items():
            out = stage / 'embeddings' / f'embeddings_{model}' / relative.with_suffix('.npy')
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, matrix[int(record['row_id'])], allow_pickle=False)
    groups = {}
    for record in manifest:
        groups.setdefault((record['dataset'], record['video_id']), []).append(dict(
            video_id=record['video_id'], keyframe_id=int(record['keyframe_id']),
            frame_idx='', timestamp_sec='', fps=''))
    for (dataset, video), records in groups.items():
        write_csv(stage / 'map-keyframes' / dataset / f'{video}.csv', records)
    write_csv(stage / 'embeddings' / 'metadata.csv', manifest)
    shutil.copy2(root / 'embeddings' / 'run_info.json', stage / 'embeddings' / 'run_info.json')
    shutil.copy2(root / 'embeddings' / 'summary.csv', stage / 'embeddings' / 'summary.csv')
    # All inputs and output paths are confined to the explicit DS314 root.
    for name in ('keyframes', 'embeddings', 'map-keyframes'):
        source = (root / name).resolve()
        assert source.parent == root
        assert (stage / name).is_dir()
    archive.mkdir(parents=True)
    for name in ('keyframes', 'embeddings', 'map-keyframes'):
        (root / name).rename(archive / name)
        (stage / name).rename(root / name)
    stage.rmdir()
    verify(root, manifest, matrices)


def verify(root, manifest, matrices):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.retrieval.indexer.faiss.index_builder import build_matrix_and_metadata
    import pandas as pd
    original_read_csv = pd.read_csv
    # The builder normalizes a copy of every map dataframe. Reuse disk reads
    # during this offline validation; all parsing and matching stays unchanged.
    pd.read_csv = lru_cache(maxsize=64)(original_read_csv)
    reports = []
    lookup = {(r['dataset'], r['video_id'], r['keyframe_id']): int(r['row_id']) for r in manifest}
    for model, dim in MODELS.items():
        print(f'Validating {model}...', flush=True)
        folder = root / 'embeddings' / f'embeddings_{model}'
        matrix, metadata = build_matrix_and_metadata(folder, root / 'keyframes', root / 'map-keyframes', collection='ds314')
        assert matrix.shape == (len(manifest), dim)
        for i, record in metadata.iterrows():
            row_id = lookup[(record['dataset'], record['video_id'], record['keyframe_id'])]
            assert np.array_equal(matrix[i], matrices[model][row_id])
            assert Path(record['keyframe_path']).is_file()
            assert Path(record['map_path']).is_file()
        assert not metadata.duplicated(['dataset', 'video_id', 'keyframe_id']).any()
        metadata.to_csv(folder / 'metadata.csv', index=False)
        reports.append(dict(model=model, vectors=len(metadata), dimension=dim, verified=True))
        print(f'Validated {model}: {len(metadata)} vectors', flush=True)
    pd.read_csv = original_read_csv
    report = dict(models=reports, video_groups=len({(r['dataset'], r['video_id']) for r in manifest}),
                  video_id_source='Filename prefix; not verified against original videos',
                  timing='Unknown: frame_idx, timestamp_sec and fps intentionally blank',
                  originals='experiments/embedding_import_original')
    (root / 'embeddings' / 'conversion_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
