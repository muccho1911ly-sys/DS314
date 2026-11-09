import unittest

from src.api.main import _dataset_runtime_config, load_config
from src.api.schemas.legacy import SearchRequest
from src.api.schemas.search import SemanticSearchRequest


class DatasetConfigTests(unittest.TestCase):
    def test_dataset_selects_only_its_own_model_indexes(self):
        for dataset, expected in [('ds314', {'clip_b32', 'siglip2_384', 'jina_clip_v2', 'siglip2_so400m', 'siglip2_giant', 'qwen3_vl_2b'}),
                                  ('aic', {'siglip2-gopt16'}), ('cam', {'siglip2-gopt16'})]:
            with self.subTest(dataset=dataset):
                base = load_config('configs/app.yaml' if dataset == 'ds314' else 'configs/app_all_datasets.yaml')
                cfg = _dataset_runtime_config(base, dataset, base['datasets']['available'][dataset])
                self.assertEqual({m['model_key'] for m in cfg['semantic']['models']}, expected)
                self.assertIn(cfg['semantic']['default_model_key'], expected)
                for model in cfg['semantic']['models']:
                    self.assertIn(f'/data/{dataset.upper()}/database/', model['index_path'])
                    self.assertTrue(model['vector_cache_path'].endswith('/vectors_fp32.npy'))
                if dataset == 'ds314':
                    self.assertIsNone(cfg['ocr'])
                    self.assertIsNone(cfg['asr'])
                    self.assertFalse(cfg['translate']['enabled_default'])
                    self.assertEqual(cfg['fusion']['local_semantic_model'], 'clip_b32')
                    self.assertEqual(cfg['fusion']['global_semantic_model'], 'jina_clip_v2')


    def test_ds314_accepted_by_both_search_contracts(self):
        for schema in (SearchRequest, SemanticSearchRequest):
            with self.subTest(schema=schema.__name__):
                request = schema(dataset='ds314', query='a person riding a motorbike', model_key='clip_b32')
                self.assertEqual(request.dataset, 'ds314')

    def test_active_runtime_loads_ds314_only(self):
        cfg = load_config('configs/app.yaml')
        self.assertEqual(set(cfg['datasets']['available']), {'ds314'})
        self.assertEqual(cfg['datasets']['default'], 'ds314')
        self.assertEqual(cfg['semantic']['default_model_key'], 'clip_b32')
        self.assertEqual({m['model_key'] for m in cfg['semantic']['models']},
                         {'clip_b32', 'siglip2_384', 'jina_clip_v2', 'siglip2_so400m', 'siglip2_giant', 'qwen3_vl_2b'})
