# Temporal video reranker pilot

This experiment keeps the production search path untouched. It requests temporal candidate sequences from the current backend, then submits each ordered frame sequence plus the same query to vLLM's multimodal `/rerank` endpoint.

## 1. Inspect and annotate the sample video

The sample is `L21_V001` (AIC). Generate or refresh the 24-frame contact sheet and annotation CSV:

```powershell
cd D:\new_ERS\event-retrieval-system
backend\venv\Scripts\python.exe backend\scripts\reranker_pilot\prepare_video_assets.py --video-id L21_V001 --samples 24
```

Review `data\AIC\experiments\reranker_pilot\L21_V001\contact_sheet.jpg`. Fill `frame_annotations.csv` with descriptions and event order. Then edit `queries.yaml` with 2-, 3-, and 4-event natural-language queries that match visible events, joined by uppercase `THEN` in temporal order. Keep `use_split: true` so the current backend parser makes these ordered temporal events. Fill the ground-truth interval for each query. The intended video is included in the ground truth for evaluation; retrieval searches the full AIC set.

## 2. Start the backend

From a PowerShell window:

```powershell
cd D:\new_ERS\event-retrieval-system\backend
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Wait until startup completes and `/docs` loads. The first launch may download/load the configured SigLIP model and index; that is separate from reranker readiness.

## 3. Start vLLM (Linux/WSL2 with compatible NVIDIA GPU)

Run inside the Linux/WSL environment with vLLM installed. The official Qwen3-VL reranker serving setup uses the pooling runner and the model's classification override/template:

```bash
vllm serve Qwen/Qwen3-VL-Reranker-2B \
  --runner pooling \
  --max-model-len 4096 \
  --hf_overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template examples/pooling/score/template/qwen3_vl_reranker.jinja \
  --host 0.0.0.0 --port 8001
```

Keep the supplied chat template from the same vLLM version as the server. If the Windows host cannot reach WSL through `127.0.0.1:8001`, use the WSL IP as `reranker_url` in `queries.yaml`.

## 4. Run and inspect

After filling the query and ground-truth fields:

```powershell
cd D:\new_ERS\event-retrieval-system
backend\venv\Scripts\python.exe backend\scripts\reranker_pilot\run_pilot.py
```

The JSON report in `data\AIC\experiments\reranker_pilot\runs` records the temporal baseline rank, reranked rank and score, ordered frame IDs/timestamps, skipped missing images, and backend query plan. Compare whether the known target video and time interval move upward and whether the frame order matches the annotated event order. This is a small qualitative pilot; meaningful claims need multiple queries with labeled relevant intervals and a baseline-vs-reranker metric such as Recall@K/MRR/nDCG.

## Current limitations to report

- The backend's production reranker is not wired; this pilot calls vLLM separately.
- The pilot requires actual event queries and ground-truth intervals before it can produce an evaluation.
- Any result whose `matched_sequence` frame image is absent locally is excluded and listed in the report.
- vLLM is intended to run in Linux/WSL2 with a compatible GPU; the current Windows backend venv is not the place to serve it.

References: [vLLM scoring documentation](https://docs.vllm.ai/en/latest/models/pooling_models/scoring/), [vLLM multimodal rerank example](https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/vision_rerank_api_online.py), [Qwen3-VL-Reranker model card](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B).
