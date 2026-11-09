"""Run temporal retrieval candidates through a vLLM multimodal /rerank endpoint."""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml


def image_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def frame_path(frame: dict[str, Any], keyframes_root: Path) -> Path | None:
    rel = frame.get("image_rel_path")
    if rel:
        candidate = keyframes_root / Path(str(rel).replace("/", "\\"))
        if candidate.is_file():
            return candidate
    direct = frame.get("keyframe_path")
    if direct and Path(str(direct)).is_file():
        return Path(str(direct))
    return None


def sequence_document(result: dict[str, Any], sequence: list[dict[str, Any]], paths: list[Path]) -> dict[str, Any]:
    video_id = str(result.get("video_id", "unknown"))
    description = [f"Candidate video {video_id}. Ordered temporal sequence:"]
    content: list[dict[str, Any]] = []
    for index, (frame, path) in enumerate(zip(sequence, paths), start=1):
        timestamp = frame.get("timestamp_sec", frame.get("timestamp", 0))
        description.append(f"Frame {index} at {float(timestamp or 0):.3f} seconds.")
        content.append({"type": "text", "text": description[-1]})
        content.append({"type": "image_url", "image_url": {"url": image_data_uri(path)}})
    # Explicit video id context is useful for auditing, but relevance should be
    # judged from the ordered images and query.
    return {"content": [{"type": "text", "text": description[0]}, *content]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("queries.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repo = args.repo_root.resolve()
    dataset = str(config.get("dataset", "aic")).lower()
    keyframes_root = repo / "data" / dataset.upper() / "keyframes"
    out_dir = repo / "data" / dataset.upper() / "experiments" / "reranker_pilot" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = 0

    for spec in config.get("queries", []):
        query_id = str(spec.get("id", "query"))
        query = str(spec.get("query", "")).strip()
        if not query:
            print(f"SKIP {query_id}: fill query in {args.config}")
            continue
        expected_events = len(re.split(r"\s+THEN\s+", query))
        if expected_events not in (2, 3, 4):
            print(f"SKIP {query_id}: expected 2, 3, or 4 THEN-separated events; got {expected_events}")
            errors += 1
            continue
        search_payload = {
            "query": query,
            "dataset": dataset,
            "video_ids": [],
            "top_k": int(config.get("top_k", 20)),
            "candidate_multiplier": int(config.get("candidate_multiplier", 4)),
            "use_split": bool(config.get("use_split", False)),
            "reasoning": False,
            "use_translate": False,
            "search_mode": "temporal",
            "model_key": config.get("model_key", "siglip2-gopt16"),
            "duration_limit": float(config.get("duration_limit", -1)),
        }
        try:
            response = requests.post(
                config["backend_url"].rstrip("/") + "/api/search",
                json=search_payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            search = response.json()
            results = search.get("results", [])
            candidates = []
            missing_frames = []
            for rank, result in enumerate(results, start=1):
                sequence = result.get("matched_sequence") or []
                if len(sequence) != expected_events:
                    continue
                resolved = [frame_path(frame, keyframes_root) for frame in sequence]
                if any(path is None for path in resolved):
                    missing_frames.append({"video_id": result.get("video_id"), "rank": rank})
                    continue
                paths = [p for p in resolved if p is not None]
                candidates.append({
                    "baseline_rank": rank,
                    "video_id": result.get("video_id"),
                    "sequence": sequence,
                    "result": result,
                    "document": sequence_document(result, sequence, paths),
                })
            if not candidates:
                raise RuntimeError("No rerankable candidates with complete local keyframe images; inspect backend response and missing frames.")

            rerank_payload = {
                "model": config.get("reranker_model", "Qwen/Qwen3-VL-Reranker-2B"),
                "query": query,
                "documents": [candidate["document"] for candidate in candidates],
                "top_n": len(candidates),
            }
            rerank_response = requests.post(
                config["reranker_url"].rstrip("/") + "/rerank",
                json=rerank_payload,
                timeout=args.timeout,
            )
            rerank_response.raise_for_status()
            rerank_data = rerank_response.json()
            items = rerank_data.get("results", rerank_data.get("data", []))
            score_by_index = {
                int(item["index"]): float(item.get("relevance_score", item.get("score", 0.0)))
                for item in items
            }
            ordered_indices = sorted(score_by_index, key=score_by_index.get, reverse=True)
            reranked = []
            for new_rank, index in enumerate(ordered_indices, start=1):
                if 0 <= index < len(candidates):
                    candidate = candidates[index]
                    reranked.append({
                        "reranked_rank": new_rank,
                        "reranker_score": score_by_index[index],
                        "baseline_rank": candidate["baseline_rank"],
                        "video_id": candidate["video_id"],
                        "matched_sequence": candidate["sequence"],
                        "result": candidate["result"],
                    })
            report = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "query_id": query_id,
                "query": query,
                "expected_event_count": expected_events,
                "search_payload": search_payload,
                "query_plan_events": search.get("events"),
                "baseline_candidates": [
                    {"baseline_rank": c["baseline_rank"], "video_id": c["video_id"], "matched_sequence": c["sequence"]}
                    for c in candidates
                ],
                "reranked": reranked,
                "skipped_missing_keyframes": missing_frames,
                "reranker_model": rerank_payload["model"],
                "reranker_response": rerank_data,
                "ground_truth": {k: spec.get(k) for k in ("ground_truth_video_id", "ground_truth_start_sec", "ground_truth_end_sec")},
            }
            target = out_dir / f"{query_id}.json"
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"OK {query_id}: {len(candidates)} candidates; report={target}")
            for entry in reranked[:10]:
                print(f"  #{entry['reranked_rank']} {entry['video_id']} score={entry['reranker_score']:.4f} (baseline #{entry['baseline_rank']})")
        except Exception as exc:
            errors += 1
            print(f"ERROR {query_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                print(exc.response.text[:2000], file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
