"""Create a contact sheet and annotation CSV for a local AIC video's keyframes."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--video-id", default="L21_V001")
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    video_id = args.video_id
    group = video_id.split("_")[0]
    frames_dir = repo / "data" / "AIC" / "keyframes" / group / video_id
    map_path = repo / "data" / "AIC" / "map-keyframes" / group / f"{video_id}.csv"
    output_dir = repo / "data" / "AIC" / "experiments" / "reranker_pilot" / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No keyframe JPGs found: {frames_dir}")
    timestamp_by_id: dict[int, float] = {}
    with map_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp_by_id[int(row["keyframe_id"])] = float(row["timestamp_sec"])

    count = min(max(1, args.samples), len(frames))
    indices = sorted({round(i * (len(frames) - 1) / max(1, count - 1)) for i in range(count)})
    selected = [frames[i] for i in indices]
    thumb_w, thumb_h, label_h = 240, 150, 42
    columns = 4
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    annotation_rows = []
    for pos, path in enumerate(selected):
        keyframe_id = int(path.stem)
        timestamp = timestamp_by_id.get(keyframe_id)
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_w, thumb_h))
            x = (pos % columns) * thumb_w + (thumb_w - im.width) // 2
            y = (pos // columns) * (thumb_h + label_h)
            sheet.paste(im, (x, y))
        label_x = (pos % columns) * thumb_w + 5
        label_y = (pos // columns) * (thumb_h + label_h) + thumb_h + 3
        timestamp_text = f"{timestamp:.2f}s" if timestamp is not None else "time unknown"
        draw.text((label_x, label_y), f"{path.name}  |  {timestamp_text}", fill="black", font=font)
        draw.text((label_x, label_y + 16), "Description: __________________", fill="black", font=font)
        annotation_rows.append({
            "video_id": video_id,
            "keyframe_id": path.stem,
            "timestamp_sec": "" if timestamp is None else f"{timestamp:.6f}",
            "description": "",
            "event_order_if_relevant": "",
            "ground_truth_relevant": "",
            "notes": "",
        })

    sheet_path = output_dir / "contact_sheet.jpg"
    sheet.save(sheet_path, quality=92)
    csv_path = output_dir / "frame_annotations.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotation_rows[0]))
        writer.writeheader()
        writer.writerows(annotation_rows)
    print(f"video={video_id} total_keyframes={len(frames)} sampled={len(selected)}")
    print(f"contact_sheet={sheet_path}")
    print(f"annotations={csv_path}")


if __name__ == "__main__":
    main()
