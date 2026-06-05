#!/usr/bin/env python3
"""验收 OVD 全量产物（最佳完成标准）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def count_unique_images(predictions_path: Path) -> int:
    data = json.loads(predictions_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return len({r.get("image_id") for r in data if "image_id" in r})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baseline-map", type=float, default=0.424)
    parser.add_argument("--min-images", type=int, default=5000)
    args = parser.parse_args()

    n_img = count_unique_images(args.predictions)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    mAP = float(metrics.get("mAP", 0))

    ok_images = n_img >= args.min_images
    ok_map = mAP > args.baseline_map
    print(f"unique image_id: {n_img} (need >= {args.min_images}) -> {'OK' if ok_images else 'FAIL'}")
    print(f"mAP: {mAP:.4f} (need > {args.baseline_map}) -> {'OK' if ok_map else 'FAIL'}")

    return 0 if (ok_images and ok_map) else 1


if __name__ == "__main__":
    raise SystemExit(main())
