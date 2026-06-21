#!/usr/bin/env python3
"""Validate final supplementary evidence for the report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_ovd_artifacts import count_unique_images  # noqa: E402

DATASET_SPLITS = {
    "refcoco": ["validation", "testB"],
    "refcoco+": ["validation", "testB"],
    "refcocog": ["validation"],
}


def check_file(path: Path, label: str) -> bool:
    ok = path.is_file()
    print(f"[{'OK' if ok else 'FAIL'}] {label}: {path}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=Path("supplementary/metrics"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("supplementary/predictions"))
    parser.add_argument("--baseline-map", type=float, default=0.424)
    parser.add_argument("--min-ovd-images", type=int, default=4990)
    args = parser.parse_args()

    ok = True
    metrics_dir = args.metrics_dir
    predictions_dir = args.predictions_dir

    required_metrics = {
        "OVD self metrics": metrics_dir / "ovd_self_metrics.json",
        "OVD official comparison": metrics_dir / "ovd_official_comparison.json",
        "VG full metrics": metrics_dir / "vg_full_metrics.json",
        "Prompt ablation": metrics_dir / "prompt_ablation.json",
        "VG sweep best subset": metrics_dir / "vg_sweep_best_subset.json",
        "VG protocol comparison": metrics_dir / "vg_protocol_comparison.json",
        "CUDA build status": metrics_dir / "cuda_build_status.json",
    }
    for label, path in required_metrics.items():
        ok &= check_file(path, label)

    ovd_pred = predictions_dir / "ovd_self_predictions.json"
    ok &= check_file(ovd_pred, "OVD self predictions")
    ovd_metrics_path = metrics_dir / "ovd_self_metrics.json"
    if ovd_pred.is_file() and ovd_metrics_path.is_file():
        n_img = count_unique_images(ovd_pred)
        mAP = float(json.loads(ovd_metrics_path.read_text(encoding="utf-8"))["mAP"])
        ok_img = n_img >= args.min_ovd_images
        ok_map = mAP > args.baseline_map
        print(f"[{'OK' if ok_img else 'FAIL'}] OVD unique image_id: {n_img} (>= {args.min_ovd_images})")
        print(f"[{'OK' if ok_map else 'FAIL'}] OVD mAP: {mAP:.4f} (> {args.baseline_map})")
        ok &= ok_img and ok_map

    vg_metrics_path = metrics_dir / "vg_full_metrics.json"
    if vg_metrics_path.is_file():
        metrics = json.loads(vg_metrics_path.read_text(encoding="utf-8"))
        for dataset, splits in DATASET_SPLITS.items():
            file_dataset = dataset.replace("+", "+")
            for split in splits:
                split_metrics = metrics.get(dataset, {}).get(split)
                split_slug = "validation" if split == "validation" else split
                pred_name = f"vg_{file_dataset}_{split_slug}_predictions.json"
                pred_path = predictions_dir / pred_name
                ok &= check_file(pred_path, f"VG predictions {dataset}/{split}")
                if split_metrics is None:
                    print(f"[FAIL] VG metrics missing split {dataset}/{split}")
                    ok = False
                    continue
                acc = float(split_metrics.get("acc", 0))
                n_total = int(split_metrics.get("n_total", 0))
                ok_split = n_total > 1000
                print(f"[{'OK' if ok_split else 'FAIL'}] VG {dataset}/{split}: acc={acc*100:.2f}% n_total={n_total}")
                ok &= ok_split

    print("\n=== Final supplementary validation:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
