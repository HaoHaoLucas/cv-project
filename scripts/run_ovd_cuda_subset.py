#!/usr/bin/env python3
"""CUDA 算子编译后 OVD 子集消融（对比 fallback 基线）。"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.utils.codec  # noqa: F401

from src.ovd.eval_coco import run_eval
from src.utils.io import load_yaml, save_json


def main() -> int:
    subset = int(os.environ.get("OVD_SUBSET", "500"))
    out = ROOT / "results" / f"exp_{date.today()}_cuda_built"
    out.mkdir(parents=True, exist_ok=True)

    sweep_best = ROOT / "results/exp_2026-05-23_ovd_sweep/best_subset.json"
    best = json.loads(sweep_best.read_text(encoding="utf-8"))
    cfg = load_yaml("configs/coco_ovd.yaml")
    cfg["inference"]["box_threshold"] = best["box_threshold"]
    cfg["inference"]["text_threshold"] = best["text_threshold"]
    cfg["output"]["predictions_path"] = str(out / f"predictions_subset{subset}.json")
    cfg["output"]["metrics_path"] = str(out / f"metrics_subset{subset}.json")

    metrics = run_eval(cfg, subset=subset)
    save_json(
        {
            "subset": subset,
            "cuda_status": json.loads((out / "status.json").read_text())
            if (out / "status.json").is_file()
            else None,
            "box_threshold": best["box_threshold"],
            "text_threshold": best["text_threshold"],
            **metrics,
        },
        out / f"manifest_subset{subset}.json",
    )

    baseline_path = ROOT / "results/exp_2026-05-23_ovd_aligned/metrics.json"
    if baseline_path.is_file() and metrics.get("mAP") is not None:
        ours_full = json.loads(baseline_path.read_text())
        ablation = {
            "cuda_subset_mAP": metrics.get("mAP"),
            "fallback_full_mAP": ours_full.get("mAP"),
            "subset": subset,
            "note": "子集 mAP 不可直接与全量 46.16 对比，仅验证 CUDA 管线可跑通",
        }
        save_json(ablation, out / "ablation_vs_baseline.json")

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
