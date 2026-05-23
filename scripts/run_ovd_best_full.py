#!/usr/bin/env python3
"""用 sweep 最佳阈值跑 OVD 全量。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.utils.codec  # noqa: F401

from src.ovd.eval_coco import run_eval
from src.utils.io import load_yaml, save_json


def main():
    sweep_best = Path("results/exp_2026-05-23_ovd_sweep/best_subset.json")
    if not sweep_best.exists():
        raise FileNotFoundError(f"缺少 sweep 结果: {sweep_best}")

    import json

    best = json.loads(sweep_best.read_text(encoding="utf-8"))
    cfg = load_yaml("configs/coco_ovd.yaml")
    cfg["inference"]["box_threshold"] = best["box_threshold"]
    cfg["inference"]["text_threshold"] = best["text_threshold"]

    out = Path("results/exp_2026-05-23_ovd_aligned")
    out.mkdir(parents=True, exist_ok=True)
    cfg["output"]["predictions_path"] = str(out / "predictions.json")
    cfg["output"]["metrics_path"] = str(out / "metrics.json")

    print("Best thresholds:", best["box_threshold"], best["text_threshold"])
    metrics = run_eval(cfg, subset=None)
    save_json(
        {
            "box_threshold": best["box_threshold"],
            "text_threshold": best["text_threshold"],
            "prompt_mode": cfg.get("prompt", {}).get("mode"),
            **metrics,
        },
        out / "manifest.json",
    )
    print(metrics)


if __name__ == "__main__":
    main()
