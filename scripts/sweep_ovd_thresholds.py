#!/usr/bin/env python3
"""OVD 阈值网格扫描（子集）。"""
from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.utils.codec  # noqa: F401

from src.ovd.eval_coco import run_eval
from src.utils.io import load_yaml, save_json

BOX_THRESH = [0.15, 0.20, 0.25]
TEXT_THRESH = [0.05, 0.10, 0.20]
SUBSET = 1000


def main():
    base_cfg = load_yaml("configs/coco_ovd.yaml")
    out_dir = Path("results/exp_2026-05-23_ovd_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.log").touch()

    results = []
    for box_thr, txt_thr in itertools.product(BOX_THRESH, TEXT_THRESH):
        cfg = copy.deepcopy(base_cfg)
        cfg["inference"]["box_threshold"] = box_thr
        cfg["inference"]["text_threshold"] = txt_thr
        tag = f"b{box_thr}_t{txt_thr}"
        cfg["output"]["predictions_path"] = str(out_dir / f"pred_{tag}.json")
        cfg["output"]["metrics_path"] = str(out_dir / f"metrics_{tag}.json")

        print(f"\n=== sweep {tag} subset={SUBSET} ===")
        metrics = run_eval(cfg, subset=SUBSET)
        row = {"box_threshold": box_thr, "text_threshold": txt_thr, **metrics}
        results.append(row)

    results.sort(key=lambda x: x.get("mAP", 0), reverse=True)
    save_json(results, out_dir / "sweep_summary.json")
    save_json(results[0], out_dir / "best_subset.json")
    print("Best:", results[0])


if __name__ == "__main__":
    main()
