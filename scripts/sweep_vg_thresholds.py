#!/usr/bin/env python3
"""VG 阈值网格扫描（子集）。"""
from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.utils.codec  # noqa: F401

from scripts.run_vg_eval import DATASET_SPLITS, eval_one_split, load_parquet_samples
from src.model_wrapper import GroundingDINOWrapper
from src.utils.io import load_yaml, save_json

BOX_THRESH = [0.05, 0.10, 0.20]
TEXT_THRESH = [0.05, 0.10, 0.20]
MAX_SAMPLES = 500


def main():
    base_cfg = load_yaml("configs/refcoco.yaml")
    out_dir = Path("results/exp_2026-05-23_vg_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    coco_img_dir = Path(base_cfg["data"]["coco_img_dir"])
    hf_dir = Path("data/refcoco_hf")
    model = GroundingDINOWrapper.from_config(base_cfg)

    # 用 refcoco validation 子集代表扫描
    samples = load_parquet_samples(hf_dir / "refcoco" / "validation.parquet", coco_img_dir)

    results = []
    for box_thr, txt_thr in itertools.product(BOX_THRESH, TEXT_THRESH):
        infer = copy.deepcopy(base_cfg.get("inference", {}))
        infer["box_threshold"] = box_thr
        infer["text_threshold"] = txt_thr
        tag = f"b{box_thr}_t{txt_thr}"
        print(f"\n=== VG sweep {tag} max={MAX_SAMPLES} ===")
        metrics = eval_one_split(
            model, samples, infer_cfg=infer, max_samples=MAX_SAMPLES
        )
        row = {
            "box_threshold": box_thr,
            "text_threshold": txt_thr,
            "acc": metrics["acc"],
            "n_hit": metrics["n_hit"],
            "n_total": metrics["n_total"],
        }
        results.append(row)

    results.sort(key=lambda x: x["acc"], reverse=True)
    save_json(results, out_dir / "sweep_summary.json")
    save_json(results[0], out_dir / "best_subset.json")
    print("Best:", results[0])


if __name__ == "__main__":
    main()
