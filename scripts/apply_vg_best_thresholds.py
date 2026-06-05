#!/usr/bin/env python3
"""将 VG sweep 最佳阈值写回 configs/refcoco.yaml。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml

BEST = Path("results/exp_2026-05-23_vg_sweep/best_subset.json")
CFG = Path("configs/refcoco.yaml")


def main() -> None:
    if not BEST.exists():
        raise FileNotFoundError(f"缺少 {BEST}，请先运行 sweep_vg_thresholds.py")
    best = json.loads(BEST.read_text(encoding="utf-8"))
    cfg = load_yaml(str(CFG))
    cfg["inference"]["box_threshold"] = best["box_threshold"]
    cfg["inference"]["text_threshold"] = best["text_threshold"]
    with open(CFG, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"已更新 {CFG}: box={best['box_threshold']} text={best['text_threshold']}")


if __name__ == "__main__":
    main()
