#!/usr/bin/env python3
"""「最佳完成」全量验收：OVD + VG + sweep 产物。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_vg_eval import DATASET_SPLITS  # noqa: E402
from scripts.validate_ovd_artifacts import count_unique_images  # noqa: E402

BASELINE = ROOT / "results/exp_2026-05-23_baseline/refcoco_gdino.metrics.baseline.json"


def check_file(path: Path, label: str) -> bool:
    ok = path.is_file()
    print(f"[{'OK' if ok else 'FAIL'}] {label}: {path}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ovd-dir", type=Path, default=Path("results/exp_2026-05-23_ovd_aligned"))
    parser.add_argument("--ovd-sweep-dir", type=Path, default=Path("results/exp_2026-05-23_ovd_sweep"))
    parser.add_argument("--vg-sweep-dir", type=Path, default=Path("results/exp_2026-05-23_vg_sweep"))
    parser.add_argument("--vg-metrics", type=Path, default=Path("results/refcoco_gdino/metrics.json"))
    parser.add_argument("--baseline-map", type=float, default=0.424)
    parser.add_argument("--min-ovd-images", type=int, default=4990)
    args = parser.parse_args()

    ok = True

    # sweep 产物
    ok &= check_file(args.ovd_sweep_dir / "best_subset.json", "OVD sweep best_subset")
    ok &= check_file(args.ovd_sweep_dir / "sweep_summary.json", "OVD sweep summary")
    ok &= check_file(args.vg_sweep_dir / "best_subset.json", "VG sweep best_subset")

    # OVD 全量
    pred = args.ovd_dir / "predictions.json"
    met = args.ovd_dir / "metrics.json"
    if pred.is_file() and met.is_file():
        n_img = count_unique_images(pred)
        mAP = float(json.loads(met.read_text())["mAP"])
        ok_img = n_img >= args.min_ovd_images
        ok_map = mAP > args.baseline_map
        print(
            f"[{'OK' if ok_img else 'FAIL'}] OVD unique image_id: {n_img} (>= {args.min_ovd_images})"
        )
        print(f"[{'OK' if ok_map else 'FAIL'}] OVD mAP: {mAP:.4f} (> {args.baseline_map})")
        ok &= ok_img and ok_map
    else:
        print(f"[FAIL] OVD aligned artifacts missing under {args.ovd_dir}")
        ok = False

    # VG 全量：各 split n_total 与 baseline 一致（全量规模）
    if not args.vg_metrics.is_file():
        print(f"[FAIL] VG metrics missing: {args.vg_metrics}")
        ok = False
    else:
        metrics = json.loads(args.vg_metrics.read_text())
        baseline = json.loads(BASELINE.read_text()) if BASELINE.is_file() else {}
        for ds, splits in DATASET_SPLITS.items():
            for sp in splits:
                key_ok = ds in metrics and sp in metrics[ds]
                if not key_ok:
                    print(f"[FAIL] VG missing split {ds}/{sp}")
                    ok = False
                    continue
                n = metrics[ds][sp].get("n_total", 0)
                expected = baseline.get(ds, {}).get(sp, {}).get("n_total")
                if expected is None:
                    print(f"[WARN] no baseline n_total for {ds}/{sp}")
                    ok_split = n > 1000
                else:
                    ok_split = n == expected
                acc = metrics[ds][sp].get("acc", 0)
                print(
                    f"[{'OK' if ok_split else 'FAIL'}] VG {ds}/{sp}: "
                    f"acc={acc*100:.2f}% n_total={n} (expected {expected})"
                )
                ok &= ok_split

    # CUDA 记录
    cuda_status = ROOT / "results/exp_2026-05-23_cuda_ops/status.json"
    ok &= check_file(cuda_status, "CUDA ops status")

    print("\n=== 总验收:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
