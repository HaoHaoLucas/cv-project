#!/usr/bin/env python3
"""运行官方 GroundingDINO COCO zero-shot AP 评测并保存 metrics。

封装 third_party/GroundingDINO/demo/test_ap_on_coco.py 的评测逻辑，
将 mAP 写入 results/exp_<date>_official_ovd/metrics.json。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "third_party" / "GroundingDINO" / "demo" / "test_ap_on_coco.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--weights", default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--anno-path", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--image-dir", default="data/coco/val2017")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="默认 results/exp_<today>_official_ovd",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = args.out_dir or (ROOT / "results" / f"exp_{date.today()}_official_ovd")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    if not DEMO.is_file():
        print(f"FAIL: 未找到 {DEMO}", file=sys.stderr)
        return 1
    if not Path(args.weights).is_file():
        print(f"FAIL: 权重不存在 {args.weights}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        str(DEMO),
        "-c",
        str(ROOT / args.config),
        "-p",
        str(ROOT / args.weights),
        "--anno_path",
        str(ROOT / args.anno_path),
        "--image_dir",
        str(ROOT / args.image_dir),
        "--num_workers",
        str(args.num_workers),
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    print(proc.stdout[-3000:] if proc.stdout else "")
    if proc.stderr:
        print(proc.stderr[-2000:], file=sys.stderr)

    mAP = None
    for line in (proc.stdout or "").splitlines():
        if "Final results:" in line:
            # Final results: [0.484, ...]
            import ast

            try:
                stats = ast.literal_eval(line.split(":", 1)[1].strip())
                mAP = float(stats[0])
            except Exception:
                pass

    metrics = {
        "source": "third_party/GroundingDINO/demo/test_ap_on_coco.py",
        "mAP": mAP,
        "exit_code": proc.returncode,
        "log": str(log_path),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # 对比本仓库 OVD 最佳
    ours_path = ROOT / "results/exp_2026-05-23_ovd_aligned/metrics.json"
    if ours_path.is_file() and mAP is not None:
        ours = json.loads(ours_path.read_text())
        comparison = {
            "official_mAP": mAP,
            "ours_mAP": ours.get("mAP"),
            "delta": mAP - float(ours.get("mAP", 0)),
        }
        (out_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n对比: official={mAP:.4f} ours={ours.get('mAP'):.4f} delta={comparison['delta']:+.4f}")

    return 0 if proc.returncode == 0 and mAP is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
