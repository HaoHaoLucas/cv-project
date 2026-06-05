#!/usr/bin/env bash
# 按「最佳完成」标准依次执行复现流水线（需在 GPU + 数据就绪环境运行）
set -euo pipefail
cd "$(dirname "$0")/.."
export REPO="$(pwd)"

echo "=== 1. 环境检查 ==="
python scripts/setup_env.py

echo "=== 2. HF text_threshold 已修复于 src/model_wrapper.py ==="

echo "=== 3. CUDA 算子 ==="
python scripts/try_cuda_ops.py || true

echo "=== 4. OVD sweep (1k 子集) ==="
python scripts/sweep_ovd_thresholds.py
test -f results/exp_2026-05-23_ovd_sweep/best_subset.json

echo "=== 5. OVD 全量 5k（最佳阈值）==="
python scripts/run_ovd_best_full.py
python scripts/validate_ovd_artifacts.py \
  --predictions results/exp_2026-05-23_ovd_aligned/predictions.json \
  --metrics results/exp_2026-05-23_ovd_aligned/metrics.json

echo "=== 6. VG sweep ==="
python scripts/sweep_vg_thresholds.py
test -f results/exp_2026-05-23_vg_sweep/best_subset.json

python scripts/apply_vg_best_thresholds.py

echo "=== 7. VG 全量 semantic --all ==="
python scripts/run_vg_eval.py --all --fresh-metrics --hf-dir data/refcoco_hf

echo "=== 8. 完成；请更新 reports/report.md 实验矩阵 ==="
