#!/usr/bin/env bash
# 等待 OVD sweep 完成，然后按最佳完成标准继续流水线
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
LOG=/tmp/pipeline_continue.log
exec >>"$LOG" 2>&1

echo "=== $(date -Iseconds) 等待 OVD sweep 完成 ==="
while ! test -f results/exp_2026-05-23_ovd_sweep/best_subset.json; do
  if ! pgrep -f "sweep_ovd_thresholds.py" >/dev/null; then
  if ! test -f results/exp_2026-05-23_ovd_sweep/sweep_summary.json; then
    echo "ERROR: sweep 进程已退出但无 best_subset.json"
    tail -30 /tmp/ovd_sweep.log || true
    exit 1
  fi
  fi
  sleep 30
done
cat results/exp_2026-05-23_ovd_sweep/best_subset.json

echo "=== $(date -Iseconds) OVD 全量 5k ==="
python3 scripts/run_ovd_best_full.py
python3 scripts/validate_ovd_artifacts.py \
  --predictions results/exp_2026-05-23_ovd_aligned/predictions.json \
  --metrics results/exp_2026-05-23_ovd_aligned/metrics.json \
  --min-images 4990 || true

echo "=== $(date -Iseconds) 准备 VG：解压 train2014（若需要）==="
if test -f data/coco/train2014.zip && ! test -d data/coco/train2014; then
  unzip -q -o data/coco/train2014.zip -d data/coco
fi

echo "=== $(date -Iseconds) VG sweep ==="
python3 scripts/sweep_vg_thresholds.py
python3 scripts/apply_vg_best_thresholds.py

echo "=== $(date -Iseconds) VG 全量 semantic --all ==="
python3 scripts/run_vg_eval.py --all --fresh-metrics --hf-dir data/refcoco_hf

echo "=== $(date -Iseconds) 流水线阶段完成 ==="
