#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if test -f /root/miniconda3/etc/profile.d/conda.sh; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate cv
fi
LOG=/tmp/vg_pipeline.log
exec >>"$LOG" 2>&1

echo "=== $(date -Iseconds) 等待 train2014 解压 ==="
ZIP=data/coco/train2014.zip
DIR=data/coco/train2014
while true; do
  if test -d "$DIR" && test "$(find "$DIR" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)" -gt 80000; then
    break
  fi
  if test -f "$ZIP" && ! pgrep -f "aria2c.*train2014" >/dev/null; then
    if unzip -t "$ZIP" >/dev/null 2>&1; then
      echo "解压 $ZIP ..."
      unzip -q -o "$ZIP" -d data/coco
    fi
  fi
  sleep 120
done
echo "train2014 images: $(find "$DIR" -maxdepth 1 -name '*.jpg' | wc -l)"

echo "=== $(date -Iseconds) VG sweep ==="
python3 scripts/sweep_vg_thresholds.py
python3 scripts/apply_vg_best_thresholds.py

echo "=== $(date -Iseconds) VG 全量 semantic ==="
mkdir -p results/refcoco_gdino
mv -f results/refcoco_gdino/metrics.json results/refcoco_gdino/metrics.json.bak.$(date +%s) 2>/dev/null || true
python3 scripts/run_vg_eval.py --all --fresh-metrics --hf-dir data/refcoco_hf

echo "=== $(date -Iseconds) VG 完成 ==="
