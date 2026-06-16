#!/usr/bin/env bash
# 监视 train2014 下载 → 解压 → VG sweep → 全量 eval → 更新 report，直至完成。
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# conda cv（含 torch / groundingdino）
if test -f /root/miniconda3/etc/profile.d/conda.sh; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate cv
fi

LOG=/tmp/full_pipeline_watch.log
ZIP=data/coco/train2014.zip
DIR=data/coco/train2014
EXPECTED_SIZE=13510573713

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== 监视器启动 ==="

zip_ready() {
  local sz
  sz=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
  test "$sz" -eq "$EXPECTED_SIZE" && unzip -t "$ZIP" >/dev/null 2>&1
}

# --- 阶段 1: 等待 train2014 下载完成且 zip 校验通过 ---
while ! zip_ready; do
  if pgrep -f 'aria2c.*train2014' >/dev/null 2>&1; then
    du_h=$(du -h "$ZIP" 2>/dev/null | cut -f1 || echo 0)
    prog=$(grep -oE '\[#[^]]+\]' /tmp/train2014_aria2.log 2>/dev/null | tail -1 || echo "?")
    log "下载中: 实占=${du_h} ${prog}"
  else
    sz=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
    log "aria2 未运行，zip size=$sz / $EXPECTED_SIZE；等待校验或重启下载..."
    if test "$sz" -lt "$EXPECTED_SIZE" && ! pgrep -f 'aria2c.*train2014' >/dev/null 2>&1; then
      log "重新启动 aria2 断点续传..."
      (cd data/coco && nohup aria2c -c -x16 -s16 -k2M --file-allocation=none \
        --summary-interval=120 -o train2014.zip \
        "http://images.cocodataset.org/zips/train2014.zip" >> /tmp/train2014_aria2.log 2>&1 &)
    fi
  fi
  sleep 300
done
log "zip 校验通过 size=$EXPECTED_SIZE"

# --- 阶段 2: 解压 ---
nimg=$(find "$DIR" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
if test "$nimg" -lt 80000; then
  log "解压 $ZIP → data/coco ..."
  unzip -q -o "$ZIP" -d data/coco
  nimg=$(find "$DIR" -maxdepth 1 -name '*.jpg' | wc -l)
fi
log "train2014 图像数: $nimg"
if test "$nimg" -lt 80000; then
  log "ERROR: 图像数不足"
  exit 1
fi

# --- 阶段 3: VG sweep（若未完成）---
SWEEP=results/exp_2026-05-23_vg_sweep/best_subset.json
if test ! -f "$SWEEP" ]; then
  log "=== VG sweep ==="
  python3 scripts/sweep_vg_thresholds.py
  python3 scripts/apply_vg_best_thresholds.py
else
  log "VG sweep 已存在，跳过"
fi

# --- 阶段 4: VG 全量 ---
METRICS=results/refcoco_gdino/metrics.json
if ! python3 -c "
import json, sys
from pathlib import Path
p = Path('$METRICS')
if not p.exists(): sys.exit(1)
m = json.loads(p.read_text())
n = m.get('refcoco', {}).get('validation', {}).get('n_total', 0)
sys.exit(0 if n > 10000 else 1)
" 2>/dev/null; then
  log "=== VG 全量 semantic --all --fresh-metrics ==="
  mkdir -p results/refcoco_gdino
  mv -f results/refcoco_gdino/metrics.json results/refcoco_gdino/metrics.json.bak.$(date +%s) 2>/dev/null || true
  python3 scripts/run_vg_eval.py --all --fresh-metrics --hf-dir data/refcoco_hf
else
  log "VG 全量 metrics 已存在，跳过"
fi

# --- 阶段 5: 更新 report §0 ---
log "=== 更新 reports/report.md ==="
python3 - <<'PY'
import json
import re
from pathlib import Path

report = Path("reports/report.md")
text = report.read_text()

sweep = Path("results/exp_2026-05-23_vg_sweep/best_subset.json")
metrics = Path("results/refcoco_gdino/metrics.json")

if sweep.exists():
    best = json.loads(sweep.read_text())
    sweep_line = f"box={best.get('box_threshold')}, text={best.get('text_threshold')}, subset acc={best.get('acc',0)*100:.2f}%"
else:
    sweep_line = "TBD"

if metrics.exists():
    m = json.loads(metrics.read_text())
    rc = m.get("refcoco", {}).get("validation", {})
    acc_val = f"{rc.get('acc', 0) * 100:.2f}%"
    n_val = rc.get("n_total", "TBD")
    vg_all_line = f"**{acc_val}**"
    vg_n_line = f"n_total=**{n_val}**"
    vg_status = "**已完成**"
else:
    vg_all_line = "TBD"
    vg_n_line = "TBD"
    vg_status = "**进行中**"

text = re.sub(
    r"\| VG semantic \*\*全量 --all\*\* \| — \| — \| TBD \| 进行中.*?\| \*\*进行中\*\* \|",
    f"| VG semantic **全量 --all** | — | — | {vg_all_line} | {vg_n_line} | {vg_status} |",
    text,
)
text = re.sub(
    r"\| VG sweep \| — \| — \| TBD \| 排队 \| \*\*进行中\*\* \|",
    f"| VG sweep | — | — | {sweep_line.split(',')[2] if 'acc=' in sweep_line else 'TBD'} | {sweep_line} | **已完成** |" if sweep.exists() else text,
    text,
)
text = re.sub(
    r"3\. \*\*VG 进行中\*\*：.*",
    f"3. **VG 全量已完成**：sweep {sweep_line}；RefCOCO val Acc@0.5 {acc_val if metrics.exists() else 'TBD'}（`results/refcoco_gdino/metrics.json`）。",
    text,
)

report.write_text(text)
print("report updated")
PY

# --- 阶段 6: 最终验收 ---
log "=== 最终验收 validate_all.py ==="
if python3 scripts/validate_all.py | tee -a "$LOG"; then
  log "验收 PASS"
  touch /tmp/pipeline_all_done.flag
else
  log "验收 FAIL — 见上方日志"
  exit 1
fi

# --- 阶段 7: 推送 GitHub ---
log "=== 推送 GitHub ==="
if bash scripts/push_to_github.sh; then
  log "GitHub 推送完成"
else
  log "GitHub 推送失败 — 见 /tmp/github_push.log"
fi

log "=== 全部完成 ==="
