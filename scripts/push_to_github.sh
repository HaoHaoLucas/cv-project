#!/usr/bin/env bash
# 验收通过后提交并推送到 GitHub（跳过 data/ weights/ 等大文件，遵循 .gitignore）
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=/tmp/github_push.log
exec >>"$LOG" 2>&1

log() { echo "[$(date -Iseconds)] $*"; }

if ! test -f /tmp/pipeline_all_done.flag; then
  log "ERROR: pipeline 未完成，拒绝推送"
  exit 1
fi

PYTHON="${PYTHON:-}"
if test -x "${CONDA_PREFIX:-}/bin/python3" && "${CONDA_PREFIX:-}/bin/python3" -c "import numpy" 2>/dev/null; then
  PYTHON="${CONDA_PREFIX}/bin/python3"
elif test -x /root/miniconda3/envs/cv/bin/python3; then
  PYTHON=/root/miniconda3/envs/cv/bin/python3
else
  PYTHON=python3
fi

if ! "$PYTHON" scripts/validate_all.py; then
  log "ERROR: validate_all 未通过，拒绝推送"
  exit 1
fi

# 导出 GroundingDINO patch（避免提交嵌套 .git）
PATCH_DIR=patches
mkdir -p "$PATCH_DIR"
if test -d third_party/GroundingDINO/.git; then
  git -C third_party/GroundingDINO diff groundingdino/models/GroundingDINO/ms_deform_attn.py \
    > "$PATCH_DIR/groundingdino_ms_deform_attn_fallback.patch" 2>/dev/null || true
fi

log "=== git add ==="
git add \
  configs/ \
  reports/report.md \
  docs/ \
  scripts/ \
  src/ \
  patches/ \
  results/exp_2026-05-23_ovd_aligned/ \
  results/exp_2026-05-23_ovd_sweep/ \
  results/exp_2026-05-23_cuda_ops/ \
  results/refcoco_gdino/metrics.json \
  2>/dev/null || true

# 若 VG sweep 完成则加入
test -d results/exp_2026-05-23_vg_sweep && git add results/exp_2026-05-23_vg_sweep/

if git diff --cached --quiet; then
  log "无新变更可提交"
else
  git commit -m "$(cat <<'EOF'
完成 OVD/VG 高精度复现流水线与验收产物

OVD sweep 全量 mAP 46.16；VG sweep + semantic 全量评测；
新增监视/验收脚本与 report §0 矩阵更新。
EOF
)"
  log "commit 完成: $(git log -1 --oneline)"
fi

log "=== git push origin main ==="
export GIT_TERMINAL_PROMPT=0
if git push origin main; then
  log "推送成功"
  touch /tmp/github_push_done.flag
else
  log "推送失败 — 请检查 GitHub 凭据（SSH key 或 PAT）"
  exit 1
fi
