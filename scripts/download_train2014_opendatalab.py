#!/usr/bin/env python3
"""从 OpenDataLab（国内 CDN）下载 COCO train2014.zip。

需先配置 OpenXLab 凭据（任选其一）：
  1. openxlab login
  2. 环境变量 OPENXLAB_AK / OPENXLAB_SK
  3. python -c "import openxlab; openxlab.login(ak='...', sk='...')"

用法：
    python scripts/download_train2014_opendatalab.py
    python scripts/download_train2014_opendatalab.py --target data/coco
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_REPO = "OpenDataLab/COCO_2014"
SOURCE_PATH = "/raw/train2014.zip"
EXPECTED_SIZE = 13_510_573_713  # OpenDataLab 标注 12.58 GiB，与官方 zip 接近


def _login_if_needed() -> None:
    import openxlab

    ak = os.environ.get("OPENXLAB_AK", "").strip()
    sk = os.environ.get("OPENXLAB_SK", "").strip()
    if ak and sk:
        openxlab.login(ak=ak, sk=sk)
        return
    cfg = Path.home() / ".openxlab" / "config.json"
    if cfg.is_file():
        return
    print(
        "未找到 OpenXLab 凭据。请先执行：\n"
        "  openxlab login\n"
        "或在 https://sso.openxlab.org.cn 个人中心 → 密钥管理 获取 AK/SK，然后：\n"
        "  export OPENXLAB_AK=... OPENXLAB_SK=...\n"
        "  python scripts/download_train2014_opendatalab.py",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=Path("data/coco"))
    args = parser.parse_args()
    target = args.target
    target.mkdir(parents=True, exist_ok=True)
    dest = target / "train2014.zip"

    if dest.is_file() and dest.stat().st_size >= EXPECTED_SIZE * 0.99:
        print(f"[跳过] 已存在完整 zip: {dest} ({dest.stat().st_size // 1024 // 1024} MB)")
        return

    _login_if_needed()

    from openxlab.dataset import download

    print(f"从 OpenDataLab CDN 下载 {SOURCE_PATH} → {target}")
    download(
        dataset_repo=DATASET_REPO,
        source_path=SOURCE_PATH,
        target_path=str(target),
    )
    if not dest.is_file():
        # openxlab 可能保留原始文件名在子目录
        candidates = list(target.rglob("train2014.zip"))
        if len(candidates) == 1 and candidates[0] != dest:
            candidates[0].rename(dest)
    size_mb = dest.stat().st_size // 1024 // 1024 if dest.is_file() else 0
    print(f"完成: {dest} ({size_mb} MB)")


if __name__ == "__main__":
    main()
