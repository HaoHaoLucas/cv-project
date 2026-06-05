#!/usr/bin/env python3
"""下载 RefCOCO/+/g 的 HuggingFace parquet 标注（供 run_vg_eval.py 使用）。

用法:
    python scripts/download_refcoco_hf.py
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_refcoco_hf.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# HF 仓库与本地目录名；parquet 在 data/<split>-00000-of-00001-*.parquet
DATASETS = {
    "refcoco": "jxu124/refcoco",
    "refcoco+": "jxu124/refcocoplus",
    "refcocog": "jxu124/refcocog",
}
SPLITS = {
    "refcoco": ["validation", "testB"],
    "refcoco+": ["validation", "testB"],
    "refcocog": ["validation"],
}


def _find_parquet(repo_id: str, split: str) -> str:
    from huggingface_hub import list_repo_files

    prefix = f"data/{split}-"
    for name in list_repo_files(repo_id, repo_type="dataset"):
        if name.startswith(prefix) and name.endswith(".parquet"):
            return name
    raise FileNotFoundError(f"{repo_id}: no parquet for split {split}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/refcoco_hf"))
    args = parser.parse_args()
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("请安装: pip install huggingface_hub")
        raise SystemExit(1)

    for name, repo_id in DATASETS.items():
        ds_dir = out_root / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS[name]:
            dest = ds_dir / f"{split}.parquet"
            if dest.exists() and dest.stat().st_size > 1000:
                print(f"[跳过] {dest}")
                continue
            hf_name = _find_parquet(repo_id, split)
            print(f"下载 {repo_id} / {hf_name} -> {dest}")
            path = hf_hub_download(repo_id=repo_id, filename=hf_name, repo_type="dataset")
            shutil.copy2(path, dest)
            print(f"  OK {dest.stat().st_size // 1024} KiB")

    print(f"\n完成。运行: python scripts/run_vg_eval.py --all --fresh-metrics --hf-dir {out_root}")


if __name__ == "__main__":
    main()
