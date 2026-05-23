"""下载 Grounding DINO 预训练权重和模型配置文件。

用法：
    python scripts/download_weights.py
    python scripts/download_weights.py --weights-dir weights/

依赖：requests（conda 环境中已包含）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


WEIGHTS = {
    "groundingdino_swint_ogc.pth": (
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha/groundingdino_swint_ogc.pth"
    ),
    "groundingdino_swinb_cogcoor.pth": (
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    ),
}

CONFIG_SRC = (
    "third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
CONFIG_DST = "configs/GroundingDINO_SwinT_OGC.py"


def download_file(url: str, dst: Path) -> None:
    import requests
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"  [跳过] 已存在: {dst}")
        return

    print(f"  下载: {url}")
    print(f"    → {dst}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    {pct:.1f}% ({downloaded // 1024 // 1024} MB)", end="", flush=True)
    print(f"\n  完成: {dst.stat().st_size // 1024 // 1024} MB")


def copy_config() -> None:
    src = Path(CONFIG_SRC)
    dst = Path(CONFIG_DST)
    if not src.exists():
        print(f"  [警告] 未找到模型配置文件: {src}")
        print("  请先执行: git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO")
        return
    if dst.exists():
        print(f"  [跳过] 已存在: {dst}")
        return
    import shutil
    shutil.copy2(src, dst)
    print(f"  已复制配置文件: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", default="weights/", help="权重保存目录")
    parser.add_argument(
        "--model",
        choices=["swint", "swinb", "all"],
        default="swint",
        help="下载哪个权重（swint=Swin-T OGC，swinb=Swin-B，all=全部）",
    )
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)

    to_download = {}
    if args.model in ("swint", "all"):
        to_download["groundingdino_swint_ogc.pth"] = WEIGHTS["groundingdino_swint_ogc.pth"]
    if args.model in ("swinb", "all"):
        to_download["groundingdino_swinb_cogcoor.pth"] = WEIGHTS["groundingdino_swinb_cogcoor.pth"]

    print("=== 下载 Grounding DINO 权重 ===")
    for filename, url in to_download.items():
        download_file(url, weights_dir / filename)

    print("\n=== 复制模型配置文件 ===")
    copy_config()

    print("\n完成！下一步：")
    print("  pip install -e third_party/GroundingDINO  （或配置 backend: hf）")
    print("  python src/ovd/eval_coco.py --config configs/coco_ovd.yaml --subset 100")


if __name__ == "__main__":
    main()
