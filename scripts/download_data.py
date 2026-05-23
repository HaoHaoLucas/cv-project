"""数据集下载辅助脚本。

提供 COCO val2017、COCO train2014（RefCOCO 图像来源）
以及 RefCOCO/+/g 标注文件的下载和解压说明，
并在可自动化的地方直接执行。

用法：
    python scripts/download_data.py --dataset coco-val
    python scripts/download_data.py --dataset coco-train2014
    python scripts/download_data.py --dataset refcoco

注意：
  COCO 图像压缩包较大（val2017 ~1 GB，train2014 ~13 GB），
  建议先确认磁盘空间充足。
  RefCOCO 标注文件需登录后从 UNC 官方页面手动下载。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


COCO_VAL_IMGS = "http://images.cocodataset.org/zips/val2017.zip"
COCO_TRAIN14_IMGS = "http://images.cocodataset.org/zips/train2014.zip"
COCO_VAL_ANNS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# RefCOCO 标注需手动下载，此处仅提供路径说明
REFCOCO_MANUAL_URL = "https://github.com/lichengunc/refer"


def _download(url: str, dest: Path) -> Path:
    """使用 wget 或 requests 下载文件。返回下载后的文件路径。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    filename = dest / url.split("/")[-1]
    if filename.exists():
        print(f"  [跳过] 已存在: {filename}")
        return filename

    print(f"  下载: {url}")
    # 优先用 wget（速度更快、自带断点续传）
    wget_path = shutil.which("wget")
    if wget_path:
        subprocess.run([wget_path, "-c", url, "-O", str(filename)], check=True)
    else:
        import requests
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    f.write(chunk)
    print(f"  完成: {filename}")
    return filename


def _extract(zip_path: Path, dest: Path) -> None:
    """解压 zip 文件到目标目录。"""
    print(f"  解压: {zip_path} → {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print(f"  解压完成")


def download_coco_val(data_root: Path = Path("data/coco")) -> None:
    """下载 COCO val2017 图像 + 标注文件。"""
    print("\n=== COCO val2017 ===")

    imgs_zip = _download(COCO_VAL_IMGS, data_root)
    _extract(imgs_zip, data_root)

    anns_zip = _download(COCO_VAL_ANNS, data_root)
    _extract(anns_zip, data_root)

    print("\n  验证：")
    val_dir = data_root / "val2017"
    ann_file = data_root / "annotations" / "instances_val2017.json"
    print(f"  val2017/   存在={val_dir.exists()}  文件数≈{len(list(val_dir.glob('*.jpg'))) if val_dir.exists() else 0}")
    print(f"  instances_val2017.json 存在={ann_file.exists()}")


def download_coco_train2014(data_root: Path = Path("data/coco")) -> None:
    """下载 COCO train2014 图像（供 RefCOCO 使用）。"""
    print("\n=== COCO train2014 ===")
    print("  注意：文件约 13 GB，请确保磁盘空间充足。")

    imgs_zip = _download(COCO_TRAIN14_IMGS, data_root)
    _extract(imgs_zip, data_root)

    train_dir = data_root / "train2014"
    print(f"\n  验证：train2014/ 存在={train_dir.exists()}")


def show_refcoco_guide() -> None:
    """输出 RefCOCO 手动下载指南。"""
    print("""
=== RefCOCO / RefCOCO+ / RefCOCOg 下载指南 ===

RefCOCO 标注文件需手动下载，步骤如下：

1. 访问官方仓库：https://github.com/lichengunc/refer
   或直接下载标注 zip（需 UNC 账号）：
   https://bvisionweb1.cs.unc.edu/lichengunc/refer/data/refcoco.zip
   https://bvisionweb1.cs.unc.edu/lichengunc/refer/data/refcoco+.zip
   https://bvisionweb1.cs.unc.edu/lichengunc/refer/data/refcocog.zip

2. 将解压后的文件夹放到：
   data/refcoco/refcoco/   → 包含 refs(unc).p + instances.json
   data/refcoco/refcoco+/  → 包含 refs(unc).p + instances.json
   data/refcoco/refcocog/  → 包含 refs(google).p + instances.json

3. （可选）克隆 refer 工具到 third_party/：
   git clone https://github.com/lichengunc/refer.git third_party/refer
   （不克隆也可以，src/grounding/refer_loader.py 有内置 fallback）

期望目录结构：
   data/refcoco/
   ├── refcoco/
   │   ├── refs(unc).p
   │   └── instances.json
   ├── refcoco+/
   │   ├── refs(unc).p
   │   └── instances.json
   └── refcocog/
       ├── refs(google).p
       └── instances.json
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载评测所需数据集")
    parser.add_argument(
        "--dataset",
        choices=["coco-val", "coco-train2014", "refcoco", "all"],
        default="all",
        help="要下载的数据集（all = coco-val + coco-train2014，refcoco 仅显示指南）",
    )
    parser.add_argument("--data-root", default="data", help="数据根目录")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.dataset in ("coco-val", "all"):
        download_coco_val(data_root / "coco")

    if args.dataset in ("coco-train2014", "all"):
        download_coco_train2014(data_root / "coco")

    if args.dataset in ("refcoco", "all"):
        show_refcoco_guide()

    print("\n=== 数据下载完成 ===")
    print("下一步：")
    print("  python scripts/download_weights.py")
    print("  python src/ovd/eval_coco.py --config configs/coco_ovd.yaml --subset 100")


if __name__ == "__main__":
    main()
