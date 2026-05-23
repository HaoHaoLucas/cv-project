"""生成 OVD 定性可视化图。

从 results/coco/predictions.json 读取预测结果，
随机抽取图像并绘制检测框，保存为 PNG 文件。

用法：
    python scripts/run_qualitative.py
    python scripts/run_qualitative.py --n 20 --top-k 8
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.utils.codec  # noqa: F401

import argparse
import json
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")   # 无 GUI 模式，适合服务器/脚本
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


def draw_grid(img_ids, preds_by_img, coco_gt, img_dir, coco_cats, top_k=5, cols=4):
    rows = (len(img_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes_flat = axes.flat if rows > 1 else [axes] if cols == 1 else list(axes.flat)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, img_id in zip(axes_flat, img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            ax.set_visible(False)
            continue

        preds = sorted(preds_by_img[img_id], key=lambda x: x["score"], reverse=True)[:top_k]
        ax.imshow(pil_img)
        ax.axis("off")
        ax.set_title(f"id={img_id}  n={len(preds_by_img[img_id])}", fontsize=7)

        for i, pred in enumerate(preds):
            x, y, w, h = pred["bbox"]
            color = colors[i % len(colors)]
            ax.add_patch(mpatches.Rectangle(
                (x, y), w, h, lw=1.5, edgecolor=color, facecolor="none"
            ))
            label = coco_cats.get(pred["category_id"], "?")
            ax.text(x, max(y - 3, 0), f"{label} {pred['score']:.2f}",
                    fontsize=6, color="white",
                    bbox={"facecolor": color, "alpha": 0.75, "pad": 1})

    # 隐藏多余子图
    for ax in list(axes_flat)[len(img_ids):]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="results/coco/predictions.json")
    parser.add_argument("--ann", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--img-dir", default="data/coco/val2017")
    parser.add_argument("--out", default="results/coco/qualitative_ovd.png")
    parser.add_argument("--n", type=int, default=12, help="展示图像数量")
    parser.add_argument("--top-k", type=int, default=5, help="每张图显示前 k 个框")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("加载 COCO annotations ...")
    coco_gt = COCO(args.ann)
    coco_cats = {cat["id"]: cat["name"] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    img_dir = Path(args.img_dir)

    print("读取预测文件 ...")
    with open(args.pred, encoding="utf-8") as f:
        preds = json.load(f)

    preds_by_img = defaultdict(list)
    for p in preds:
        preds_by_img[p["image_id"]].append(p)

    # 优先选有多个检测框的图像（更有代表性）
    candidates = [iid for iid, ps in preds_by_img.items() if len(ps) >= 3]
    if len(candidates) < args.n:
        candidates = list(preds_by_img.keys())

    sample_ids = random.sample(candidates, min(args.n, len(candidates)))
    print(f"抽取 {len(sample_ids)} 张图像 ...")

    fig = draw_grid(sample_ids, preds_by_img, coco_gt, img_dir, coco_cats, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path}")

    # 额外生成一张"高分样本精选"（score 均值最高的图）
    top_imgs = sorted(
        [(iid, np.mean([p["score"] for p in ps])) for iid, ps in preds_by_img.items()],
        key=lambda x: -x[1]
    )[:args.n]
    top_ids = [iid for iid, _ in top_imgs]
    fig2 = draw_grid(top_ids, preds_by_img, coco_gt, img_dir, coco_cats, top_k=args.top_k)
    out2 = out_path.with_name("qualitative_ovd_highconf.png")
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"已保存（高置信度样本）: {out2}")


if __name__ == "__main__":
    main()
