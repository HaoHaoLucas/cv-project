"""可视化工具：在图像上绘制检测框和文本标注。"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def draw_boxes(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
    labels: Sequence[str] | None = None,
    scores: np.ndarray | None = None,
    gt_box_xyxy: np.ndarray | None = None,
    title: str = "",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """在图像上绘制预测框（蓝色）和可选的 GT 框（绿色）。

    Args:
        image: PIL RGB 图像。
        boxes_xyxy: shape (N, 4) 预测框，绝对像素 xyxy。
        labels: 每个框的文本标签。
        scores: 每个框的置信度分数。
        gt_box_xyxy: shape (4,) GT 框，绝对像素 xyxy（可选）。
        title: 图标题。
        save_path: 若指定则保存到文件。
        figsize: Matplotlib 图像尺寸。

    Returns:
        matplotlib Figure 对象。
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

        label_parts = []
        if labels is not None and i < len(labels):
            label_parts.append(labels[i])
        if scores is not None and i < len(scores):
            label_parts.append(f"{scores[i]:.2f}")
        if label_parts:
            ax.text(
                x1, max(y1 - 4, 0), " ".join(label_parts),
                fontsize=8, color="white",
                bbox={"facecolor": color, "alpha": 0.7, "pad": 1},
            )

    if gt_box_xyxy is not None:
        gx1, gy1, gx2, gy2 = gt_box_xyxy
        rect_gt = mpatches.Rectangle(
            (gx1, gy1), gx2 - gx1, gy2 - gy1,
            linewidth=2, edgecolor="lime", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect_gt)
        ax.text(gx1, gy1 - 4, "GT", fontsize=8, color="lime",
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1})

    ax.axis("off")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig


def visualize_ovd_sample(
    img_path: str,
    predictions: list[dict],
    coco_categories: dict[int, str],
    title: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """可视化 OVD 单张样本（COCO detection 格式预测）。

    Args:
        img_path: 图像路径。
        predictions: COCO detection 格式的预测列表（同一张图的多个框）。
        coco_categories: ``{cat_id: name}`` 字典。
        title: 图标题。
        save_path: 保存路径（可选）。
    """
    img = Image.open(img_path).convert("RGB")
    if not predictions:
        return draw_boxes(img, np.zeros((0, 4)), title=title, save_path=save_path)

    boxes_xywh = np.array([p["bbox"] for p in predictions])
    scores = np.array([p["score"] for p in predictions])
    labels = [coco_categories.get(p["category_id"], str(p["category_id"])) for p in predictions]

    # xywh → xyxy
    boxes_xyxy = boxes_xywh.copy()
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]

    return draw_boxes(img, boxes_xyxy, labels=labels, scores=scores,
                      title=title, save_path=save_path)


def visualize_vg_sample(
    sample: dict,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """可视化 VG 单个样本（eval_refcoco per_sample 格式）。

    Args:
        sample: 包含 ``img_path``, ``expr``, ``gt_box``, ``pred_box``, ``iou``, ``hit`` 的字典。
        save_path: 保存路径（可选）。
    """
    img = Image.open(sample["img_path"]).convert("RGB")

    pred_box = np.array([sample["pred_box"]]) if sample.get("pred_box") else np.zeros((0, 4))
    gt_box = np.array(sample["gt_box"]) if sample.get("gt_box") else None

    score = sample.get("pred_score")
    scores = np.array([score]) if score is not None and len(pred_box) > 0 else None

    hit_str = "HIT" if sample.get("hit") else "MISS"
    iou_str = f"IoU={sample.get('iou', 0):.3f}"
    title = f"[{hit_str} {iou_str}] {sample.get('expr', '')}"

    return draw_boxes(
        img, pred_box,
        labels=["pred"] * len(pred_box),
        scores=scores,
        gt_box_xyxy=gt_box,
        title=title,
        save_path=save_path,
    )
