"""边界框工具函数。

所有函数均使用 numpy；不依赖 torch，方便在纯 CPU 分析场景中使用。
坐标约定：
  xyxy → [x1, y1, x2, y2]（左上 + 右下，绝对像素）
  xywh → [x, y, w, h]（左上角 + 宽高，COCO 格式）
  cxcywh → [cx, cy, w, h]（中心点 + 宽高，归一化 [0,1]，Grounding DINO 输出格式）
"""
from __future__ import annotations

import numpy as np


def cxcywh_norm_to_xyxy(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """将归一化的 cxcywh 转换为绝对像素 xyxy。

    Args:
        boxes: shape (N, 4)，值域 [0, 1]
        img_w: 图像宽度（像素）
        img_h: 图像高度（像素）

    Returns:
        shape (N, 4) 的 xyxy 坐标（绝对像素）
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """xyxy → xywh（COCO predictions 格式）。"""
    boxes = np.asarray(boxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """计算两个 xyxy 框的 IoU（标量）。"""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def batch_iou(pred_boxes: np.ndarray, gt_box: np.ndarray) -> np.ndarray:
    """计算 N 个预测框与单个 GT 框的 IoU 向量。

    Args:
        pred_boxes: shape (N, 4) xyxy
        gt_box: shape (4,) xyxy

    Returns:
        shape (N,) IoU 值
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    gt_box = np.asarray(gt_box, dtype=np.float32)

    xa = np.maximum(pred_boxes[:, 0], gt_box[0])
    ya = np.maximum(pred_boxes[:, 1], gt_box[1])
    xb = np.minimum(pred_boxes[:, 2], gt_box[2])
    yb = np.minimum(pred_boxes[:, 3], gt_box[3])

    inter = np.maximum(0.0, xb - xa) * np.maximum(0.0, yb - ya)
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = area_pred + area_gt - inter
    return np.where(union > 0, inter / union, 0.0)
