"""VG 语义对齐选框：优先 token 对齐分数，而非单纯 box score。"""
from __future__ import annotations

import numpy as np
import torch
from groundingdino.util import box_ops as gdino_box_ops  # type: ignore
from groundingdino.util.vl_utils import create_positive_map_from_span  # type: ignore


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if not result.endswith("."):
        result += "."
    return result


def _get_tokenizer(model) -> object:
    return model.tokenizer


def _expression_token_span(caption: str) -> list[list[int]]:
    """整句表达式的字符 span（去掉句尾句号，end 为开区间）。"""
    text = caption.rstrip(".")
    if not text:
        return []
    return [[0, len(text)]]


def select_box_semantic(
    outputs: dict,
    caption: str,
    tokenizer: object,
    target_size: tuple[int, int],
    box_threshold: float = 0.10,
    text_threshold: float = 0.05,
    top_k: int = 10,
    combine_with_box_score: bool = True,
) -> tuple[np.ndarray, float, dict]:
    """从原始输出中选取与表达式语义最对齐的框。

    Returns:
        box_xyxy (4,), selection_score, debug dict
    """
    w, h = target_size
    caption = preprocess_caption(caption)
    out_logits = outputs["pred_logits"].sigmoid()[0]  # [nq, 256]
    out_boxes = outputs["pred_boxes"][0]  # [nq, 4] cxcywh norm

    tokenized = tokenizer(caption)
    span = _expression_token_span(caption)
    if not span:
        return np.zeros(4, dtype=np.float32), 0.0, {"reason": "empty_caption"}
    try:
        pos_map = create_positive_map_from_span(tokenized, [span]).to(out_logits.device)
    except (OverflowError, ValueError):
        return np.zeros(4, dtype=np.float32), 0.0, {"reason": "token_span_failed"}

    align = (out_logits @ pos_map.T).squeeze(-1)  # [nq]
    box_conf = out_logits.max(dim=1)[0]

    keep = box_conf > box_threshold
    if keep.sum() == 0:
        keep = align > text_threshold
    if keep.sum() == 0:
        return np.zeros(4, dtype=np.float32), 0.0, {"reason": "no_candidates"}

    align_k = align[keep]
    box_k = box_conf[keep]
    boxes_k = out_boxes[keep]

    if combine_with_box_score:
        selection = align_k * box_k
    else:
        selection = align_k

    k = min(int(top_k), len(selection))
    order = torch.argsort(selection, descending=True)[:k]
    best_local = int(order[0])
    best_box = boxes_k[best_local]
    best_score = float(selection[best_local].item())

    boxes_xyxy = gdino_box_ops.box_cxcywh_to_xyxy(best_box.unsqueeze(0))
    scale = torch.tensor([w, h, w, h], device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
    boxes_xyxy = boxes_xyxy * scale
    box_np = boxes_xyxy[0].cpu().numpy().astype(np.float32)

    return box_np, best_score, {
        "align": float(align_k[best_local].item()),
        "box_conf": float(box_k[best_local].item()),
        "num_candidates": int(keep.sum().item()),
    }


def select_box_argmax_score(
    result_boxes: np.ndarray,
    result_scores: np.ndarray,
) -> tuple[int, np.ndarray]:
    """旧策略：最高分框。"""
    if len(result_scores) == 0:
        return -1, np.zeros(4, dtype=np.float32)
    idx = int(np.argmax(result_scores))
    return idx, result_boxes[idx]
