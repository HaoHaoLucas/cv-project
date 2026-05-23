"""官方风格 OVD 后处理：token span + positive map。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pycocotools.coco import COCO  # type: ignore

from groundingdino.util import box_ops as gdino_box_ops  # type: ignore
from groundingdino.util.vl_utils import (  # type: ignore
    build_captions_and_token_span,
    create_positive_map_from_span,
)


@dataclass
class OvdPostprocessContext:
    caption: str
    cat_names: list[str]
    cat_ids: list[int]
    positive_map: torch.Tensor  # [num_cats, 256]
    tokenizer: object


def build_ovd_postprocess_context(
    coco: COCO,
    cat_names: list[str] | None = None,
) -> OvdPostprocessContext:
    """构建与 test_ap_on_coco.py 一致的类别文本与 positive map。"""
    categories = coco.loadCats(coco.getCatIds())
    categories = sorted(categories, key=lambda x: x["id"])
    cat_ids = [c["id"] for c in categories]
    names = cat_names or [c["name"] for c in categories]

    caption, cat2tokenspan = build_captions_and_token_span(names, force_lowercase=True)
    tokenizer = _get_tokenizer()
    tokenized = tokenizer(caption)
    tokenspanlist = [cat2tokenspan[name.lower()] for name in names]
    positive_map = create_positive_map_from_span(tokenized, tokenspanlist)

    return OvdPostprocessContext(
        caption=caption,
        cat_names=names,
        cat_ids=cat_ids,
        positive_map=positive_map,
        tokenizer=tokenizer,
    )


def _get_tokenizer():
    from groundingdino.util import get_tokenlizer  # type: ignore

    return get_tokenlizer.get_tokenlizer("bert-base-uncased")


def postprocess_ovd_outputs(
    outputs: dict,
    ctx: OvdPostprocessContext,
    target_size: tuple[int, int],
    box_threshold: float = 0.25,
    text_threshold: float = 0.05,
    num_select: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将模型原始输出转为 (boxes_xyxy, scores, category_ids)。"""
    h, w = target_size
    out_logits = outputs["pred_logits"].sigmoid()[0]  # [nq, 256]
    out_boxes = outputs["pred_boxes"][0]  # [nq, 4] cxcywh norm

    pos_map = ctx.positive_map.to(out_logits.device)
    prob_to_label = out_logits @ pos_map.T  # [nq, num_cats]

    max_logits = prob_to_label.max(dim=1)[0]
    keep = max_logits > box_threshold
    if keep.sum() == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return (
            np.zeros((0, 4), dtype=np.float32),
            empty,
            empty.astype(np.int64),
        )

    prob_kept = prob_to_label[keep]
    boxes_kept = out_boxes[keep]

    # 每个 query 取 top-1 类别（与官方 topk 后处理一致的核心映射）
    cls_scores, cls_ids = prob_kept.max(dim=1)
    text_mask = out_logits[keep] > text_threshold
    # 若 text mask 全 False，仍保留 box 分数
    final_scores = cls_scores

    boxes_xyxy = gdino_box_ops.box_cxcywh_to_xyxy(boxes_kept)
    scale = torch.tensor([w, h, w, h], device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
    boxes_xyxy = boxes_xyxy * scale

    # NMS per class would be ideal; for now keep top num_select by score
    order = torch.argsort(final_scores, descending=True)
    if len(order) > num_select:
        order = order[:num_select]

    boxes_np = boxes_xyxy[order].cpu().numpy()
    scores_np = final_scores[order].cpu().numpy()
    cat_ids_np = cls_ids[order].cpu().numpy()
    cat_ids_np = np.array([ctx.cat_ids[int(i)] for i in cat_ids_np], dtype=np.int64)

    return boxes_np, scores_np, cat_ids_np
