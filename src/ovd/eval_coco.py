"""OVD 评测主入口 —— COCO val2017。

用法：
    python src/ovd/eval_coco.py --config configs/coco_ovd.yaml
    python src/ovd/eval_coco.py --config configs/coco_ovd.yaml --subset 1000
    python src/ovd/eval_coco.py --config configs/coco_ovd.yaml --template "a photo of a {name}"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.utils.codec  # noqa: F401 — 修复 Windows 终端 UTF-8 输出

import argparse
import random

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from src.model_wrapper import GroundingDINOWrapper
from src.ovd.coco_prompt_builder import (
    COCO80_NAMES,
    build_concat_prompt,
    build_name_to_catid,
    match_phrase_to_catid,
)
from src.ovd.official_postprocess import build_ovd_postprocess_context, postprocess_ovd_outputs
from src.utils.box_ops import xyxy_to_xywh
from src.utils.io import load_yaml, save_json
from src.utils.logger import get_logger

logger = get_logger("eval_coco")


# ---------------------------------------------------------------------------
# 核心评测函数
# ---------------------------------------------------------------------------

def run_eval(
    cfg: dict,
    subset: int | None = None,
    template: str | None = None,
) -> dict:
    """执行完整 OVD 评测流程。

    Args:
        cfg: 从 YAML 加载的配置字典。
        subset: 覆盖配置中的 subset 值（命令行参数优先）。
        template: 覆盖配置中的 prompt template。

    Returns:
        包含评测指标的字典。
    """
    # ── 配置解析 ──────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    prompt_cfg = cfg.get("prompt", {})
    out_cfg = cfg["output"]

    ann_file = data_cfg["ann_file"]
    img_dir = Path(data_cfg["val_img_dir"])
    n_subset = subset if subset is not None else data_cfg.get("subset")
    tmpl = template or prompt_cfg.get("template", "{name}")
    separator = prompt_cfg.get("separator", ". ")
    prompt_mode = prompt_cfg.get("mode", "concat_token")
    num_select = int(prompt_cfg.get("num_select", 300))
    log_unmatched = out_cfg.get("log_unmatched", True)

    # ── 数据加载 ──────────────────────────────────────────────────────────
    logger.info("加载 COCO annotations: %s", ann_file)
    coco_gt = COCO(ann_file)
    img_ids = sorted(coco_gt.getImgIds())

    if n_subset:
        random.seed(42)
        img_ids = random.sample(img_ids, min(int(n_subset), len(img_ids)))
        logger.info("子采样 %d 张图像（共 %d 张）", len(img_ids), len(coco_gt.getImgIds()))

    name2catid = build_name_to_catid(coco_gt)

    # ── Prompt 构建 ────────────────────────────────────────────────────────
    if prompt_mode in ("concat", "concat_token"):
        prompt = build_concat_prompt(COCO80_NAMES, template=tmpl, separator=separator)
        logger.info("Prompt mode=%s, 前 80 字符: %s...", prompt_mode, prompt[:80])
    else:
        # single 模式：每类单独推理，此处只预构建各类 prompt
        single_prompts = {name: tmpl.format(name=name) for name in COCO80_NAMES}

    # ── 模型加载 ──────────────────────────────────────────────────────────
    model = GroundingDINOWrapper.from_config(cfg)
    ovd_ctx = None
    if prompt_mode == "concat_token":
        ovd_ctx = build_ovd_postprocess_context(coco_gt, cat_names=list(name2catid.keys()))

    infer_cfg = cfg.get("inference", {})
    box_thr = float(infer_cfg.get("box_threshold", 0.25))
    txt_thr = float(infer_cfg.get("text_threshold", 0.20))

    # ── 推理循环 ──────────────────────────────────────────────────────────
    predictions: list[dict] = []
    unmatched_phrases: list[str] = []

    for img_id in tqdm(img_ids, desc="OVD 推理"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logger.warning("图像不存在，跳过: %s", img_path)
            continue

        if prompt_mode == "concat_token":
            try:
                _, outputs, (orig_w, orig_h) = model.predict_raw(pil_img, ovd_ctx.caption)
                boxes, scores, cat_ids = postprocess_ovd_outputs(
                    outputs,
                    ovd_ctx,
                    target_size=(orig_h, orig_w),
                    box_threshold=box_thr,
                    text_threshold=txt_thr,
                    num_select=num_select,
                )
                _append_predictions_token(predictions, img_id, boxes, scores, cat_ids)
            except RuntimeError as e:
                logger.warning("推理失败，跳过 img_id=%s: %s", img_id, e)
                continue
        elif prompt_mode == "concat":
            try:
                result = model.predict(pil_img, prompt)
            except RuntimeError as e:
                logger.warning("推理失败，跳过 img_id=%s: %s", img_id, e)
                continue
            _append_predictions(
                predictions, img_id, result,
                name2catid, unmatched_phrases if log_unmatched else None,
            )
        else:
            # single 模式：逐类推理
            for name, single_prompt in single_prompts.items():
                cat_id = name2catid.get(name)
                if cat_id is None:
                    continue
                result = model.predict(pil_img, single_prompt)
                _append_predictions_single(predictions, img_id, result, cat_id)

    if log_unmatched and unmatched_phrases:
        unique = sorted(set(unmatched_phrases))
        logger.warning("未匹配短语（共 %d 个）: %s", len(unique), unique[:30])

    # ── 保存预测结果 ──────────────────────────────────────────────────────
    pred_path = out_cfg["predictions_path"]
    save_json(predictions, pred_path)
    logger.info("预测结果已保存: %s（%d 条）", pred_path, len(predictions))

    # ── COCO 评测 ─────────────────────────────────────────────────────────
    metrics = {}
    if len(predictions) == 0:
        logger.warning("没有有效预测，跳过评测。")
    else:
        coco_dt = coco_gt.loadRes(pred_path)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = img_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        stat_names = [
            "mAP", "AP50", "AP75", "AP_s", "AP_m", "AP_l",
            "AR@1", "AR@10", "AR@100", "AR_s", "AR_m", "AR_l",
        ]
        metrics = {k: float(v) for k, v in zip(stat_names, evaluator.stats)}
        logger.info("mAP=%.4f  AP50=%.4f  AP75=%.4f", metrics["mAP"], metrics["AP50"], metrics["AP75"])

    metrics_path = out_cfg["metrics_path"]
    save_json(metrics, metrics_path)
    logger.info("指标已保存: %s", metrics_path)
    return metrics


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _append_predictions(
    predictions: list[dict],
    img_id: int,
    result,
    name2catid: dict[str, int],
    unmatched_log: list[str] | None,
) -> None:
    """将 concat 模式的推理结果转换为 COCO detection 格式并追加到列表。"""
    if len(result.boxes_xyxy) == 0:
        return

    boxes_xywh = xyxy_to_xywh(result.boxes_xyxy)
    for box, score, phrase in zip(boxes_xywh, result.scores, result.phrases):
        cat_id = match_phrase_to_catid(phrase, name2catid, unmatched_log)
        if cat_id is None:
            continue
        predictions.append({
            "image_id": img_id,
            "category_id": cat_id,
            "bbox": box.tolist(),
            "score": float(score),
        })


def _append_predictions_token(
    predictions: list[dict],
    img_id: int,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    cat_ids: np.ndarray,
) -> None:
    if len(boxes_xyxy) == 0:
        return
    boxes_xywh = xyxy_to_xywh(boxes_xyxy)
    for box, score, cat_id in zip(boxes_xywh, scores, cat_ids):
        predictions.append({
            "image_id": img_id,
            "category_id": int(cat_id),
            "bbox": box.tolist(),
            "score": float(score),
        })


def _append_predictions_single(
    predictions: list[dict],
    img_id: int,
    result,
    cat_id: int,
) -> None:
    """将 single 模式的推理结果追加（类别已确定）。"""
    if len(result.boxes_xyxy) == 0:
        return
    boxes_xywh = xyxy_to_xywh(result.boxes_xyxy)
    for box, score in zip(boxes_xywh, result.scores):
        predictions.append({
            "image_id": img_id,
            "category_id": cat_id,
            "bbox": box.tolist(),
            "score": float(score),
        })


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounding DINO — OVD on COCO val2017")
    parser.add_argument("--config", default="configs/coco_ovd.yaml", help="配置文件路径")
    parser.add_argument("--subset", type=int, default=None, help="子采样张数（覆盖配置）")
    parser.add_argument("--template", type=str, default=None,
                        help="prompt 模板，如 'a photo of a {name}'（覆盖配置）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.config)
    run_eval(cfg, subset=args.subset, template=args.template)
