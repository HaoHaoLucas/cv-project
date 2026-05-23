"""Visual Grounding 评测脚本（使用 HuggingFace parquet 格式标注）。

从 data/refcoco_hf/ 下的 parquet 文件加载标注，
对每条表达式预测 top-1 框并与 GT 框计算 IoU，
输出 Acc@0.5 指标。

用法：
    python scripts/run_vg_eval.py --dataset refcoco --split validation
    python scripts/run_vg_eval.py --dataset refcocog --split validation
    python scripts/run_vg_eval.py --all   # 跑所有数据集所有 split
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.utils.codec  # noqa: F401

import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.model_wrapper import GroundingDINOWrapper
from src.utils.box_ops import batch_iou
from src.vg.box_selection import select_box_argmax_score, select_box_semantic
from src.utils.io import load_yaml, save_json
from src.utils.logger import get_logger

logger = get_logger("vg_eval")

DATASET_SPLITS = {
    "refcoco":  ["validation", "testB"],
    "refcoco+": ["validation", "testB"],
    "refcocog": ["validation"],
}


def load_parquet_samples(parquet_path: Path, coco_img_dir: Path) -> list[dict]:
    """从 parquet 文件加载 VG 样本。

    每行对应一个 reference（可含多个 sentences）。
    返回展开后的逐 sentence 样本列表。
    """
    df = pd.read_parquet(parquet_path)
    samples = []

    for _, row in df.iterrows():
        # file_name 列有额外后缀（如 _4.jpg），需从 raw_image_info 读取真实文件名
        raw_info = json.loads(row["raw_image_info"])
        img_file = raw_info["file_name"]          # e.g. "COCO_train2014_000000580957.jpg"
        img_path = coco_img_dir / img_file

        # bbox 字段：[x1, y1, x2, y2] (xyxy 格式)
        bbox = row["bbox"]
        gt_box = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

        # captions 字段：表达式列表
        captions = row["captions"] if hasattr(row["captions"], "__iter__") else [row["captions"]]
        for expr in captions:
            if isinstance(expr, str) and expr.strip():
                samples.append({
                    "img_path": str(img_path),
                    "expr": expr.strip(),
                    "gt_box_xyxy": gt_box,
                    "ref_id": int(row["ref_id"]),
                    "ann_id": int(row["ann_id"]),
                })

    return samples


def eval_one_split(
    model: GroundingDINOWrapper,
    samples: list[dict],
    infer_cfg: dict | None = None,
    iou_threshold: float = 0.5,
    max_samples: int | None = None,
) -> dict:
    """对单个 split 计算 Acc@0.5。"""
    if max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, min(max_samples, len(samples)))

    infer_cfg = infer_cfg or {}
    selection = infer_cfg.get("selection", "semantic")
    top_k = int(infer_cfg.get("top_k", 10))
    box_thr = float(infer_cfg.get("box_threshold", model.cfg.box_threshold))
    txt_thr = float(infer_cfg.get("text_threshold", model.cfg.text_threshold))

    n_hit = 0
    per_sample = []

    for sample in tqdm(samples, desc="  推理", leave=False):
        img_path = sample["img_path"]
        expr = sample["expr"]
        gt_box = np.array(sample["gt_box_xyxy"], dtype=np.float32)

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            per_sample.append({**sample, "iou": 0.0, "hit": False, "pred_box": None})
            continue

        if selection == "semantic" and model.cfg.backend == "gdino":
            try:
                _, outputs, (orig_w, orig_h) = model.predict_raw(pil_img, expr)
            except RuntimeError:
                per_sample.append({**sample, "iou": 0.0, "hit": False, "pred_box": None})
                continue
            pred_box, pred_score, dbg = select_box_semantic(
                outputs,
                expr,
                model._model.tokenizer,
                (orig_w, orig_h),
                box_threshold=box_thr,
                text_threshold=txt_thr,
                top_k=top_k,
            )
            if pred_score <= 0:
                per_sample.append({**sample, "iou": 0.0, "hit": False, "pred_box": None})
                continue
            best_iou = float(batch_iou(pred_box[None], gt_box)[0])
        else:
            result = model.predict(pil_img, expr, box_threshold=box_thr, text_threshold=txt_thr)
            if len(result.boxes_xyxy) == 0:
                per_sample.append({**sample, "iou": 0.0, "hit": False, "pred_box": None})
                continue
            ious = batch_iou(result.boxes_xyxy, gt_box)
            best_idx, pred_box = select_box_argmax_score(result.boxes_xyxy, result.scores)
            best_iou = float(ious[best_idx])
            pred_score = float(result.scores[best_idx])

        hit = best_iou >= iou_threshold
        if hit:
            n_hit += 1

        per_sample.append({
            "ref_id": sample["ref_id"],
            "ann_id": sample["ann_id"],
            "expr": expr,
            "img_path": img_path,
            "gt_box": gt_box.tolist(),
            "pred_box": pred_box.tolist(),
            "pred_score": pred_score,
            "iou": best_iou,
            "hit": hit,
            "selection": selection,
        })

    n_total = len(per_sample)
    acc = n_hit / n_total if n_total > 0 else 0.0
    return {"acc": acc, "n_hit": n_hit, "n_total": n_total, "per_sample": per_sample}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/refcoco.yaml")
    parser.add_argument("--dataset", default="refcoco",
                        choices=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--all", action="store_true", help="跑所有数据集所有 split")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="每个 split 最多评测样本数（调试用）")
    parser.add_argument("--hf-dir", default="data/refcoco_hf",
                        help="parquet 文件根目录")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    coco_img_dir = Path(cfg["data"]["coco_img_dir"])
    out_dir = Path(cfg["output"].get("predictions_dir", "results/refcoco"))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(cfg["output"].get("metrics_path", "results/refcoco/metrics.json"))

    logger.info("加载模型 ...")
    model = GroundingDINOWrapper.from_config(cfg)

    tasks = []
    if args.all:
        for ds, splits in DATASET_SPLITS.items():
            for sp in splits:
                tasks.append((ds, sp))
    else:
        tasks = [(args.dataset, args.split)]

    all_metrics: dict = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            all_metrics = json.load(f)

    for dataset, split in tasks:
        parquet_path = Path(args.hf_dir) / dataset / f"{split}.parquet"
        if not parquet_path.exists():
            logger.warning("未找到: %s，跳过", parquet_path)
            continue

        logger.info("加载标注: %s / %s (%s)", dataset, split, parquet_path)
        samples = load_parquet_samples(parquet_path, coco_img_dir)
        logger.info("  共 %d 个表达式", len(samples))

        result = eval_one_split(
            model, samples, infer_cfg=cfg.get("inference", {}), max_samples=args.max_samples
        )
        logger.info("  Acc@0.5 = %.4f  (%d / %d)", result["acc"], result["n_hit"], result["n_total"])

        # 保存 per-sample 预测
        pred_path = out_dir / f"{dataset}_{split}_predictions.json"
        save_json(result["per_sample"], pred_path)

        # 更新汇总指标（每个 split 完成后立刻写盘，防止中途崩溃丢数据）
        if dataset not in all_metrics:
            all_metrics[dataset] = {}
        all_metrics[dataset][split] = {
            "acc": result["acc"], "n_hit": result["n_hit"], "n_total": result["n_total"]
        }
        save_json(all_metrics, metrics_path)
        logger.info("指标已写盘: %s / %s", dataset, split)

    logger.info("所有指标已保存: %s", metrics_path)

    print("\n=== VG 评测汇总 ===")
    for ds, splits in all_metrics.items():
        for sp, m in splits.items():
            print(f"  {ds}/{sp:12s}: Acc@0.5 = {m['acc']*100:.2f}%  ({m['n_hit']}/{m['n_total']})")


if __name__ == "__main__":
    main()
