"""Visual Grounding 评测主入口 —— RefCOCO / RefCOCO+ / RefCOCOg。

用法：
    python src/grounding/eval_refcoco.py --config configs/refcoco.yaml --dataset refcoco
    python src/grounding/eval_refcoco.py --config configs/refcoco.yaml --dataset refcoco+
    python src/grounding/eval_refcoco.py --config configs/refcoco.yaml --dataset refcocog
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.utils.codec  # noqa: F401

import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.grounding.refer_loader import RefDataset, Split, load_refer_samples
from src.model_wrapper import GroundingDINOWrapper
from src.utils.box_ops import batch_iou
from src.utils.io import load_yaml, save_json
from src.utils.logger import get_logger

logger = get_logger("eval_refcoco")


# ---------------------------------------------------------------------------
# 核心评测函数
# ---------------------------------------------------------------------------

def eval_split(
    model: GroundingDINOWrapper,
    samples: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """对单个 split 执行推理并计算 Acc@iou_threshold。

    Args:
        model: 已初始化的推理接口。
        samples: :func:`load_refer_samples` 返回的样本列表。
        iou_threshold: 命中阈值，默认 0.5。

    Returns:
        包含 ``acc``, ``n_hit``, ``n_total``, ``per_sample`` 的字典。
    """
    n_hit = 0
    per_sample: list[dict] = []

    for sample in tqdm(samples, desc="VG 推理", leave=False):
        img_path = sample["img_path"]
        expr = sample["expr"]
        gt_box = np.array(sample["gt_box_xyxy"], dtype=np.float32)

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logger.warning("图像不存在，跳过: %s", img_path)
            per_sample.append({**sample, "iou": 0.0, "hit": False, "pred_box": None})
            continue

        result = model.predict(pil_img, expr)

        if len(result.boxes_xyxy) == 0:
            per_sample.append({**sample, "iou": 0.0, "hit": False, "pred_box": None})
            continue

        # 取得分最高的框（scores 已按降序排列）
        ious = batch_iou(result.boxes_xyxy, gt_box)
        best_idx = int(np.argmax(result.scores))
        best_iou = float(ious[best_idx])
        hit = best_iou >= iou_threshold
        if hit:
            n_hit += 1

        per_sample.append({
            "ref_id": sample["ref_id"],
            "ann_id": sample["ann_id"],
            "expr": expr,
            "img_path": img_path,
            "gt_box": gt_box.tolist(),
            "pred_box": result.boxes_xyxy[best_idx].tolist(),
            "pred_score": float(result.scores[best_idx]),
            "iou": best_iou,
            "hit": hit,
        })

    n_total = len(per_sample)
    acc = n_hit / n_total if n_total > 0 else 0.0
    return {
        "acc": acc,
        "n_hit": n_hit,
        "n_total": n_total,
        "per_sample": per_sample,
    }


def run_eval(cfg: dict, dataset: RefDataset) -> dict:
    """执行指定数据集的全部 split 评测。

    Args:
        cfg: 从 YAML 加载的配置字典。
        dataset: 数据集名称。

    Returns:
        所有 split 的评测结果汇总。
    """
    data_cfg = cfg["data"]
    out_cfg = cfg["output"]

    refer_root = data_cfg["refer_root"]
    coco_img_dir = data_cfg["coco_img_dir"]
    splits_cfg: dict = data_cfg.get("splits", {})
    splits: list[Split] = splits_cfg.get(dataset, ["val"])

    model = GroundingDINOWrapper.from_config(cfg)

    all_results: dict[str, dict] = {}
    for split in splits:
        logger.info("评测 %s / %s ...", dataset, split)
        try:
            samples = load_refer_samples(refer_root, dataset, split, coco_img_dir)
        except Exception as e:
            logger.error("加载 %s/%s 失败: %s", dataset, split, e)
            continue

        logger.info("  共 %d 个表达式", len(samples))
        result = eval_split(model, samples)
        all_results[split] = result
        logger.info("  Acc@0.5 = %.4f  (%d / %d)", result["acc"], result["n_hit"], result["n_total"])

    # ── 保存结果 ──────────────────────────────────────────────────────────
    out_dir = Path(out_cfg.get("predictions_dir", "results/refcoco"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, res in all_results.items():
        pred_path = out_dir / f"{dataset}_{split}_predictions.json"
        # per_sample 可能很大，单独保存
        save_json(res["per_sample"], pred_path)
        logger.info("预测结果已保存: %s", pred_path)

    # 指标汇总（不含 per_sample）
    summary = {
        split: {k: v for k, v in res.items() if k != "per_sample"}
        for split, res in all_results.items()
    }
    metrics_path = Path(out_cfg.get("metrics_path", "results/refcoco/metrics.json"))
    # 合并到已有文件
    existing: dict = {}
    if metrics_path.exists():
        from src.utils.io import load_json
        existing = load_json(metrics_path)
    existing[dataset] = summary
    save_json(existing, metrics_path)
    logger.info("指标已保存: %s", metrics_path)

    return summary


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounding DINO — VG on RefCOCO/+/g")
    parser.add_argument("--config", default="configs/refcoco.yaml")
    parser.add_argument(
        "--dataset",
        choices=["refcoco", "refcoco+", "refcocog"],
        default="refcoco",
        help="要评测的数据集",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.config)
    run_eval(cfg, dataset=args.dataset)
