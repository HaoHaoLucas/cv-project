"""失败案例挖掘工具。

从评测结果（per_sample 列表）中自动筛选失败案例，
并按预定义的失败类型进行自动标注。

失败类型分类：
  1. small_obj   — 小目标（GT 框面积 < 32*32 像素）
  2. multi_inst  — 同类多实例混淆（同图存在多个同类标注）
  3. lang_ambig  — 语言歧义（expr 中缺少明确空间关系词）
  4. rare_word   — 罕见词（expr 包含非常见词汇列表中的词）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

FailureType = Literal["small_obj", "multi_inst", "lang_ambig", "rare_word", "unknown"]

# 空间关系词列表（用于 lang_ambig 判断）
SPATIAL_WORDS = {
    "left", "right", "top", "bottom", "above", "below", "behind", "front",
    "near", "far", "center", "middle", "corner", "inside", "outside",
    "between", "next to", "on top of", "in front of",
    "左", "右", "上", "下", "前", "后", "中间", "旁边",
}

# 粗略罕见词列表（可根据需要扩充）
RARE_WORDS = {
    "umpire", "referee", "goalie", "batter", "pitcher", "catcher",
    "parachutist", "rickshaw", "kibbutz", "tepee", "windmill",
}

SMALL_OBJ_THRESHOLD = 32 * 32  # 像素面积


# ---------------------------------------------------------------------------
# 主要 API
# ---------------------------------------------------------------------------

def mine_failures(
    per_sample: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """筛选失败案例并标注失败类型。

    Args:
        per_sample: :func:`eval_split` 输出的 ``per_sample`` 列表。
        iou_threshold: 低于此 IoU 视为失败。

    Returns:
        失败样本列表，每个样本新增 ``failure_types`` 字段（列表）。
    """
    failures = [s for s in per_sample if not s.get("hit", False)]
    for sample in failures:
        sample["failure_types"] = _classify(sample)
    return failures


def summarize_failures(failures: list[dict]) -> dict:
    """统计各失败类型数量。

    Returns:
        ``{failure_type: count}`` 字典。
    """
    from collections import Counter
    counter: Counter = Counter()
    for s in failures:
        for ft in s.get("failure_types", ["unknown"]):
            counter[ft] += 1
    return dict(counter)


def select_examples(
    failures: list[dict],
    failure_type: FailureType | None = None,
    n: int = 10,
    sort_by: Literal["iou_asc", "iou_desc", "score_desc"] = "iou_asc",
) -> list[dict]:
    """从失败样本中选取典型案例供可视化。

    Args:
        failures: :func:`mine_failures` 的输出。
        failure_type: 若指定，只选该类型的失败样本。
        n: 选取数量。
        sort_by: 排序方式。
            - ``iou_asc``    — IoU 最低的放前面（最典型失败）
            - ``iou_desc``   — IoU 最高但仍未命中（边界案例）
            - ``score_desc`` — 模型最自信但仍失败（高 score 低 IoU）

    Returns:
        最多 n 个样本。
    """
    pool = failures
    if failure_type is not None:
        pool = [s for s in failures if failure_type in s.get("failure_types", [])]

    if sort_by == "iou_asc":
        pool = sorted(pool, key=lambda s: s.get("iou", 0.0))
    elif sort_by == "iou_desc":
        pool = sorted(pool, key=lambda s: s.get("iou", 0.0), reverse=True)
    elif sort_by == "score_desc":
        pool = sorted(pool, key=lambda s: s.get("pred_score", 0.0), reverse=True)

    return pool[:n]


# ---------------------------------------------------------------------------
# 内部分类函数
# ---------------------------------------------------------------------------

def _classify(sample: dict) -> list[FailureType]:
    types: list[FailureType] = []

    gt_box = sample.get("gt_box") or sample.get("gt_box_xyxy")
    if gt_box:
        x1, y1, x2, y2 = gt_box
        area = (x2 - x1) * (y2 - y1)
        if area < SMALL_OBJ_THRESHOLD:
            types.append("small_obj")

    expr = (sample.get("expr") or "").lower()
    has_spatial = any(w in expr for w in SPATIAL_WORDS)
    if not has_spatial and len(expr.split()) >= 3:
        types.append("lang_ambig")

    words = set(expr.replace(",", " ").replace(".", " ").split())
    if words & RARE_WORDS:
        types.append("rare_word")

    if not types:
        types.append("unknown")

    return types


# ---------------------------------------------------------------------------
# 命令行入口（批量分析并保存报告）
# ---------------------------------------------------------------------------

def analyze_from_file(predictions_json: str, output_json: str) -> None:
    """从预测 JSON 文件加载并输出失败案例分析报告。"""
    with open(predictions_json, encoding="utf-8") as f:
        per_sample = json.load(f)

    failures = mine_failures(per_sample)
    summary = summarize_failures(failures)

    report = {
        "total": len(per_sample),
        "n_failures": len(failures),
        "failure_rate": len(failures) / max(len(per_sample), 1),
        "failure_type_counts": summary,
        "failures": failures,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"失败案例分析完成：{len(failures)}/{len(per_sample)} 条失败")
    print("失败类型分布：", summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_json", help="per_sample 预测文件路径")
    parser.add_argument("--output", default="results/refcoco/failure_analysis.json")
    args = parser.parse_args()
    analyze_from_file(args.predictions_json, args.output)
