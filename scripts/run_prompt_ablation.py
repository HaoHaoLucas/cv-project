"""Prompt 模板消融实验（OVD on COCO val 子集）。

对比三种 prompt 模板在 200 张随机子集上的 mAP / AP50：
  1. "{name}"              → 直接类别名（当前默认）
  2. "a {name}"            → 加不定冠词
  3. "a photo of a {name}" → CLIP 风格模板

结果保存到 results/coco/ablation_prompt.json 并打印表格。

用法：
    python scripts/run_prompt_ablation.py
    python scripts/run_prompt_ablation.py --n 100  # 更快的小规模验证
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
import tempfile

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm

from src.model_wrapper import GroundingDINOWrapper
from src.ovd.coco_prompt_builder import COCO80_NAMES, build_concat_prompt, build_name_to_catid, match_phrase_to_catid
from src.utils.box_ops import xyxy_to_xywh
from src.utils.io import load_yaml, save_json


TEMPLATES = [
    ("{name}",              "直接类别名"),
    ("a {name}",            "加不定冠词"),
    ("a photo of a {name}", "CLIP 风格"),
]


def eval_template(
    model: GroundingDINOWrapper,
    template: str,
    img_ids: list[int],
    coco_gt: COCO,
    img_dir: Path,
    name2catid: dict[str, int],
    box_thr: float,
    txt_thr: float,
) -> dict:
    prompt = build_concat_prompt(COCO80_NAMES, template=template)
    preds = []

    for img_id in tqdm(img_ids, desc=f"  [{template}]", leave=False):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            continue

        result = model.predict(pil_img, prompt, box_threshold=box_thr, text_threshold=txt_thr)
        if len(result.boxes_xyxy) == 0:
            continue

        boxes_xywh = xyxy_to_xywh(result.boxes_xyxy)
        for box, score, phrase in zip(boxes_xywh, result.scores, result.phrases):
            cat_id = match_phrase_to_catid(phrase, name2catid)
            if cat_id is None:
                continue
            preds.append({"image_id": img_id, "category_id": cat_id,
                          "bbox": box.tolist(), "score": float(score)})

    if not preds:
        return {"mAP": 0.0, "AP50": 0.0, "n_preds": 0}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(preds, f)
        tmp_path = f.name

    coco_dt = coco_gt.loadRes(tmp_path)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    return {
        "mAP":   float(ev.stats[0]),
        "AP50":  float(ev.stats[1]),
        "AP75":  float(ev.stats[2]),
        "n_preds": len(preds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/coco_ovd.yaml")
    parser.add_argument("--n", type=int, default=200, help="子集图像数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/coco/ablation_prompt.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    random.seed(args.seed)

    print("加载 COCO annotations ...")
    coco_gt = COCO(cfg["data"]["ann_file"])
    name2catid = build_name_to_catid(coco_gt)
    img_dir = Path(cfg["data"]["val_img_dir"])

    all_ids = sorted(coco_gt.getImgIds())
    img_ids = random.sample(all_ids, min(args.n, len(all_ids)))
    print(f"子集：{len(img_ids)} 张（seed={args.seed}）")

    box_thr = cfg["inference"]["box_threshold"]
    txt_thr = cfg["inference"]["text_threshold"]

    print("加载模型 ...")
    model = GroundingDINOWrapper.from_config(cfg)

    results = {}
    for tmpl, desc in TEMPLATES:
        print(f"\n=== 模板：{desc}  [{tmpl}] ===")
        m = eval_template(model, tmpl, img_ids, coco_gt, img_dir, name2catid, box_thr, txt_thr)
        results[tmpl] = {**m, "desc": desc}
        print(f"  mAP={m['mAP']*100:.1f}  AP50={m['AP50']*100:.1f}  n_preds={m['n_preds']}")

    print("\n" + "=" * 55)
    print(f"{'Prompt 模板':<30} {'mAP':>6} {'AP50':>6}")
    print("-" * 55)
    for tmpl, m in results.items():
        print(f"{m['desc']+' ('+tmpl+')':<30} {m['mAP']*100:>6.1f} {m['AP50']*100:>6.1f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, out_path)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
