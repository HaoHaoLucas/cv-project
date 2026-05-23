"""RefCOCO / RefCOCO+ / RefCOCOg 数据加载器。

依赖 ``refer`` 工具（https://github.com/lichengunc/refer），
需在 ``third_party/refer`` 中 clone 并加入 Python 路径，
或直接用内置的纯 Python 实现（:class:`SimpleReferLoader`，无需安装 refer）。

对外暴露统一接口 :func:`load_refer_samples`，返回样本列表：
    [{"img_path": ..., "expr": ..., "gt_box_xyxy": [...], "ref_id": ..., "split": ...}, ...]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Literal

RefDataset = Literal["refcoco", "refcoco+", "refcocog"]
Split = Literal["val", "testA", "testB", "test"]


class SimpleReferLoader:
    """不依赖 refer 工具包的轻量加载器。

    直接读取 ``refs(*.p)`` 和 ``instances.json``，解析样本。
    支持 refcoco / refcoco+ / refcocog 三个数据集。
    """

    def __init__(self, refer_root: str, dataset: RefDataset, coco_img_dir: str) -> None:
        """
        Args:
            refer_root: 包含 refcoco/ refcoco+/ refcocog/ 的根目录。
            dataset: 数据集名称。
            coco_img_dir: COCO train2014 图像目录。
        """
        self.refer_root = Path(refer_root)
        self.dataset = dataset
        self.coco_img_dir = Path(coco_img_dir)
        self._refs: list[dict] = []
        self._annots: dict[int, dict] = {}  # ann_id → annotation
        self._images: dict[int, dict] = {}  # image_id → image info
        self._load()

    def _load(self) -> None:
        import json

        dataset_dir = self.refer_root / self.dataset
        # 确定 refs 文件名
        if self.dataset == "refcocog":
            refs_file = dataset_dir / "refs(google).p"
            if not refs_file.exists():
                refs_file = dataset_dir / "refs(umd).p"
        else:
            refs_file = dataset_dir / "refs(unc).p"

        with open(refs_file, "rb") as f:
            self._refs = pickle.load(f)

        instances_file = dataset_dir / "instances.json"
        with open(instances_file, encoding="utf-8") as f:
            instances = json.load(f)

        self._annots = {a["id"]: a for a in instances["annotations"]}
        self._images = {img["id"]: img for img in instances["images"]}

    def get_samples(self, split: Split) -> list[dict]:
        """返回指定 split 的样本列表。

        每个样本：
        ``{"img_path", "expr", "gt_box_xyxy", "ref_id", "ann_id", "split"}``
        """
        samples = []
        for ref in self._refs:
            if ref["split"] != split:
                continue

            ann_id = ref["ann_id"]
            ann = self._annots.get(ann_id)
            if ann is None:
                continue

            img_id = ref["image_id"]
            img_info = self._images.get(img_id)
            if img_info is None:
                continue

            img_path = self.coco_img_dir / img_info["file_name"]

            # COCO bbox 格式：[x, y, w, h] → 转 xyxy
            x, y, w, h = ann["bbox"]
            gt_box_xyxy = [x, y, x + w, y + h]

            for sent in ref["sentences"]:
                samples.append({
                    "img_path": str(img_path),
                    "expr": sent["sent"],
                    "gt_box_xyxy": gt_box_xyxy,
                    "ref_id": ref["ref_id"],
                    "ann_id": ann_id,
                    "split": split,
                })

        return samples


def load_refer_samples(
    refer_root: str,
    dataset: RefDataset,
    split: Split,
    coco_img_dir: str,
) -> list[dict]:
    """统一加载函数，优先尝试官方 refer 工具，失败则回退到 SimpleReferLoader。

    Args:
        refer_root: RefCOCO 数据根目录。
        dataset: ``"refcoco"``, ``"refcoco+"``, 或 ``"refcocog"``。
        split: ``"val"``, ``"testA"``, ``"testB"``。
        coco_img_dir: COCO train2014 图像目录。

    Returns:
        样本字典列表。
    """
    # 尝试使用官方 refer 工具
    refer_tool_path = Path("third_party/refer")
    if refer_tool_path.exists():
        sys.path.insert(0, str(refer_tool_path))

    try:
        from refer import REFER  # type: ignore  # noqa: F401
        return _load_via_refer_tool(refer_root, dataset, split, coco_img_dir)
    except ImportError:
        loader = SimpleReferLoader(refer_root, dataset, coco_img_dir)
        return loader.get_samples(split)


def _load_via_refer_tool(
    refer_root: str,
    dataset: RefDataset,
    split: Split,
    coco_img_dir: str,
) -> list[dict]:
    """使用官方 refer 工具加载样本。"""
    from refer import REFER  # type: ignore

    splitBy = "google" if dataset == "refcocog" else "unc"
    refer = REFER(refer_root, dataset, splitBy)

    ref_ids = refer.getRefIds(split=split)
    samples = []
    for ref_id in ref_ids:
        ref = refer.Refs[ref_id]
        ann = refer.Anns[ref["ann_id"]]
        img_info = refer.Imgs[ref["image_id"]]

        img_path = Path(coco_img_dir) / img_info["file_name"]
        x, y, w, h = ann["bbox"]
        gt_box_xyxy = [x, y, x + w, y + h]

        for sent in ref["sentences"]:
            samples.append({
                "img_path": str(img_path),
                "expr": sent["sent"],
                "gt_box_xyxy": gt_box_xyxy,
                "ref_id": ref_id,
                "ann_id": ref["ann_id"],
                "split": split,
            })

    return samples
