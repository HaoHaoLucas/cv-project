"""COCO 80 类 prompt 构建工具。

提供两种 prompt 模式：
* ``concat`` — 所有类别名拼为一条长 prompt（官方推荐，速度快）
* ``single`` — 每个类别单独一条（慢，但映射无歧义）

同时提供类别名 → COCO category_id 的映射字典构建函数。
"""
from __future__ import annotations

from pycocotools.coco import COCO  # type: ignore


# COCO 官方 80 类名称（与 annotations 文件保持一致）
COCO80_NAMES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def build_concat_prompt(names: list[str], template: str = "{name}", separator: str = ". ") -> str:
    """将类别名列表拼接为一条 prompt。

    例如：``"person. bicycle. car. ..."``

    Args:
        names: 类别名列表。
        template: 每个类别的格式模板，``{name}`` 占位符。
        separator: 类别之间的分隔符。

    Returns:
        拼接后的 prompt 字符串，末尾自动加句点。
    """
    parts = [template.format(name=n) for n in names]
    prompt = separator.join(parts)
    if not prompt.endswith("."):
        prompt += "."
    return prompt


def build_name_to_catid(coco: COCO) -> dict[str, int]:
    """构建 ``类别名 → COCO category_id`` 的映射字典。

    注意：COCO category_id 不连续（1–90，其中有空缺），需从 annotations 动态读取。
    """
    return {cat["name"]: cat["id"] for cat in coco.loadCats(coco.getCatIds())}


def match_phrase_to_catid(
    phrase: str,
    name2catid: dict[str, int],
    unmatched_log: list[str] | None = None,
) -> int | None:
    """将模型返回的短语匹配到最近的 COCO 类别 ID。

    匹配策略（按优先级）：
    1. 完全相等（小写化后）
    2. 类别名是 phrase 的子串
    3. phrase 是类别名的子串

    Args:
        phrase: 模型输出的短语，例如 ``"person"`` 或 ``"a car"``。
        name2catid: :func:`build_name_to_catid` 的输出。
        unmatched_log: 若不为 None，将未匹配的短语追加其中，便于调试。

    Returns:
        匹配到的 category_id；失败返回 ``None``。
    """
    phrase_lower = phrase.lower().strip()

    # 策略 1：完全匹配
    if phrase_lower in name2catid:
        return name2catid[phrase_lower]

    # 策略 2：类别名作为子串出现在 phrase 中
    for name, cat_id in name2catid.items():
        if name in phrase_lower:
            return cat_id

    # 策略 3：phrase 出现在类别名中（处理截断情况）
    for name, cat_id in name2catid.items():
        if phrase_lower in name:
            return cat_id

    if unmatched_log is not None:
        unmatched_log.append(phrase)
    return None
