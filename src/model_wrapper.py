"""统一推理接口，支持两种后端：

* ``gdino``  — 从 ``third_party/GroundingDINO`` 源码安装的原始实现
              （需要编译 MultiScaleDeformableAttention CUDA 算子）
* ``hf``     — HuggingFace ``transformers`` 实现
              （``IDEA-Research/grounding-dino-tiny``，纯 Python，无需编译）

对外统一暴露 :meth:`GroundingDINOWrapper.predict`，返回 (boxes_xyxy, scores, phrases)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from src.utils.box_ops import cxcywh_norm_to_xyxy
from src.utils.logger import get_logger

logger = get_logger(__name__)

Backend = Literal["gdino", "hf"]


@dataclass
class ModelConfig:
    backend: Backend = "gdino"
    config_path: str = "configs/GroundingDINO_SwinT_OGC.py"
    weights_path: str = "weights/groundingdino_swint_ogc.pth"
    hf_model_id: str = "IDEA-Research/grounding-dino-tiny"
    device: str = "cuda"
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    max_side: int = 800


@dataclass
class PredictResult:
    """单张图像的推理结果。"""
    boxes_xyxy: np.ndarray          # shape (N, 4) 绝对像素坐标
    scores: np.ndarray              # shape (N,)
    phrases: list[str]              # 长度 N，模型预测的短语
    img_w: int = 0
    img_h: int = 0
    extra: dict = field(default_factory=dict)


class GroundingDINOWrapper:
    """封装 Grounding DINO 推理，对外屏蔽后端差异。"""

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._processor = None
        self._load_model()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self.cfg.backend == "gdino":
            self._load_gdino()
        else:
            self._load_hf()

    def _load_gdino(self) -> None:
        try:
            from groundingdino.util.inference import load_model  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Grounding DINO 源码后端未安装。\n"
                "请执行：pip install -e third_party/GroundingDINO\n"
                "或将配置中 model.backend 改为 'hf'。"
            ) from e

        logger.info("加载 Grounding DINO（gdino 后端）: %s", self.cfg.weights_path)
        self._model = load_model(self.cfg.config_path, self.cfg.weights_path)
        self._model = self._model.to(self.cfg.device).eval()

    def _load_hf(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore
        logger.info("加载 Grounding DINO（HuggingFace 后端）: %s", self.cfg.hf_model_id)
        self._processor = AutoProcessor.from_pretrained(self.cfg.hf_model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.cfg.hf_model_id
        ).to(self.cfg.device).eval()

    # ------------------------------------------------------------------
    # 图像预处理（统一 resize）
    # ------------------------------------------------------------------

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """HF 后端：长边不超过 max_side。"""
        w, h = image.size
        scale = self.cfg.max_side / max(w, h)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
        return image

    def _resize_image_gdino(self, image: Image.Image) -> Image.Image:
        """gdino 官方风格：短边缩放到 max_side，长边不超过 1333。"""
        w, h = image.size
        scale = self.cfg.max_side / min(w, h)
        max_long = 1333
        if max(w * scale, h * scale) > max_long:
            scale = max_long / max(w, h)
        new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
        return image.resize((new_w, new_h), Image.BILINEAR)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def predict_raw(
        self,
        image: Image.Image,
        caption: str,
    ) -> tuple[Image.Image, dict, tuple[int, int]]:
        """返回 resize 后图像、模型原始 outputs、原图尺寸 (w,h)。"""
        import torchvision.transforms as T

        image = image.convert("RGB")
        orig_w, orig_h = image.size
        if self.cfg.backend == "gdino":
            img_resized = self._resize_image_gdino(image)
        else:
            img_resized = self._resize_image(image)

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(img_resized).unsqueeze(0).to(self.cfg.device)

        import torch
        if self.cfg.backend == "gdino":
            from groundingdino.util.inference import preprocess_caption  # type: ignore

            caption = preprocess_caption(caption)
        with torch.no_grad():
            outputs = self._model(img_tensor, captions=[caption])

        return img_resized, outputs, (orig_w, orig_h)

    def predict(
        self,
        image: Image.Image,
        caption: str,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> PredictResult:
        """对单张图像做推理。

        Args:
            image: PIL RGB 图像（原始分辨率）。
            caption: 文本 prompt，例如 ``"person. car. dog."``。
            box_threshold: 覆盖默认阈值。
            text_threshold: 覆盖默认阈值。

        Returns:
            :class:`PredictResult`，boxes 已转换为绝对像素 xyxy 坐标。
        """
        box_thr = box_threshold if box_threshold is not None else self.cfg.box_threshold
        txt_thr = text_threshold if text_threshold is not None else self.cfg.text_threshold

        image = image.convert("RGB")
        orig_w, orig_h = image.size
        if self.cfg.backend == "gdino":
            img_resized = self._resize_image_gdino(image)
        else:
            img_resized = self._resize_image(image)
        img_w, img_h = img_resized.size

        if self.cfg.backend == "gdino":
            result = self._predict_gdino(img_resized, caption, box_thr, txt_thr, img_w, img_h)
        else:
            result = self._predict_hf(img_resized, caption, box_thr, txt_thr, img_w, img_h)

        # COCO / RefCOCO GT 使用原图坐标，需将框从 resize 空间映射回去
        if (img_w, img_h) != (orig_w, orig_h) and len(result.boxes_xyxy) > 0:
            sx, sy = orig_w / img_w, orig_h / img_h
            result.boxes_xyxy[:, [0, 2]] *= sx
            result.boxes_xyxy[:, [1, 3]] *= sy
            result.img_w, result.img_h = orig_w, orig_h
        return result

    def _predict_gdino(
        self,
        image: Image.Image,
        caption: str,
        box_thr: float,
        txt_thr: float,
        img_w: int,
        img_h: int,
    ) -> PredictResult:
        import torch
        from groundingdino.util.inference import predict as gdino_predict  # type: ignore
        from groundingdino.util import box_ops as gdino_box_ops  # type: ignore
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # gdino predict() 内部会 image[None]，此处保持 [C,H,W]
        img_tensor = transform(image).to(self.cfg.device)

        with torch.no_grad():
            boxes, scores, phrases = gdino_predict(
                model=self._model,
                image=img_tensor,
                caption=caption,
                box_threshold=box_thr,
                text_threshold=txt_thr,
                device=self.cfg.device,
            )

        # boxes 是归一化 cxcywh，转绝对 xyxy
        boxes_np = boxes.cpu().numpy()
        boxes_xyxy = cxcywh_norm_to_xyxy(boxes_np, img_w, img_h)
        scores_np = scores.cpu().numpy()

        return PredictResult(
            boxes_xyxy=boxes_xyxy,
            scores=scores_np,
            phrases=phrases,
            img_w=img_w,
            img_h=img_h,
        )

    def _predict_hf(
        self,
        image: Image.Image,
        caption: str,
        box_thr: float,
        txt_thr: float,
        img_w: int,
        img_h: int,
    ) -> PredictResult:
        import torch

        # max_length=256 是 GroundingDINO 文本编码器的硬限制，长 prompt 需截断
        inputs = self._processor(
            images=image, text=caption, return_tensors="pt",
            truncation=True, max_length=256,
        )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_thr,
            text_threshold=txt_thr,
            target_sizes=[(img_h, img_w)],
        )[0]

        boxes_xyxy = results["boxes"].cpu().numpy()    # already xyxy absolute
        scores_np = results["scores"].cpu().numpy()
        phrases = results["labels"]

        return PredictResult(
            boxes_xyxy=boxes_xyxy,
            scores=scores_np,
            phrases=phrases,
            img_w=img_w,
            img_h=img_h,
        )

    # ------------------------------------------------------------------
    # 工厂方法（从 yaml 配置字典构建）
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg_dict: dict) -> "GroundingDINOWrapper":
        m = cfg_dict.get("model", cfg_dict)
        infer = cfg_dict.get("inference", {})
        model_cfg = ModelConfig(
            backend=m.get("backend", "gdino"),
            config_path=m.get("config_path", "configs/GroundingDINO_SwinT_OGC.py"),
            weights_path=m.get("weights_path", "weights/groundingdino_swint_ogc.pth"),
            hf_model_id=m.get("hf_model_id", "IDEA-Research/grounding-dino-tiny"),
            device=m.get("device", "cuda"),
            box_threshold=infer.get("box_threshold", 0.25),
            text_threshold=infer.get("text_threshold", 0.20),
            max_side=infer.get("max_side", 800),
        )
        return cls(model_cfg)
