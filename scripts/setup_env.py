"""验证 conda 环境是否正确配置，并给出下一步提示。

用法（在 cv 环境激活后执行）：
    python scripts/setup_env.py
"""
from __future__ import annotations

import sys

print("=" * 60)
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print()


def check(name: str, import_str: str, extra: str = "") -> bool:
    try:
        mod = __import__(import_str.split(".")[0])
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK] {name} {ver}  {extra}")
        return True
    except ImportError:
        print(f"  [FAIL] {name} 未安装  {extra}")
        return False


print("── 核心依赖 ──────────────────────────────────")
check("torch", "torch")

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    print(f"  CUDA available: {cuda_ok}  ({device_name})")
    if cuda_ok:
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  显存: {mem_gb:.1f} GB")
except Exception:
    pass

check("torchvision", "torchvision")
check("transformers", "transformers")
check("timm", "timm")

print()
print("── 评测工具 ──────────────────────────────────")
check("pycocotools", "pycocotools")
check("opencv-python", "cv2")
check("PIL", "PIL")
check("numpy", "numpy")
check("scipy", "scipy")

print()
print("── 可视化 ────────────────────────────────────")
check("matplotlib", "matplotlib")
check("supervision", "supervision")

print()
print("── Grounding DINO 后端 ───────────────────────")
gdino_ok = check("groundingdino (源码)", "groundingdino",
                 "→ 需执行: pip install -e third_party/GroundingDINO")
if not gdino_ok:
    print("  [INFO] 可使用 HuggingFace 后端 (backend: hf) 作为替代")

print()
print("=" * 60)
print("下一步：")
print("  1. python scripts/download_weights.py")
print("  2. python src/ovd/eval_coco.py --config configs/coco_ovd.yaml --subset 100")
print("  3. 打开 notebooks/01_demo.ipynb")
