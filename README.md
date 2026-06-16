# Grounding DINO 复现与评测

基于 [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)（Swin-T 主干）复现
**开放词表目标检测（OVD）** 与 **视觉定位（Visual Grounding）** 两类任务，
在 COCO val2017 和 RefCOCO/+/g 上进行零样本（zero-shot）评测。

---

## 环境搭建

### 1. 创建 conda 环境（推荐）

在 PowerShell 中，`&&` 无效，命令需分开执行：

```powershell
conda env create -f environment.yml
conda activate cv
```

**已验证可用的替代方案**（miniconda base，Python 3.10.13）：

```powershell
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

注意：`numpy` 须 `<2.0`（torch 2.1.2 基于 NumPy 1.x ABI），
`transformers` 须 `<5.0`（5.x 要求 PyTorch >= 2.4）。

### 2. 克隆 Grounding DINO

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
```

#### Windows 编译 CUDA 算子（需 VS 2019/2022 C++ Build Tools）

```bash
pip install -e third_party/GroundingDINO
```

#### 备用方案：HuggingFace 后端（无需编译）

若 CUDA 算子编译失败，在 `configs/coco_ovd.yaml` 和 `configs/refcoco.yaml` 中将
`model.backend` 改为 `hf`，代码将自动切换到 `transformers` 实现：

```yaml
model:
  backend: hf
  hf_model_id: IDEA-Research/grounding-dino-tiny
```

### 3. 下载预训练权重

```bash
python scripts/download_weights.py
```

或手动下载：

```bash
# Swin-T OGC 权重
wget -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# 对应配置文件
cp third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py configs/
```

---

## 数据集准备

### COCO val2017（OVD 评测）

```
data/coco/
├── val2017/               # ~1 GB
└── annotations/
    └── instances_val2017.json
```

下载：<https://cocodataset.org/#download>

### COCO train2014（RefCOCO 图像来源）

```
data/coco/
└── train2014/             # ~13 GB
```

### RefCOCO / RefCOCO+ / RefCOCOg

```
data/refcoco/
├── refcoco/    {refs(unc).p,    instances.json}
├── refcoco+/   {refs(unc).p,    instances.json}
└── refcocog/   {refs(google).p, instances.json}
```

参考：<https://github.com/lichengunc/refer>

---

## 运行评测

### OVD — COCO val2017

```bash
# 快速子集验证（1000 张）
python src/ovd/eval_coco.py --config configs/coco_ovd.yaml --subset 1000

# 完整评测（全部 5000 张）
python src/ovd/eval_coco.py --config configs/coco_ovd.yaml
```

结果保存至 `results/coco/predictions.json` 和 `results/coco/metrics.json`。

### Visual Grounding — RefCOCO/+/g

```bash
python src/grounding/eval_refcoco.py --config configs/refcoco.yaml --dataset refcoco
python src/grounding/eval_refcoco.py --config configs/refcoco.yaml --dataset refcoco+
python src/grounding/eval_refcoco.py --config configs/refcoco.yaml --dataset refcocog
```

结果保存至 `results/refcoco/`。

---

## 主要结果

**口径**：本仓库为零样本推理。VG 应与论文 **零样本 @1** 对比，而非微调列 89%。详见 [`docs/论文指标口径说明.md`](docs/论文指标口径说明.md)。

| 任务 | 数据集 / 划分 | 指标 | 本复现（HF 历史） | 论文零样本 | 论文微调 |
|------|--------------|------|-------------------|------------|----------|
| OVD  | COCO val2017 | mAP  | 43.8 | **48.4** | — |
| VG   | RefCOCO val  | Acc@0.5 | 22.47 | **50.41** | 89.19 |
| VG   | RefCOCO testB | Acc@0.5 | 28.97 | **43.21** | 85.89 |
| VG   | RefCOCO+ val | Acc@0.5 | 23.56 | **51.40** | 81.22 |
| VG   | RefCOCO+ testB | Acc@0.5 | 29.54 | **45.81** | 74.18 |
| VG   | RefCOCOg val | Acc@0.5 | 30.66 | **60.42** | 86.94 |

**云服务器 gdino 最佳完成（`results/exp_2026-05-23_ovd_aligned/`、`results/refcoco_gdino/`）**：

| 任务 | 数据集 / 划分 | 指标 | 本仓库 | 论文零样本 | 论文微调 |
|------|--------------|------|--------|------------|----------|
| OVD  | COCO val2017 | mAP  | **46.16** | 48.4 | — |
| VG   | RefCOCO val  | Acc@0.5 | **50.85** | 50.41 | 89.19 |
| VG   | RefCOCO testB | Acc@0.5 | **45.38** | 43.21 | 85.89 |
| VG   | RefCOCO+ val | Acc@0.5 | **51.67** | 51.40 | 81.22 |
| VG   | RefCOCO+ testB | Acc@0.5 | **46.45** | 45.81 | 74.18 |
| VG   | RefCOCOg val | Acc@0.5 | **60.95** | 60.42 | 86.94 |

*完整分析见 `reports/report.md` §0.2。*

### 高精度复现计划（OVD + VG）

| 环境 | 文档 |
|------|------|
| **云服务器 Ubuntu 22 + V100**（推荐） | **[docs/高精度复现计划_OVD_VG.md](docs/高精度复现计划_OVD_VG.md)** §0、§11 |
| WSL | 同上；算子编译见文档 §5 |

Cursor Plan 面板若看不到 `.cursor/plans`，直接打开上述 Markdown 或将 **§11** 的 Agent 提示粘贴到对话开头。

---

## Notebooks

| 文件 | 用途 |
|------|------|
| `notebooks/01_demo.ipynb` | 单图推理 demo，验证环境 |
| `notebooks/02_qualitative.ipynb` | 定性可视化（每数据集 10–20 张） |
| `notebooks/03_failure_analysis.ipynb` | 失败案例归类与 prompt 消融分析 |

---

## 项目结构

```
cv-project/
├── environment.yml          # conda 环境（推荐）
├── requirements.txt         # pip fallback
├── configs/
│   ├── coco_ovd.yaml        # OVD 评测配置
│   └── refcoco.yaml         # VG 评测配置
├── src/
│   ├── model_wrapper.py     # 统一推理接口（支持 gdino / hf 双后端）
│   ├── ovd/
│   │   ├── coco_prompt_builder.py
│   │   └── eval_coco.py
│   ├── grounding/
│   │   ├── refer_loader.py
│   │   └── eval_refcoco.py
│   ├── analysis/
│   │   ├── failure_miner.py
│   │   └── visualize.py
│   └── utils/
│       ├── io.py
│       ├── box_ops.py
│       └── logger.py
├── notebooks/
├── results/
├── reports/report.md
├── third_party/GroundingDINO/   # git clone（不入库）
├── weights/                     # 预训练权重（不入库）
└── data/                        # 数据集（不入库）
```
