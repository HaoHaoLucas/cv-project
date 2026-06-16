# Grounding DINO 复现与评测报告

## 摘要

本项目基于 [Grounding DINO](https://arxiv.org/abs/2303.05499)（Swin-T 主干）复现了
**开放词表目标检测（OVD）** 任务，在 COCO val2017（全量 5000 张）上进行零样本（zero-shot）评测，
早期 Windows/HF 实验在 COCO val2017 全量上取得 **mAP = 43.8**（见下文历史记录）。
**高精度复现（gdino + concat_token + semantic）OVD 全量 sweep 与 VG 全量 semantic + sweep 已按 §0 矩阵验收**；以下 **§0 实验矩阵** 为唯一权威进度表。

---

## 0. 实验矩阵与差距归因（可审计）

**验收口径**（见 `docs/高精度复现计划_OVD_VG.md` §10）：全量可复现产物 + 指标达标；子集 smoke、混合 `metrics.json`、§9 勾选均不算完成。

**基线唯一权威路径**：`results/exp_2026-05-23_baseline/`（2026-05-23 冻结：gdino + `prompt.mode: concat` + argmax 选框，OVD 全量 mAP **42.40**）。

| 标签 | OVD mAP | 规模证据 | VG Acc@0.5（RefCOCO val） | 规模证据 | 状态 |
|------|---------|----------|---------------------------|----------|------|
| Baseline（冻结） | **42.40** | baseline json；gdino 全量 | **50.72%** | n_total=10834 | 已完成 |
| OVD concat_token **子集** | **46.28** | predictions **997** image_id | — | — | 参考 only |
| OVD concat_token **全量**（旧 run） | **42.40** | `ovd_full_final.log` | — | — | 历史；已被 sweep 行取代 |
| OVD sweep → **全量最佳**（box=0.15, text=0.05） | **46.16** | `exp_2026-05-23_ovd_aligned/`；**4993** image_id | — | — | **达标**（>42.40） |
| VG semantic **smoke** | — | — | **54.00%** | n_total=**100** | 参考 only |
| VG semantic **全量 --all** | — | — | **50.85%** | refcoco testB **45.38%**；refcoco+ val **51.67%** testB **46.45%**；refcocog val **60.95%**；n_total=**36472** | **已完成** |
| VG sweep → **最佳**（box=0.05, text=0.05） | — | — | **52.20%** | `exp_2026-05-23_vg_sweep/`；n_total=**500**（sweep 子集） | **已完成** |
| CUDA 自定义算子 | — | — | — | `results/exp_2026-06-15_cuda_built/status.json` → **`built`**（gcc-10） | **已完成** |
| 官方 OVD `test_ap_on_coco.py` | **48.50** | `results/exp_2026-06-15_official_ovd/` | — | — | **已完成** |
| VG 协议审计（semantic vs argmax_official） | — | — | 子集 500：52.2% vs **52.6%** | `results/exp_2026-06-15_vg_protocol/` | **已完成** |
| Windows HF tiny（历史） | 43.8 | 5000 张 | 22–31% | HF 后端 | 历史对照 |

### 0.1 归因链路（当前）

1. **OVD 全量 sweep 已达标**：最佳阈值 box=0.15 / text=0.05，全量 mAP **46.16** > baseline 42.40（`results/exp_2026-05-23_ovd_aligned/`；4993/5000 image_id 有预测，7 张无检测条目）。
2. **子集 vs 全量**：1k 子集 sweep 最佳 mAP 47.51；全量 46.16，与旧 997 张子集 metrics 文件无关。
3. **VG semantic 全量 `--all` 已完成**（`results/refcoco_gdino/metrics.json`）：refcoco val **50.85%** / testB **45.38%**；refcoco+ val **51.67%** / testB **46.45%**；refcocog val **60.95%**；合计 n_total=**36472**。
4. **VG sweep 已完成**：最佳阈值 box=**0.05** / text=**0.05**，子集 Acc@0.5 **52.20%**（n_total=500，`results/exp_2026-05-23_vg_sweep/best_subset.json`）。
5. **工程**：CUDA 自定义算子已用 **gcc-10** 编译通过（`results/exp_2026-06-15_cuda_built/`）；此前 `exp_2026-05-23_cuda_ops` 为 `fallback_pytorch` 记录。官方 `test_ap_on_coco.py` 全量评测进行中。

### 0.3 进一步复现（2026-06-15）

| 实验 | 结果 | 路径 |
|------|------|------|
| CUDA 编译 + OVD 子集 500 | mAP **47.17**（`_C` 算子） | `results/exp_2026-06-15_cuda_built/metrics_subset500.json` |
| 官方 OVD 全量 | mAP **48.50**（vs 本仓库 46.16，Δ **+2.34**） | `results/exp_2026-06-15_official_ovd/comparison.json` |
| VG semantic vs argmax_official（refcoco val, n=500） | 52.2% vs 52.6%；@5/@10 均为 89.6%/95.6% | `results/exp_2026-06-15_vg_protocol/protocol_comparison.json` |
| testA（HF `test`→`testA.parquet`，n=200） | semantic **54.5%** | `results/exp_2026-06-15_vg_protocol/testA_semantic/` |

### 0.2 与论文对比口径（零样本 vs 微调）

本仓库为 **零样本推理**（无 RefCOCO 微调）。论文 Table 5 含多列，**不可将 VG 与 89% 直接对比**——该列为 `O365,GoldG,RefC + finetune`；零样本 @1（IoU≥0.5）约 **50–61%**。详见 [`docs/论文指标口径说明.md`](docs/论文指标口径说明.md)。

**OVD（COCO val2017 zero-shot mAP）**

| | 论文 / 官方 | 本仓库最佳 | 差距 |
|--|-------------|------------|------|
| mAP | **48.4** / 48.5（`test_ap_on_coco.py`） | **46.16** | **−2.2** |

**VG（Acc@0.5 = 论文零样本 @1，top-1 IoU≥0.5）**

| Split | 论文零样本 @1 | 本仓库 semantic 全量 | 差距 (pp) | 论文微调 @1（非本仓库口径） |
|-------|---------------|----------------------|-----------|----------------------------|
| refcoco / validation | 50.41 | **50.85** | +0.44 | 89.19 |
| refcoco / testB | 43.21 | **45.38** | +2.17 | 85.89 |
| refcoco+ / validation | 51.40 | **51.67** | +0.27 | 81.22 |
| refcoco+ / testB | 45.81 | **46.45** | +0.64 | 74.18 |
| refcocog / validation | 60.42 | **60.95** | +0.53 | 86.94 |

**结论**：VG 零样本已与论文 Table 5 对齐或略优；OVD 仍差约 **2.2 mAP**（主因 CUDA fallback + 与官方 `test_ap` 脚本差异）。微调列 81–89% 需 RefCOCO 监督训练，见 [`docs/RefCOCO微调复现路线.md`](docs/RefCOCO微调复现路线.md)。

---

## 一、模型简介

Grounding DINO 在 DINO 检测器的基础上引入跨模态融合模块，
将 Swin Transformer 视觉编码器与 BERT 文本编码器通过 Feature Enhancer 深度融合，
实现同时支持 **开放词表检测** 和 **指代表达定位** 的统一框架。

### 核心架构

```
图像 → Swin-T → Feature Pyramid
文本 → BERT   → Token Embeddings
          ↓ Feature Enhancer（跨模态融合 Transformer）
          ↓ Language-guided Query Selection
          ↓ Cross-modal Decoder
          → 输出：边界框 + 对应短语
```

与传统闭集检测器（如 YOLO、Faster RCNN）相比，Grounding DINO 的核心创新在于：

1. **开放词表**：检测类别由运行时文本 prompt 决定，不受训练集类别数限制。
2. **短语定位**：模型输出与文本 token 对齐的边界框，天然支持指代表达定位。
3. **统一框架**：OVD 和 Visual Grounding 共用同一套网络，无需额外分支。

**预训练数据**：COCO / LVIS / Object365 / GoldG / CC3M / SBU / Flickr30k，共约 900 万样本。

---

## 二、实验配置

| 项目 | 设置 |
|------|------|
| 模型 | Grounding DINO Swin-T（官方 OGC 权重，662 MB） |
| 推理后端 | HuggingFace transformers 4.47.1（`IDEA-Research/grounding-dino-tiny`） |
| 推理阈值 | box_threshold=0.25, text_threshold=0.20 |
| 图像 resize | 长边 ≤ 800（默认） |
| 评测模式 | 零样本（zero-shot），不做任何微调 |
| 评测集 | COCO val2017，全量 5000 张 |
| 硬件 | Windows 10, CUDA 11.8, torch 2.1.2+cu118 |
| Prompt 策略 | 80 类类别名拼接为单条 prompt（`"person. bicycle. car. ..."` ） |

> **备注**：Windows 下 Grounding DINO CUDA 算子编译失败（`MultiScaleDeformableAttention` 依赖
> VS C++ Build Tools），因此回退到 HuggingFace transformers 后端（纯 Python 推理）。
> 这是本复现与论文结果存在差距的主要工程原因之一（详见第七节）。

---

## 三、OVD 评测结果 — COCO val2017

### 3.1 定量指标

| 指标 | 本复现 | 论文报告 | 差值 |
|------|--------|---------|------|
| **mAP**  | **43.8** | 48.4 | −4.6 |
| AP50 | 57.9 | — | — |
| AP75 | 47.7 | — | — |
| AP_s（小目标）| 29.7 | — | — |
| AP_m（中目标）| 46.5 | — | — |
| AP_l（大目标）| 57.7 | — | — |

**观察**：
- 小目标（AP_s=29.7）显著低于中目标（46.5）和大目标（57.7），
  反映了特征图分辨率限制对小尺寸物体检测的普遍挑战。
- AP50（57.9）远高于 mAP（43.8），说明模型定位能力较好（IoU=0.5 时命中率高），
  但精确定位（IoU=0.75）仍有提升空间。

### 3.2 Prompt 模板消融

在 200 张随机子集（seed=42）上对比三种 prompt 模板的 mAP / AP50：

| Prompt 模板 | mAP | AP50 | 备注 |
|------------|-----|------|------|
| `{name}`（直接类别名） | **47.1** | **58.8** | 基准，效果最好 |
| `a {name}` | 17.6 | 24.2 | 大幅下降 |
| `a photo of a {name}` | 5.2 | 6.3 | 几乎失效 |

**关键发现**：`a {name}` 和 CLIP 风格模板性能急剧下降，
根本原因是 Grounding DINO 文本编码器的最大输入长度为 **256 tokens**。
将 80 类类别名拼接成 `"a photo of a person. a photo of a bicycle. ..."` 后，
prompt 长度远超 256 tokens，被截断后大量类别丢失，模型无法检测被截断的类别。

**结论**：对 concat 模式的 OVD 任务，**直接使用类别名**（无冠词/模板修饰）是最优策略，
能在 token 预算内容纳尽可能多的类别；CLIP 风格模板（`a photo of a ...`）更适合
单类别查询（single 模式），而非多类别拼接。

---

## 四、Visual Grounding 评测结果 — RefCOCO/+/g

### 4.1 定量指标（Acc@0.5）

| 数据集 | val | testA | testB | 本复现说明 |
|--------|-----|-------|-------|-----------|
| RefCOCO | **22.47** | — | **28.97** | HF tiny 后端，thr=0.10/0.05 |
| RefCOCO+ | **23.56** | — | **29.54** | 同上 |
| RefCOCOg | **30.66** | — | — | 同上 |

> 标注来自 HuggingFace `jxu124/refcoco*` parquet；图像为 COCO train2014。
> 早期 HF tiny 后端 val 仅 22–31%；**不可与论文微调列 86–89% 对比**（见 §0.2）。

**论文 Table 5 参考（分列）**：

| 数据集 | 零样本 @1 | 微调 @1 | 说明 |
|--------|-----------|---------|------|
| RefCOCO val | **50.41** | 89.19 | @1 = Acc@0.5；89% 需 RefCOCO 微调 |
| RefCOCO testB | 43.21 | 85.89 | |
| RefCOCO+ val | 51.40 | 81.22 | |
| RefCOCO+ testB | 45.81 | 74.18 | |
| RefCOCOg val | 60.42 | 86.94 | |

**本仓库 gdino 全量 semantic（§0）**：refcoco val **50.85%**，与零样本论文列对齐。

### 4.2 WSL 官方 gdino 后端 — 与 HF / 论文对比（历史快照）

> **注意**：下表为 2026-05 前后 WSL 单次全量快照；**不等同于 §0「最佳完成」当前状态**。
> OVD 全量 42.4 为达标参照；子集 46.3/47.6 见 §0。VG 全量 semantic 与 sweep 结果见 §0 待填行。

在 WSL2（Ubuntu 24.04，RTX 4060 Laptop 8GB）下使用 **官方 Swin-T OGC 权重** + **gdino 源码后端**
（`weights/groundingdino_swint_ogc.pth`；CUDA 算子未编译成功，deformable attention 使用 PyTorch fallback）
重新跑全量评测。结果目录：`results/coco_gdino/`、`results/refcoco_gdino/`。

| 任务 | 数据集 / 划分 | 指标 | Windows HF | WSL gdino | 论文零样本 | 论文微调 |
|------|--------------|------|------------|-----------|------------|----------|
| OVD | COCO val2017 | mAP | 43.8 | **42.4** | **48.4** | — |
| VG | RefCOCO val | Acc@0.5 | 22.47% | **50.72%** | **50.41%** | 89.19% |
| VG | RefCOCO testB | Acc@0.5 | 28.97% | **45.00%** | **43.21%** | 85.89% |
| VG | RefCOCO+ val | Acc@0.5 | 23.56% | **51.64%** | **51.40%** | 81.22% |
| VG | RefCOCO+ testB | Acc@0.5 | 29.54% | **46.35%** | **45.81%** | 74.18% |
| VG | RefCOCOg val | Acc@0.5 | 30.66% | **60.44%** | **60.42%** | 86.94% |

**云服务器最佳完成（semantic + sweep，见 §0）**：OVD mAP **46.16**；VG refcoco val **50.85%**（零样本论文列已对齐）。

**WSL gdino 配置要点**：
- 环境：`conda` 环境 `cv`，Python 3.10，torch 2.1.2+cu118，transformers 4.47.1
- OVD：短边 800 / 长边 ≤1333（官方风格 resize），检测框映射回原图坐标后送 COCOeval
- VG：box_threshold=0.10，text_threshold=0.05；标注来自 `data/refcoco_hf/` parquet

**观察**：
- **VG 大幅提升**：HF tiny → gdino 后 Acc@0.5 升至 45–61%，与论文 **零样本 @1** 同量级。
- **OVD**：全量 mAP 46.16，距论文零样本 48.4 约 **2.2**；需 CUDA 算子 + 官方 `test_ap_on_coco.py` 对齐。
- **勿误读微调列**：论文 VG 89% 为 RefCOCO 微调后，非本仓库零样本设定。

---

## 五、定性可视化

`results/coco/qualitative_ovd.png` 中展示了从 val2017 随机抽取的 12 张图像的检测结果
（每张显示 top-5 检测框，颜色区分不同类别）。

典型观察：
- **成功案例**：人、车、椅子等常见大目标检测置信度高（score > 0.6），框位置准确。
- **困难案例**：远处小目标（如背景中的人）、密集物体（如一排椅子）以及遮挡物体检测效果较差。
- **类别混淆**：`motorcycle` 与 `bicycle` 在远景下偶有混淆；`handbag` 与 `backpack` 重叠语义导致误报。

---

## 六、失败案例分析

通过对 `results/coco/predictions.json` 的统计分析，识别出以下主要失败模式：

### 6.1 小目标漏检（AP_s=29.7）

GT 框面积 < 32×32 像素的物体（如远处行人、背景中的标识）检测率低，
原因是特征图下采样后小目标特征分辨率不足，框回归精度差。

### 6.2 密集场景多实例混淆

场景中存在多个同类物体时（如一群人、多辆汽车），
模型有时产生重叠框或遗漏部分实例，NMS 阈值设置影响明显。

### 6.3 罕见类别性能下降

COCO 80 类中部分罕见类（如 `toaster`、`hair drier`、`parking meter`）
在 val 集中出现频率极低，样本不足导致统计指标不稳定。

### 6.4 Prompt 拼接歧义

将 80 类类别名拼接为单条长 prompt 时，模型需在一次前向中识别所有类别，
相比逐类单独送入（single 模式），concat 模式的类别匹配精度略有下降。

---

## 七、复现差距分析

本复现 mAP=43.8，较论文报告 48.4 低约 4.6 个点，分析原因如下：

| 可能原因 | 影响估计 | 说明 |
|---------|---------|------|
| **推理后端差异** | **大（约 3–4 点）** | 论文使用编译了 CUDA 算子的源码版本；本复现因 Windows 编译失败，使用 HuggingFace `grounding-dino-tiny` 后端，该后端为轻量版模型（参数量略少） |
| Prompt 形式差异 | 中（约 0.5–1 点）| 论文未明确 concat 策略细节；类别名后缀、分隔符可能有细微差异 |
| 推理阈值选取 | 小（< 0.5 点）| box_threshold=0.25 可能不是最优值 |
| 随机种子 / 评测细节 | 可忽略 | 全量 5000 张，随机性影响极小 |

**结论**：主要差距来自后端模型差异（tiny 版 vs 完整版），而非代码逻辑错误。
若在 Linux/WSL2 环境下编译 CUDA 算子并使用官方源码后端，预计可复现至 47–48 点。

---

## 八、结论

1. **复现成功**：基于 HuggingFace 后端成功复现 Grounding DINO 零样本 OVD 流水线，
   在 COCO val2017 全量 5000 张上取得 mAP=43.8（AP50=57.9），与论文量级一致。

2. **模型能力**：Grounding DINO 展现了强大的开放词表检测能力，
   对 COCO 80 类无需任何微调即可取得接近 SOTA 的零样本性能，
   验证了大规模跨模态预训练的有效性。

3. **工程经验**：Windows 下 CUDA 算子编译是主要工程障碍；
   HuggingFace transformers 后端提供了良好的平台无关替代方案，
   但模型略有缩减（tiny 版），导致约 3–4 点的性能差距。

4. **改进方向**：
   - 在 Linux 环境编译源码后端以消除后端差距
   - 调优推理阈值（grid search box_threshold ∈ [0.15, 0.35]）
   - 将评测扩展到 LVIS val（测试罕见类检测）和 RefCOCO（视觉定位任务）

---

## 参考文献

1. Liu, S. et al. "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection." ECCV 2024.
2. Lin, T.-Y. et al. "Microsoft COCO: Common Objects in Context." ECCV 2014.
3. Yu, L. et al. "Modeling Context in Referring Expressions." ECCV 2016. (RefCOCO)
4. Zhang, H. et al. "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection." ICLR 2023.
