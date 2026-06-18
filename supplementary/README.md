# Supplementary Material

This supplementary package contains selected evaluation artifacts used by the final report.

## Contents

- `metrics/ovd_self_metrics.json`: main self-implemented COCO OVD result, mAP 46.16.
- `metrics/ovd_official_comparison.json`: official GroundingDINO COCO evaluation comparison, official mAP 48.50.
- `metrics/vg_full_metrics.json`: main RefCOCO / RefCOCO+ / RefCOCOg visual grounding results.
- `metrics/prompt_ablation.json`: OVD prompt ablation results.
- `metrics/vg_sweep_best_subset.json`: best visual grounding threshold sweep result.
- `metrics/vg_protocol_comparison.json`: semantic vs argmax_official box selection protocol check.
- `metrics/cuda_build_status.json`: CUDA custom operator build status from the reproduction environment.
- `metrics/5090_ovd_cuda_fixed_metrics.json`: RTX 5090 CUDA-fixed engineering validation result.
- `metrics/5090_vg_testA_metrics.json`: RTX 5090 RefCOCO testA supplementary evaluation result.

The final report is self-contained. These files are included only to make the reported numbers auditable.
