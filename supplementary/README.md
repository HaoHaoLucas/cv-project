# Supplementary Material

This supplementary package contains selected evaluation artifacts used by the final report.

## Metrics

- `metrics/ovd_self_metrics.json`: main self-implemented COCO OVD result, mAP 46.16.
- `metrics/ovd_official_comparison.json`: official GroundingDINO COCO evaluation comparison, official mAP 48.50.
- `metrics/vg_full_metrics.json`: main RefCOCO / RefCOCO+ / RefCOCOg visual grounding results.
- `metrics/prompt_ablation.json`: OVD prompt ablation results.
- `metrics/vg_sweep_best_subset.json`: best visual grounding threshold sweep result.
- `metrics/vg_protocol_comparison.json`: semantic vs argmax_official box selection protocol check.
- `metrics/cuda_build_status.json`: CUDA custom operator build status from the reproduction environment.
- `metrics/5090_ovd_cuda_fixed_metrics.json`: RTX 5090 CUDA-fixed engineering validation result.
- `metrics/5090_vg_testA_metrics.json`: RTX 5090 RefCOCO testA supplementary evaluation result.

## Predictions

- `predictions/ovd_self_predictions.json`: raw COCO-format predictions for the main self-implemented OVD result.
- `predictions/vg_refcoco_validation_predictions.json`: raw predictions for RefCOCO validation.
- `predictions/vg_refcoco_testB_predictions.json`: raw predictions for RefCOCO testB.
- `predictions/vg_refcoco+_validation_predictions.json`: raw predictions for RefCOCO+ validation.
- `predictions/vg_refcoco+_testB_predictions.json`: raw predictions for RefCOCO+ testB.
- `predictions/vg_refcocog_validation_predictions.json`: raw predictions for RefCOCOg validation.

The full sweep prediction files are intentionally not included because they are threshold-tuning intermediates. The final report is self-contained; these selected files make the reported numbers auditable.
