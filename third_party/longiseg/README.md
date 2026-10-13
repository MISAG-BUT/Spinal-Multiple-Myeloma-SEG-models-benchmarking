# Modified LongiSeg

All models were trained with [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) (`nnUNetTrainer`, `nnUNetPlans`, `3d_fullres`).
The **evaluation** (`LongiSeg_evaluate_folder`) uses a lightly modified copy of
[MIC-DKFZ/LongiSeg](https://github.com/MIC-DKFZ/LongiSeg) (Apache-2.0).
Training and inference are unaffected by the modifications; only the lesion-wise detection metrics (F1 and instance statistics) are changed.

- Upstream commit used: [`3a5cef3`](https://github.com/MIC-DKFZ/LongiSeg/commit/3a5cef35ad9eb1146c9a455ab3143a7a78de0427) (upstream has evolved since).
- Our changes: [`longiseg_modifications.patch`](longiseg_modifications.patch) (plain `git diff` against that commit).
- Install: [`install_longiseg.sh`](install_longiseg.sh) clones upstream, checks out the commit and applies the patch.

```bash
bash third_party/longiseg/install_longiseg.sh   # run inside the activated environment
```

## What was changed

| File | Change | Why |
|------|--------|-----|
| `longiseg/evaluation/metrics/detection_metrics.py` | `get_inst_TPFPFN` / `compute_detection_metrics` take an `iou_threshold` argument instead of the hard-coded `max_iou > 0.1`. | The instance-matching IoU threshold is varied in the IoU sensitivity analysis (0.05-0.90, `evaluation/run_iou_threshold_analysis.sh`). |
| `longiseg/evaluation/evaluate_predictions_longi.py` | New CLI option `-iou_threshold` (default `0.1`), passed through `compute_longi_metrics_on_folder2` -> `compute_longi_metrics_on_folder` -> `compute_longi_metrics` -> `compute_detection_metrics`. The `starmap` call now passes `distance_threshold=1`, `size_threshold=0`, `footprint=0` explicitly so that `iou_threshold` lands in the right positional argument. | Makes the IoU threshold configurable. |
| `longiseg/evaluation/evaluate_predictions_longi.py` | Lesion-size filter: connected components smaller than `size_threshold` (mm³) are removed from both reference and prediction with `skimage.morphology.remove_small_objects(..., max_size=...)` before any metric is computed. `size_threshold` and an `empty_mask` flag are stored in the per-case results. An (unused) range filter `filter_objects_by_size` is included. | Lesion-size analyses ("larger than 0.3 cm³" / "0.5 cm³"). |

## Reproducing the lesion-size variants

The size threshold is a constant inside `compute_longi_metrics` (`evaluation/evaluate_predictions_longi.py`, mm³):

| Analysis (`ANALYSIS_NAME`) | `size_threshold` |
|----------------------------|------------------|
| `longi_summary_all` | `0` (patch default) |
| `longi_summary_larger_than_0_3_cubic_cm` | `300` |
| `longi_summary_larger_than_0_5_cubic_cm` | `500` |

Edit the constant (the values are in the commented lines), reinstall is not needed (editable install), then run e.g.

```bash
ANALYSIS_NAME=longi_summary_larger_than_0_5_cubic_cm bash evaluation/run_evaluation.sh
```

## Notes

- `remove_small_objects(max_size=...)` requires `scikit-image >= 0.26` (the results were produced with 0.26.0).
- The patch is kept verbatim as used for the paper, including some commented-out experiments.
- Please cite the original LongiSeg work when using this code (see the upstream repository for the reference).
