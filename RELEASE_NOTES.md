# v1.0.0

First official public release of **Spinal-Multiple-Myeloma-SEG models benchmarking**.

- Benchmarking workflow for 19 nnU-Net configurations using conventional CT,
  virtual monoenergetic images, and calcium-suppressed dual-energy CT reconstructions.
- Five-fold cross-validation inference scripts, channel-ablation workflows,
  lesion-wise evaluation, lesion-size and IoU-threshold sensitivity analyses,
  statistical analysis, and inference time and GPU usage summaries.
- Stored evaluation outputs, cross-validation splits, figures, and tables under
  `results/`, with analysis scripts and environment setup instructions.
- Software citation metadata in `CITATION.cff`; code licensed under Apache-2.0.

The 19 trained benchmarking models (five folds each) are archived separately on
[Zenodo (DOI: 10.5281/zenodo.19708429)](https://doi.org/10.5281/zenodo.19708429).
That DOI identifies the model weights, not this software release.

Source data: [TCIA (DOI: 10.7937/k4qv-hh78)](https://doi.org/10.7937/k4qv-hh78);
[dataset publication](https://doi.org/10.1038/s41597-026-08061-x).
The associated manuscript, *Benchmarking Spectral Reconstructions (VMI and
Calcium-Suppressed) for Multiple Myeloma Segmentation on Dual-Energy CT*, is in
preparation and has no DOI.
