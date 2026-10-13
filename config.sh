#!/bin/bash
# Central path configuration for the shell scripts. Source it: `source "$REPO_ROOT/config.sh"`.
# Every variable can be overridden from the environment.

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Repository results (evaluation JSONs, logs, tables, figures)
ANALYSIS_ROOT="${MM_RESULTS_DIR:-$CONFIG_DIR/results}"

# nnU-Net folders (see https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md)
export nnUNet_raw="${nnUNet_raw:-$CONFIG_DIR/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$CONFIG_DIR/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$CONFIG_DIR/data/nnUNet_results}"

RAW_ROOT="$nnUNet_raw"
RESULTS_ROOT="$nnUNet_results"
TRAINED_MODEL_ROOT="$nnUNet_results"

# Predictions produced by inference/run_predictions.sh
PREDICT_ROOT="${PREDICT_ROOT:-$nnUNet_results/predict_new}"

# Ground-truth labels split by cross-validation fold (labelsTr_fold_0 ... labelsTr_fold_4)
GT_ROOT="${GT_ROOT:-$nnUNet_raw/MM_GT_DATA}"
