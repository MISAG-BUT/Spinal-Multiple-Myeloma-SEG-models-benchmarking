"""Central path configuration shared by all Python scripts.

Defaults are relative to the repository root. Override with environment variables:

    MM_RESULTS_DIR   directory with evaluation results, logs, tables and figures (default: ./results)
    nnUNet_results   nnU-Net trained models (only needed for GPU/model-configuration tables)
"""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.abspath(os.environ.get("MM_RESULTS_DIR", os.path.join(REPO_ROOT, "results")))
# NNUNET_RESULTS = os.environ.get("nnUNet_results", os.path.join(REPO_ROOT, "nnUNet_results"))
NNUNET_RESULTS = os.environ.get("nnUNet_results", "/home/nohel/DATA/nnUNet_results")

FULL_MODELS_DIR = os.path.join(RESULTS_DIR, "full_models")
ZERO_INPUT_MODELS_DIR = os.path.join(RESULTS_DIR, "zero_input_models")
IOU_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "iou_threshold_analysis_full_models")
GPU_LOG_DIR = os.path.join(RESULTS_DIR, "gpu_logs")
INFERENCE_LOG_DIR = os.path.join(RESULTS_DIR, "inference_logs")
SPLITS_FILE = os.path.join(RESULTS_DIR, "splits_final.json")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
