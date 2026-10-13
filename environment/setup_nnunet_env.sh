#!/bin/bash
# Environment for training / inference / evaluation (nnU-Net + modified LongiSeg).
# Usage: bash environment/setup_nnunet_env.sh [CUDA_TAG]   (default CUDA_TAG=cu126, see https://pytorch.org/get-started/locally/)
# Tested: Python 3.12, torch 2.10.0+cu130 (inference/evaluation), torch 2.6.0+cu124 / 2.7.0 (training), nnunetv2 2.6.4.

set -euo pipefail

CUDA_TAG="${1:-cu126}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3.12 -m venv .venv-nnunet
source .venv-nnunet/bin/activate

pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# nnU-Net at the commit used for the paper (nnunetv2 2.6.4)
mkdir -p third_party_src
[[ -d third_party_src/nnUNet ]] || git clone https://github.com/MIC-DKFZ/nnUNet.git third_party_src/nnUNet
git -C third_party_src/nnUNet checkout f6d221d1b79cd2173650f78f97ecfee273e0cf86
pip install -e third_party_src/nnUNet

# Modified LongiSeg (evaluation only)
bash third_party/longiseg/install_longiseg.sh "$REPO_ROOT/third_party_src/LongiSeg_modified"

# Analysis dependencies
pip install numpy pandas scipy matplotlib seaborn openpyxl

echo "Done. Activate with: source .venv-nnunet/bin/activate && source config.sh"
