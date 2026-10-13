#!/bin/bash
# Installs LongiSeg at the exact upstream commit used in the paper and applies our modifications.
# Usage (inside the activated environment): bash third_party/longiseg/install_longiseg.sh [TARGET_DIR]

set -euo pipefail

UPSTREAM_URL="https://github.com/MIC-DKFZ/LongiSeg.git"
UPSTREAM_COMMIT="3a5cef35ad9eb1146c9a455ab3143a7a78de0427"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-$HERE/../../third_party_src/LongiSeg_modified}"

if [[ ! -d "$TARGET_DIR/.git" ]]; then
    git clone "$UPSTREAM_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"
git checkout "$UPSTREAM_COMMIT"

if git apply --check "$HERE/longiseg_modifications.patch" 2>/dev/null; then
    git apply "$HERE/longiseg_modifications.patch"
    echo "Patch applied."
else
    echo "Patch already applied or does not apply cleanly - skipping."
fi

pip install -e .
