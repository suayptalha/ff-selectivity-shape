#!/usr/bin/env bash
# ==============================================================================
# Forward-Forward Rebuttal Experiment Launcher
# Runs full Dense MLP & ConvNet Rebuttal Suite using CUDA Python environment
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# Locate PyTorch Python environment
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for _p in "/home/user2/newvenv/bin/python" \
            "${HOME}/miniconda3/envs/torch-env/bin/python" \
            "${HOME}/anaconda3/envs/torch-env/bin/python"; do
    if [ -x "$_p" ] && "$_p" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      PYTHON="$_p"; break
    fi
  done
  [ -z "$PYTHON" ] && PYTHON=python3
fi

echo "=================================================================="
echo "  Forward-Forward Rebuttal Execution Suite"
echo "  Python Executable: $PYTHON"
echo "=================================================================="
echo ""

echo "--> Running Dense MLP Rebuttal Suite (55k/5k/10k split, 5-seed eval)..."
"$PYTHON" -u run_rebuttal.py

echo ""
echo "=================================================================="
echo "  All Rebuttal Experiments Completed Successfully!"
echo "=================================================================="
