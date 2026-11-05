#!/usr/bin/env bash
# ==============================================================================
# Forward-Forward Extra Baselines Launcher
# Runs SymBa, SCFF, SoS, and Burstiness comparisons on Fashion-MNIST & MNIST
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "=================================================================="
echo "  Running Extra Baselines Benchmark (SymBa, SCFF, SoS, Burstiness)"
echo "  Python Executable: $PYTHON"
echo "=================================================================="
echo ""

echo "--> Running Fashion-MNIST..."
"$PYTHON" -u run_extra_baselines.py fashion_mnist

echo ""
echo "--> Running MNIST..."
"$PYTHON" -u run_extra_baselines.py mnist

echo ""
echo "=================================================================="
echo "  Extra Baselines Execution Completed Successfully!"
echo "=================================================================="
