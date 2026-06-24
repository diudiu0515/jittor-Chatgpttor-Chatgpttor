#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-123}"
mkdir -p experiments/enhanced_vm_v2_resume80 results logs

scripts/run_in_runtime_container.sh \
	python run.py --task configs/task/train_enhanced_vm_v2_resume80.yaml --seed "$SEED" \
	2>&1 | tee "logs/train_enhanced_v2_resume80_container_seed_${SEED}.log"