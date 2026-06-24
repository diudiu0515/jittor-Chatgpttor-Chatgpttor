#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-456}"
mkdir -p experiments/enhanced_vm_v3_conservative results logs

scripts/run_in_runtime_container.sh \
	python run.py --task configs/task/train_enhanced_vm_v3_conservative.yaml --seed "$SEED" \
	2>&1 | tee "logs/train_enhanced_v3_conservative_container_seed_${SEED}.log"