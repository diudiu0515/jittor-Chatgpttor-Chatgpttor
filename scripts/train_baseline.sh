#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source scripts/activate_env.sh

SEED="${1:-123}"
mkdir -p experiments results logs

python run.py --task configs/task/train_vm.yaml --seed "$SEED" 2>&1 | tee "logs/train_seed_${SEED}.log"
