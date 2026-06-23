#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CKPT="${1:-experiments/enhanced_vm/checkpoint_99.pkl}"
EVAL_LOG="logs/train_enhanced_eval_cd_p2s.log"
OUTPUT_CSV="experiments/enhanced_vm/validation_metrics.csv"
LIMIT="${LIMIT:-50}"

echo "waiting for $CKPT ..."
until [ -f "$CKPT" ]; do
	sleep 30
done

echo "found $CKPT"
scripts/run_in_runtime_container.sh python validate_metrics.py \
	--checkpoint "$CKPT" \
	--data_config configs/data/train_enhanced.yaml \
	--transform_config configs/transform/vm.yaml \
	--model_config configs/model/enhanced_vm.yaml \
	--limit "$LIMIT" \
	--output_csv "$OUTPUT_CSV" | tee "$EVAL_LOG"

echo "evaluation log: $ROOT_DIR/$EVAL_LOG"
echo "evaluation csv: $ROOT_DIR/$OUTPUT_CSV"