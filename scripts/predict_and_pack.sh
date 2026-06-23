#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source scripts/activate_env.sh

CKPT="${1:-experiments/vm/checkpoint_99.pkl}"
PRED_TASK="configs/task/predict_vm.local.yaml"

python - <<PY
from pathlib import Path
src = Path('configs/task/predict_vm.yaml')
dst = Path('$PRED_TASK')
text = src.read_text()
lines = []
for line in text.splitlines():
    if line.startswith('load_ckpt:'):
        lines.append('load_ckpt: $CKPT')
    elif line.strip() == 'save_dir: tmp_predict':
        lines.append('  save_dir: results/dataset_test_noisy')
    else:
        lines.append(line)
dst.write_text('\n'.join(lines) + '\n')
print(f'wrote {dst}')
PY

python run.py --task "$PRED_TASK"

cd results/dataset_test_noisy
zip -qr ../result.zip shapenet
cd "$ROOT_DIR"
echo "submission: $ROOT_DIR/results/result.zip"
