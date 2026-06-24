#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CKPT="${1:-experiments/enhanced_vm/checkpoint_99.pkl}"
PRED_TASK="configs/task/predict_enhanced_vm.local.yaml"
PRED_DATA="configs/data/predict_submit.local.yaml"
PRED_DIR="results/dataset_test_noisy"
ZIP_PATH="results/result.zip"

scripts/run_in_runtime_container.sh rm -rf "$PRED_DIR" "$ZIP_PATH"

python - <<PY
from pathlib import Path
src = Path('configs/data/predict.yaml')
dst = Path('$PRED_DATA')
text = src.read_text()
text = text.replace('num_workers: 8', 'num_workers: 0')
dst.write_text(text)
print(f'wrote {dst}')
PY

python - <<PY
from pathlib import Path
src = Path('configs/task/predict_enhanced_vm.yaml')
dst = Path('$PRED_TASK')
text = src.read_text()
lines = []
for line in text.splitlines():
    if line.startswith('load_ckpt:'):
        lines.append('load_ckpt: $CKPT')
    elif line.strip() == 'data: predict':
        lines.append('  data: predict_submit.local')
    elif line.strip() == 'save_dir: tmp_predict_enhanced':
        lines.append('  save_dir: results/dataset_test_noisy')
    elif line.strip() == 'save_name: predict':
        lines.append('  save_name: denoised')
    else:
        lines.append(line)
dst.write_text('\n'.join(lines) + '\n')
print(f'wrote {dst}')
PY

scripts/run_in_runtime_container.sh python run.py --task "$PRED_TASK"

python - <<'PY'
from pathlib import Path
import zipfile

root = Path('results/dataset_test_noisy')
out = Path('results/result.zip')
files = sorted(p for p in root.rglob('*') if p.is_file())
if not files:
    raise SystemExit(f'no prediction files found under {root}')
if out.exists():
    out.unlink()
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for p in files:
        zf.write(p, p.relative_to(root).as_posix())
print(f'wrote {out} with {len(files)} files')
PY
echo "submission: $ROOT_DIR/results/result.zip"