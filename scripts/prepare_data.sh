#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(cd "$ROOT_DIR/.." && pwd)"

cd "$ROOT_DIR"

prepare_train() {
  if [[ -d dataset_train ]]; then
    echo "dataset_train already exists"
    return
  fi
  if [[ -d "$PARENT_DIR/repo_check/dataset_train" ]]; then
    ln -s "$PARENT_DIR/repo_check/dataset_train" dataset_train
    echo "linked dataset_train from repo_check"
    return
  fi
  if [[ -f "$PARENT_DIR/dataset_train.tar.gz" ]]; then
    tar xzf "$PARENT_DIR/dataset_train.tar.gz"
    echo "extracted dataset_train.tar.gz"
    return
  fi
  echo "dataset_train not found" >&2
  exit 1
}

prepare_test() {
  if [[ -d dataset_test_noisy ]]; then
    echo "dataset_test_noisy already exists"
    return
  fi
  if [[ -d "$PARENT_DIR/repo_check/dataset_test_noisy" ]]; then
    ln -s "$PARENT_DIR/repo_check/dataset_test_noisy" dataset_test_noisy
    echo "linked dataset_test_noisy from repo_check"
    return
  fi
  if [[ -f "$PARENT_DIR/dataset_test_noisy.zip" ]]; then
    unzip -q "$PARENT_DIR/dataset_test_noisy.zip"
    echo "extracted dataset_test_noisy.zip"
    return
  fi
  echo "dataset_test_noisy not found" >&2
  exit 1
}

prepare_train
prepare_test

python - <<'PY'
from pathlib import Path
checks = [
    Path('dataset_train/shapenet'),
    Path('dataset_test_noisy/shapenet'),
    Path('datalist/train.txt'),
    Path('datalist/validate.txt'),
    Path('datalist/test.txt'),
]
for path in checks:
    print(f'{path}:', 'OK' if path.exists() else 'MISSING')
PY
