#!/usr/bin/env python3
"""Validate Track 2 result.zip structure and numpy outputs."""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="results/result.zip")
    parser.add_argument("--noisy_root", default="dataset_test_noisy/shapenet")
    return parser.parse_args()


def main():
    args = parse_args()
    zip_path = Path(args.zip)
    noisy_root = Path(args.noisy_root)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    if not noisy_root.exists():
        raise FileNotFoundError(noisy_root)

    noisy_files = sorted(noisy_root.glob("*/*/noisy.npy"))
    expected = {
        f"shapenet/{path.parent.parent.name}/{path.parent.name}/denoised.npy": path
        for path in noisy_files
    }

    errors = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        denoised = [name for name in names if name.endswith("denoised.npy")]
        extra = sorted(set(denoised) - set(expected))
        missing = sorted(set(expected) - set(denoised))
        if extra:
            errors.append(f"extra denoised files: {len(extra)}")
        if missing:
            errors.append(f"missing denoised files: {len(missing)}")

        for name in denoised:
            if name not in expected:
                continue
            pred = np.load(io.BytesIO(zf.read(name)))
            noisy = np.load(expected[name])
            if pred.dtype != np.float32:
                errors.append(f"{name}: dtype {pred.dtype} != float32")
            if pred.shape != noisy.shape:
                errors.append(f"{name}: shape {pred.shape} != noisy {noisy.shape}")
            if pred.ndim != 2 or pred.shape[1] != 3:
                errors.append(f"{name}: invalid point shape {pred.shape}")

    print(f"zip: {zip_path}")
    print(f"denoised files: {len(denoised)}")
    print(f"expected files: {len(expected)}")
    if errors:
        print("FAILED")
        for err in errors[:20]:
            print(f"- {err}")
        if len(errors) > 20:
            print(f"- ... {len(errors) - 20} more")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
