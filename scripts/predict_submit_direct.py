#!/usr/bin/env python3
"""Direct Track 2 prediction without dataloader/system wrappers."""

import argparse
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jittor as jt
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from src.model.parse import get_model
from src.model.vm import patch_based_denoise
from scripts.tta_utils import parse_angles, tta_denoise

jt.flags.use_cuda = 1


def load_yaml(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_config", default="configs/model/enhanced_vm.yaml")
    parser.add_argument("--transform_config", default="configs/transform/predict.yaml")
    parser.add_argument("--list", default="datalist/test.txt")
    parser.add_argument("--input_root", default="dataset_test_noisy")
    parser.add_argument("--output_root", default="results/dataset_test_noisy")
    parser.add_argument("--zip", default="results/result.zip")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tta", action="store_true", help="Enable rotation TTA and iterative denoising")
    parser.add_argument("--tta_angles", default="0,120,240")
    parser.add_argument("--tta_iterations", type=int, default=2)
    parser.add_argument("--tta_alpha", type=float, default=0.5)
    parser.add_argument("--tta_alpha_decay", type=float, default=0.9)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--seed_k", type=int, default=6)
    parser.add_argument("--seed_k_alpha", type=int, default=1)
    parser.add_argument("--bilateral", action="store_true")
    parser.add_argument("--bilateral_k", type=int, default=20)
    parser.add_argument("--bilateral_sigma_s", type=float, default=0.01)
    parser.add_argument("--bilateral_sigma_r", type=float, default=0.02)
    return parser.parse_args()


def main():
    args = parse_args()
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_root = Path(args.output_root)
    zip_path = Path(args.zip)
    if not args.resume:
        if output_root.exists():
            shutil.rmtree(output_root)
        if zip_path.exists():
            zip_path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)

    transform_config = load_yaml(args.transform_config)
    model_config = load_yaml(args.model_config)
    model = get_model(model_config=model_config, transform_config=transform_config)
    model.load(args.checkpoint)
    model.set_predict(True)
    model.eval()

    rel_paths = [line.strip() for line in open(args.list) if line.strip()]
    if args.limit > 0:
        rel_paths = rel_paths[:args.limit]

    with jt.no_grad():
        for rel in tqdm(rel_paths, desc="direct predict"):
            noisy_path = Path(args.input_root) / rel / "noisy.npy"
            out_path = output_root / rel / "denoised.npy"
            if args.resume and out_path.exists():
                continue
            pc_noisy = np.load(noisy_path).astype(np.float32)
            if args.tta:
                pc_denoised_np = tta_denoise(
                    model=model,
                    pc_noisy=pc_noisy,
                    angles=parse_angles(args.tta_angles),
                    iterations=args.tta_iterations,
                    alpha=args.tta_alpha,
                    alpha_decay=args.tta_alpha_decay,
                    patch_size=args.patch_size,
                    seed_k=args.seed_k,
                    seed_k_alpha=args.seed_k_alpha,
                    use_bilateral=args.bilateral,
                    bilateral_k=args.bilateral_k,
                    bilateral_sigma_s=args.bilateral_sigma_s,
                    bilateral_sigma_r=args.bilateral_sigma_r,
                ).astype(np.float32)
            else:
                pc_denoised = patch_based_denoise(
                    model=model,
                    pcl_noisy=jt.array(pc_noisy),
                    patch_size=args.patch_size,
                    seed_k=args.seed_k,
                    seed_k_alpha=args.seed_k_alpha,
                )
                if pc_denoised is None:
                    raise RuntimeError(f"denoise failed: {rel}")
                pc_denoised_np = pc_denoised.detach().numpy().astype(np.float32)
            if pc_denoised_np.shape != pc_noisy.shape:
                raise RuntimeError(f"shape mismatch for {rel}: {pc_denoised_np.shape} != {pc_noisy.shape}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, pc_denoised_np)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    shapenet_root = output_root / "shapenet"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(shapenet_root.glob("*/*/denoised.npy")):
            zf.write(file_path, file_path.relative_to(output_root))
    print(f"submission: {zip_path}")


if __name__ == "__main__":
    main()
