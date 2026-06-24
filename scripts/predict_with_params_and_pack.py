#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/enhanced_vm_v2_resume80/checkpoint_best_screened.pkl")
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--seed_k", type=int, default=6)
    parser.add_argument("--seed_k_alpha", type=int, default=1)
    parser.add_argument("--zip_path", default="results/result.zip")
    return parser.parse_args()


def patch_vm_defaults(path, patch_size, seed_k, seed_k_alpha):
    text = path.read_text()
    old = "def patch_based_denoise(model: VelocityModule, pcl_noisy, patch_size=1000, seed_k=6, seed_k_alpha=1) -> jt.Var:"
    new = (
        "def patch_based_denoise(model: VelocityModule, pcl_noisy, "
        f"patch_size={patch_size}, seed_k={seed_k}, seed_k_alpha={seed_k_alpha}) -> jt.Var:"
    )
    if old not in text:
        raise RuntimeError("Could not find original patch_based_denoise defaults")
    path.write_text(text.replace(old, new))


def restore_file(path, backup):
    path.write_text(backup)


def pack_results(pred_dir, zip_path):
    files = sorted(p for p in pred_dir.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"No prediction files found under {pred_dir}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(pred_dir).as_posix())
    return len(files)


def main():
    args = parse_args()
    root = Path.cwd()
    vm_path = root / "src/model/vm.py"
    backup = vm_path.read_text()
    zip_path = Path(args.zip_path)
    pred_dir = Path("results/dataset_test_noisy")

    if zip_path.exists():
        backup_zip = zip_path.with_name(f"{zip_path.stem}_before_param_predict{zip_path.suffix}")
        shutil.copy2(zip_path, backup_zip)
        print(f"backed up existing zip to {backup_zip}")

    try:
        patch_vm_defaults(vm_path, args.patch_size, args.seed_k, args.seed_k_alpha)
        subprocess.run(
            ["bash", "scripts/predict_enhanced_container_and_pack.sh", args.checkpoint],
            check=False,
        )
    finally:
        restore_file(vm_path, backup)

    count = pack_results(pred_dir, zip_path)
    print(f"wrote {zip_path} with {count} files")


if __name__ == "__main__":
    main()