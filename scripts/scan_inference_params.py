#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/enhanced_vm_v2_resume80/checkpoint_best_screened.pkl")
    parser.add_argument("--model_config", default="configs/model/enhanced_vm.yaml")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--patch_sizes", default="800,1000,1200,1500")
    parser.add_argument("--seed_ks", default="4,6,8,10")
    parser.add_argument("--seed_k_alphas", default="1")
    parser.add_argument("--out_dir", default="experiments/enhanced_vm_v2_resume80/inference_scan")
    parser.add_argument("--summary_csv", default="experiments/enhanced_vm_v2_resume80/inference_scan_summary.csv")
    return parser.parse_args()


def parse_ints(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def extract_score(output, name):
    match = re.search(rf"{name}:\s*([0-9.]+)", output)
    return float(match.group(1)) if match else None


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for patch_size in parse_ints(args.patch_sizes):
        for seed_k in parse_ints(args.seed_ks):
            for seed_k_alpha in parse_ints(args.seed_k_alphas):
                tag = f"patch{patch_size}_seed{seed_k}_alpha{seed_k_alpha}_limit{args.limit}"
                output_csv = out_dir / f"{tag}.csv"
                cmd = [
                    "scripts/run_in_runtime_container.sh",
                    "python",
                    "validate_metrics.py",
                    "--checkpoint",
                    args.checkpoint,
                    "--model_config",
                    args.model_config,
                    "--limit",
                    str(args.limit),
                    "--output_csv",
                    str(output_csv),
                    "--patch_size",
                    str(patch_size),
                    "--seed_k",
                    str(seed_k),
                    "--seed_k_alpha",
                    str(seed_k_alpha),
                ]
                print(f"\n=== {tag} ===", flush=True)
                proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                print(proc.stdout, flush=True)
                rows.append({
                    "patch_size": patch_size,
                    "seed_k": seed_k,
                    "seed_k_alpha": seed_k_alpha,
                    "returncode": proc.returncode,
                    "mean_cd_score": extract_score(proc.stdout, "mean_cd_score"),
                    "mean_p2s_score": extract_score(proc.stdout, "mean_p2s_score"),
                    "final_score": extract_score(proc.stdout, "final_score"),
                    "output_csv": str(output_csv),
                })
                with summary_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    valid = [row for row in rows if row["final_score"] is not None]
    valid.sort(key=lambda row: row["final_score"], reverse=True)
    print("\n=== Best results ===")
    for row in valid[:10]:
        print(row)
    print(f"summary_csv: {summary_path}")


if __name__ == "__main__":
    main()