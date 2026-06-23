#!/usr/bin/env python3
"""Screen enhanced checkpoints with local CD/P2S validation."""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint paths to evaluate, relative to workspace or absolute.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out_csv", default="experiments/enhanced_vm/checkpoint_screen_metrics.csv")
    parser.add_argument("--best_out", default="experiments/enhanced_vm/checkpoint_best_screened.pkl")
    parser.add_argument("--model_config", default="configs/model/enhanced_vm.yaml")
    return parser.parse_args()


def read_score(csv_path: Path, checkpoint: str):
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows written by validate_metrics for {checkpoint}")
    cd_scores = [float(row["cd_score"]) for row in rows]
    p2s_scores = [float(row["p2s_score"]) for row in rows]
    final_scores = [float(row["final_score"]) for row in rows]
    return {
        "checkpoint": checkpoint,
        "samples": len(rows),
        "mean_cd_score": sum(cd_scores) / len(cd_scores),
        "mean_p2s_score": sum(p2s_scores) / len(p2s_scores),
        "final_score": sum(final_scores) / len(final_scores),
    }


def main():
    args = parse_args()
    root = Path.cwd()
    out_csv = root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_csv.parent / ".screen_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = root / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt)

        tmp_csv = tmp_dir / f"{ckpt_path.stem}_limit{args.limit}.csv"
        cmd = [
            sys.executable,
            "validate_metrics.py",
            "--checkpoint",
            str(ckpt_path.relative_to(root) if ckpt_path.is_relative_to(root) else ckpt_path),
            "--model_config",
            args.model_config,
            "--limit",
            str(args.limit),
            "--output_csv",
            str(tmp_csv.relative_to(root)),
        ]
        print(f"\n[screen] evaluating {ckpt_path.name} limit={args.limit}", flush=True)
        if tmp_csv.exists():
            print(f"[screen] reusing {tmp_csv.relative_to(root)}", flush=True)
        else:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"[screen] failed {ckpt_path.name}: exit {exc.returncode}; skipping", flush=True)
                continue
        result = read_score(tmp_csv, str(ckpt_path.relative_to(root) if ckpt_path.is_relative_to(root) else ckpt_path))
        results.append(result)
        print(
            "[screen] {checkpoint} final={final_score:.4f} cd={mean_cd_score:.4f} p2s={mean_p2s_score:.4f}".format(
                **result
            ),
            flush=True,
        )

    if not results:
        raise RuntimeError("No checkpoints were evaluated successfully")

    results.sort(key=lambda item: item["final_score"], reverse=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "checkpoint", "samples", "mean_cd_score", "mean_p2s_score", "final_score"],
        )
        writer.writeheader()
        for rank, row in enumerate(results, 1):
            writer.writerow({"rank": rank, **row})

    best_ckpt = root / results[0]["checkpoint"]
    best_out = root / args.best_out
    best_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt, best_out)

    print("\n[screen] ranking:")
    for rank, row in enumerate(results, 1):
        print(f"#{rank}: {row['checkpoint']} final={row['final_score']:.4f}")
    print(f"[screen] best copied to {best_out.relative_to(root)}")


if __name__ == "__main__":
    main()
