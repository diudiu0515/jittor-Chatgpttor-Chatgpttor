import argparse
import csv
import glob
import os
import random
import shutil

import jittor as jt
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.dataset import DatasetConfig, PCDatasetModule
from src.model.parse import get_model


jt.flags.use_cuda = 1


def load_yaml(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def build_validation(task_path, seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    task = load_yaml(task_path)
    components = task["components"]

    data_config = load_yaml(os.path.join("configs/data", f"{components['data']}.yaml"))
    transform_config = load_yaml(os.path.join("configs/transform", f"{components['transform']}.yaml"))
    model_config = load_yaml(os.path.join("configs/model", f"{components['model']}.yaml"))

    validate_dataset_config = DatasetConfig.parse(**data_config["validate_dataset"]).split_by_cls()
    model = get_model(model_config=model_config, transform_config=transform_config)
    model.set_predict(False)
    model.eval()
    dataset_module = PCDatasetModule(
        process_fn=model._process_fn,
        validate_dataset_config=validate_dataset_config,
        validate_transform=model.get_validate_transform(),
        debug=task.get("debug", False),
    )
    return model, dataset_module


def evaluate_checkpoint(checkpoint_path, task_path, seed):
    model, dataset_module = build_validation(task_path, seed)
    model.load(checkpoint_path)
    model.eval()

    losses = []
    validate_dataloader = dataset_module.validate_dataloader()
    assert validate_dataloader is not None, "validate_dataloader is None"
    if not isinstance(validate_dataloader, dict):
        validate_dataloader = {"validate": validate_dataloader}

    with jt.no_grad():
        for name, dataloader in validate_dataloader.items():
            total = max(1, len(dataloader) // dataloader.batch_size)
            for batch in tqdm(dataloader, total=total, desc=f"{os.path.basename(checkpoint_path)} {name}"):
                loss_dict = model.training_step(batch)
                loss = loss_dict["loss"]
                losses.append(loss.item() if isinstance(loss, jt.Var) else float(loss))

    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="configs/task/train_vm.yaml")
    parser.add_argument("--ckpt_dir", default="experiments/vm")
    parser.add_argument("--pattern", default="checkpoint_*.pkl")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save_name", default="checkpoint_best_manual.pkl")
    parser.add_argument("--log_csv", default="experiments/vm/checkpoint_validation.csv")
    args = parser.parse_args()

    checkpoint_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, args.pattern)))
    checkpoint_paths = [
        path for path in checkpoint_paths
        if not os.path.basename(path).endswith(("best.pkl", "best_manual.pkl"))
    ]
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(f"No checkpoints found in {args.ckpt_dir} with pattern {args.pattern}")

    best_path = None
    best_loss = float("inf")
    rows = []
    for checkpoint_path in checkpoint_paths:
        loss = evaluate_checkpoint(checkpoint_path, args.task, args.seed)
        rows.append((checkpoint_path, loss))
        print(f"{checkpoint_path}: validation_loss={loss:.8f}")
        if loss < best_loss:
            best_loss = loss
            best_path = checkpoint_path

    assert best_path is not None
    output_path = os.path.join(args.ckpt_dir, args.save_name)
    shutil.copyfile(best_path, output_path)

    if args.log_csv:
        log_dir = os.path.dirname(args.log_csv)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(args.log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["checkpoint", "validation_loss"])
            for checkpoint_path, loss in rows:
                writer.writerow([checkpoint_path, f"{loss:.8f}"])

    print("\nBest checkpoint")
    print(f"path: {best_path}")
    print(f"validation_loss: {best_loss:.8f}")
    print(f"copied_to: {output_path}")
    if args.log_csv:
        print(f"log_csv: {args.log_csv}")


if __name__ == "__main__":
    main()
