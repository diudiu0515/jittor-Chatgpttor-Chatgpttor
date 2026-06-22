import argparse
import csv
import os
import random

import jittor as jt
import numpy as np
import trimesh
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.data.utils import sample_vertex_groups
from src.model.parse import get_model
from src.model.vm import patch_based_denoise


jt.flags.use_cuda = 1


try:
    import point_cloud_utils as pcu
    HAS_PCU = True
except ImportError:
    HAS_PCU = False


def load_yaml(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def normalize_to_unit_sphere(pc):
    center = (pc.max(axis=0) + pc.min(axis=0)) / 2.0
    pc = pc - center
    scale = np.sqrt((pc ** 2).sum(axis=1)).max()
    if scale < 1e-12:
        return pc, center, scale
    return pc / scale, center, scale


def chamfer_distance(pc_a, pc_b):
    pc_b, center, scale = normalize_to_unit_sphere(pc_b)
    if scale < 1e-12:
        return 0.0
    pc_a = (pc_a - center) / scale
    dist_a2b, _ = cKDTree(pc_b).query(pc_a, k=1)
    dist_b2a, _ = cKDTree(pc_a).query(pc_b, k=1)
    return float((dist_a2b ** 2).mean() + (dist_b2a ** 2).mean())


def point_to_surface_distance(pc, mesh_v, mesh_f, normalize_ref_pc):
    center = (normalize_ref_pc.max(axis=0) + normalize_ref_pc.min(axis=0)) / 2.0
    ref_centered = normalize_ref_pc - center
    scale = np.sqrt((ref_centered ** 2).sum(axis=1)).max()
    if scale < 1e-12:
        return 0.0
    pc = (pc - center) / scale
    mesh_v = (mesh_v - center) / scale

    if HAS_PCU:
        dists, _, _ = pcu.closest_points_on_mesh(
            pc.astype(np.float32), mesh_v.astype(np.float32), mesh_f.astype(np.int32)
        )
        return float((dists ** 2).mean())

    dist, _ = cKDTree(mesh_v).query(pc, k=1)
    return float((dist ** 2).mean())


def metric_to_score(pred, noisy):
    if noisy < 1e-15:
        return 100.0 if pred < 1e-15 else 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - pred / noisy)))


def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)


def make_validation_sample(mesh_path, num_samples, noise_std_min, noise_std_max):
    vertices, faces = load_mesh(mesh_path)
    pc_clean, _, _, _ = sample_vertex_groups(
        vertices=vertices,
        faces=faces,
        num_samples=num_samples,
        num_vertex_samples=1024,
    )
    pc_clean, center, scale = normalize_to_unit_sphere(pc_clean)
    if scale >= 1e-12:
        vertices = (vertices - center) / scale
    noise_std = np.random.uniform(noise_std_min, noise_std_max)
    noise = np.random.laplace(0, noise_std, size=pc_clean.shape)
    pc_noisy = pc_clean + noise
    return vertices, faces, pc_clean.astype(np.float32), pc_noisy.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="configs/task/train_vm.yaml")
    parser.add_argument("--data_config", default="configs/data/train.yaml")
    parser.add_argument("--transform_config", default="configs/transform/vm.yaml")
    parser.add_argument("--model_config", default="configs/model/vm.yaml")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_csv", default="experiments/vm/validation_metrics.csv")
    parser.add_argument("--no_csv", action="store_true", help="Skip CSV output, only print summary")
    args = parser.parse_args()

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_config = load_yaml(args.data_config)
    transform_config = load_yaml(args.transform_config)
    model_config = load_yaml(args.model_config)
    model = get_model(model_config=model_config, transform_config=transform_config)
    model.load(args.checkpoint)
    model.set_predict(True)
    model.eval()

    validate_cfg = data_config["validate_dataset"]["datapath"]
    dataset_dir = validate_cfg["input_dataset_dir"]
    data_name = validate_cfg["data_name"]
    list_path = validate_cfg["data_path"]["shapenet"][0][0]
    paths = [line.strip() for line in open(list_path, "r").readlines() if line.strip()]
    if args.limit > 0:
        paths = paths[:args.limit]

    sample_aug = transform_config["validate_transform"]["augments"][0]
    noise_aug = transform_config["validate_transform"]["augments"][2]
    num_samples = sample_aug["num_samples"]
    noise_std_min = noise_aug["noise_std_min"]
    noise_std_max = noise_aug["noise_std_max"]

    rows = []
    cd_scores = []
    p2s_scores = []
    with jt.no_grad():
        for rel_path in tqdm(paths, desc="validation metrics"):
            mesh_path = os.path.join(dataset_dir, rel_path, data_name)
            mesh_v, mesh_f, pc_clean, pc_noisy = make_validation_sample(
                mesh_path, num_samples, noise_std_min, noise_std_max
            )
            pc_pred = patch_based_denoise(
                model=model,
                pcl_noisy=jt.array(pc_noisy),
                patch_size=1000,
                seed_k=6,
                seed_k_alpha=1,
            )
            pc_pred = pc_pred.detach().numpy().astype(np.float64)

            cd_pred = chamfer_distance(pc_pred, pc_clean)
            cd_noisy = chamfer_distance(pc_noisy, pc_clean)
            cd_score = metric_to_score(cd_pred, cd_noisy)

            p2s_pred = point_to_surface_distance(pc_pred, mesh_v, mesh_f, pc_clean)
            p2s_noisy = point_to_surface_distance(pc_noisy, mesh_v, mesh_f, pc_clean)
            p2s_score = metric_to_score(p2s_pred, p2s_noisy)
            final_score = 0.5 * cd_score + 0.5 * p2s_score

            rows.append([
                rel_path,
                cd_pred,
                cd_noisy,
                cd_score,
                p2s_pred,
                p2s_noisy,
                p2s_score,
                final_score,
            ])
            cd_scores.append(cd_score)
            p2s_scores.append(p2s_score)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if not args.no_csv:
        try:
            with open(args.output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "sample",
                    "cd_pred",
                    "cd_noisy",
                    "cd_score",
                    "p2s_pred",
                    "p2s_noisy",
                    "p2s_score",
                    "final_score",
                ])
                writer.writerows(rows)
        except PermissionError:
            print(f"Warning: Cannot write to {args.output_csv}, skipping CSV output")

    mean_cd = float(np.mean(cd_scores)) if cd_scores else 0.0
    mean_p2s = float(np.mean(p2s_scores)) if p2s_scores else 0.0
    final_score = 0.5 * mean_cd + 0.5 * mean_p2s

    print(f"\n{'='*60}")
    print(f"Validation Results")
    print(f"{'='*60}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"samples: {len(rows)}")
    print(f"mean_cd_score: {mean_cd:.2f}")
    print(f"mean_p2s_score: {mean_p2s:.2f}")
    print(f"final_score: {final_score:.2f}")
    print(f"{'='*60}")
    if not args.no_csv:
        print(f"output_csv: {args.output_csv}")
    if not HAS_PCU:
        print("warning: point-cloud-utils not installed, P2S used vertex approximation")


if __name__ == "__main__":
    main()
