#!/usr/bin/env python
"""快速验证脚本：仅输出平均分数"""
import argparse
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
        vertices=vertices, faces=faces, num_samples=num_samples, num_vertex_samples=1024)
    pc_clean, center, scale = normalize_to_unit_sphere(pc_clean)
    if scale >= 1e-12:
        vertices = (vertices - center) / scale
    noise_std = np.random.uniform(noise_std_min, noise_std_max)
    noise = np.random.laplace(0, noise_std, size=pc_clean.shape)
    pc_noisy = pc_clean + noise
    return pc_clean.astype(np.float32), pc_noisy.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    np.random.seed(123)
    random.seed(123)
    jt.set_global_seed(123)

    data_config = load_yaml("configs/data/train.yaml")
    transform_config = load_yaml("configs/transform/vm.yaml")
    model_config = load_yaml("configs/model/vm.yaml")
    
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

    cd_scores = []
    with jt.no_grad():
        for rel_path in tqdm(paths, desc="Validating"):
            mesh_path = os.path.join(dataset_dir, rel_path, data_name)
            pc_clean, pc_noisy = make_validation_sample(mesh_path, num_samples, noise_std_min, noise_std_max)
            pc_pred = patch_based_denoise(model=model, pcl_noisy=jt.array(pc_noisy), 
                                         patch_size=1000, seed_k=6, seed_k_alpha=1)
            pc_pred = pc_pred.detach().numpy().astype(np.float64)
            
            cd_pred = chamfer_distance(pc_pred, pc_clean)
            cd_noisy = chamfer_distance(pc_noisy, pc_clean)
            cd_score = metric_to_score(cd_pred, cd_noisy)
            cd_scores.append(cd_score)

    mean_score = float(np.mean(cd_scores))
    print(f"\n{'='*60}")
    print(f"Validation Results (Chamfer Distance only)")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {len(cd_scores)}")
    print(f"Mean CD Score: {mean_score:.2f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
