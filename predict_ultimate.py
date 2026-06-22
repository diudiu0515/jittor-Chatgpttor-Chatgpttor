"""Ultimate Pipeline: Iterative + TTA + Bilateral"""
import os
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
import jittor as jt
from omegaconf import OmegaConf

jt.flags.use_cuda = 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.vm import VelocityModule, patch_based_denoise


def rot_matrix_y(deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def bilateral_filter(points, k=20, sigma_s=0.01, sigma_r=0.02):
    tree = cKDTree(points)
    filtered = np.zeros_like(points)
    
    for i, point in enumerate(points):
        dists, indices = tree.query(point, k=k+1)
        dists, indices = dists[1:], indices[1:]
        
        if len(indices) == 0:
            filtered[i] = point
            continue
        
        neighbors = points[indices]
        w_s = np.exp(-dists**2 / (2 * sigma_s**2))
        diff_norm = np.linalg.norm(neighbors - point, axis=1)
        w_r = np.exp(-diff_norm**2 / (2 * sigma_r**2))
        weights = w_s * w_r
        weights = weights / (weights.sum() + 1e-8)
        filtered[i] = (neighbors * weights[:, np.newaxis]).sum(axis=0)
    
    return filtered


def iterative_denoise(model, pc_noisy, num_iterations=2, alpha=0.5):
    pc_current = pc_noisy.copy()
    for i in range(num_iterations):
        pc_tensor = jt.array(pc_current)
        with jt.no_grad():
            pc_denoised = patch_based_denoise(
                model=model, pcl_noisy=pc_tensor,
                patch_size=1000, seed_k=6, seed_k_alpha=1
            )
        pc_denoised = pc_denoised.detach().numpy()
        residual = pc_denoised - pc_current
        pc_current = pc_current + alpha * residual
        alpha = alpha * 0.9
    return pc_current


def main():
    vm_ckpt = "experiments/vm/checkpoint_best_manual.pkl"
    test_dir = "dataset_test_noisy/shapenet"
    output_dir = "results"  # Use existing results directory
    
    # TTA angles (reduced to 3 for speed)
    angles = [0, 120, 240]
    
    vm_cfg = OmegaConf.to_container(OmegaConf.load("configs/model/vm.yaml"))
    transform_cfg = OmegaConf.to_container(OmegaConf.load("configs/transform/vm.yaml"))
    
    print(f"Loading model: {vm_ckpt}")
    vm = VelocityModule(model_config=vm_cfg, transform_config=transform_cfg)
    vm.load(vm_ckpt)
    vm.eval()
    vm.set_predict(True)
    
    test_samples = []
    for root, dirs, files in os.walk(test_dir):
        if "noisy.npy" in files:
            test_samples.append(os.path.relpath(root, test_dir))
    test_samples.sort()
    
    print(f"Pipeline: Iterative (2x) + TTA (3 angles) + Bilateral")
    print(f"Samples: {len(test_samples)}")
    
    t0 = time.time()
    for idx, sample in enumerate(test_samples):
        noisy_path = os.path.join(test_dir, sample, "noisy.npy")
        pc_noisy = np.load(noisy_path).astype(np.float32)
        
        # Step 1: TTA with iterative refinement
        results = []
        for angle in angles:
            R = rot_matrix_y(angle)
            pc_rot = pc_noisy @ R.T
            
            # Iterative denoise
            pc_denoised_rot = iterative_denoise(vm, pc_rot, num_iterations=2)
            
            # Rotate back
            pc_back = pc_denoised_rot @ R
            results.append(pc_back)
        
        # Step 2: Fusion
        fused = results[0].copy()
        for j in range(1, len(results)):
            tree = cKDTree(results[0])
            _, nn = tree.query(results[j], k=1)
            fused += results[j][nn]
        fused /= len(results)
        
        # Step 3: Bilateral filter
        fused = bilateral_filter(fused, k=20, sigma_s=0.01, sigma_r=0.02)
        
        # Save
        out_dir = os.path.join(output_dir, "dataset_test_noisy", "shapenet", sample)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "denoised.npy"), fused.astype(np.float32))
        
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(test_samples) - idx - 1)
            print(f"  {idx+1}/{len(test_samples)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    
    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.0f}s ({elapsed/len(test_samples):.1f}s per sample)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
