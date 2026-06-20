#!/usr/bin/env python
"""评测脚本：点云降噪赛题评测

计算两个指标：
  1. Chamfer Distance (CD) — 降噪点云与干净点云之间的双向最近点距离
  2. Point-to-Surface (P2S) — 降噪点云到原始网格表面的单向距离

百分制评分公式（所有样本全局平均，不分类别）：

  对每个测试样本 i：
    CD_noisy_i  = CD(noisy_i, clean_i)
    CD_pred_i   = CD(denoised_i, clean_i)
    P2S_noisy_i = P2S(noisy_i, mesh_i)
    P2S_pred_i  = P2S(denoised_i, mesh_i)

    cd_score_i  = clamp(100 × (1 - CD_pred_i / CD_noisy_i), 0, 100)
    p2s_score_i = clamp(100 × (1 - P2S_pred_i / P2S_noisy_i), 0, 100)

  最终得分 = 0.5 × mean(cd_score) + 0.5 × mean(p2s_score)

Usage:
    python evaluate.py \\
        --pred_dir ./results \\
        --gt_dir ./test_gt \\
        --noisy_dir ./test_noisy \\
        --mesh_dir ./dataset_clean \\
        [--verbose]

目录结构：
    pred_dir/  <category>/<model_id>/denoised.xyz
    gt_dir/    <category>/<model_id>/clean.xyz
    noisy_dir/ <category>/<model_id>/noisy.xyz
    mesh_dir/  <category>/<model_id>/models/model_normalized.obj  (可选，用于P2S)
"""

import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from functools import partial

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='point_cloud_utils')

import numpy as np

try:
    import point_cloud_utils as pcu
    HAS_PCU = True
except ImportError:
    HAS_PCU = False

from scipy.spatial import cKDTree


# ======================== IO ========================

def load_pointcloud(path):
    """加载点云文件，支持 .npy 和 .xyz 格式。"""
    if path.endswith('.npy'):
        return np.load(path).astype(np.float64)
    pts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(pts, dtype=np.float64)


def load_mesh_vf(path):
    """加载网格文件，返回 (vertices, faces) numpy arrays。优先使用 pcu，退而用 trimesh。"""
    if not os.path.exists(path):
        return None, None
    if HAS_PCU:
        v, f = pcu.load_mesh_vf(path)
        return v.astype(np.float64), f.astype(np.int32)
    try:
        import trimesh
        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        return np.array(mesh.vertices, dtype=np.float64), np.array(mesh.faces, dtype=np.int32)
    except ImportError:
        return None, None


# ======================== 归一化 ========================

def normalize_to_unit_sphere(pc):
    """归一化到单位球，返回 (normalized_pc, center, scale)。"""
    center = (pc.max(axis=0) + pc.min(axis=0)) / 2.0
    pc_centered = pc - center
    scale = np.sqrt((pc_centered ** 2).sum(axis=1)).max()
    if scale < 1e-12:
        return pc_centered, center, scale
    return pc_centered / scale, center, scale


# ======================== Chamfer Distance ========================

def chamfer_distance(pc_a, pc_b, normalize=True):
    """
    CD(A, B) = mean_{a∈A} min_{b∈B} ||a-b||² + mean_{b∈B} min_{a∈A} ||b-a||²

    normalize=True 时，以 B (参考) 归一化到单位球，A 施加相同变换。
    使用 scipy cKDTree 加速最近邻查询（O(N log M)）。
    """
    if normalize:
        pc_b, center, scale = normalize_to_unit_sphere(pc_b)
        if scale < 1e-12:
            return 0.0
        pc_a = (pc_a - center) / scale

    tree_b = cKDTree(pc_b)
    dist_a2b, _ = tree_b.query(pc_a, k=1)  # 欧式距离

    tree_a = cKDTree(pc_a)
    dist_b2a, _ = tree_a.query(pc_b, k=1)

    return (dist_a2b ** 2).mean() + (dist_b2a ** 2).mean()


# ======================== Point-to-Surface ========================

def point_to_surface_distance(pc, mesh_v, mesh_f, normalize_ref_pc=None):
    """
    P2S: 每个点到网格表面最近距离²的均值。
    使用 point_cloud_utils (pcu) 的 closest_points_on_mesh（基于 BVH 加速）。
    如果 pcu 不可用，回退到 scipy cKDTree 近似（用网格顶点代替表面）。

    返回 mean(||p - closest_on_surface(p)||²)
    """
    if mesh_v is None or mesh_f is None:
        return None

    vertices = mesh_v.copy()

    if normalize_ref_pc is not None:
        center = (normalize_ref_pc.max(axis=0) + normalize_ref_pc.min(axis=0)) / 2.0
        centered = normalize_ref_pc - center
        scale = np.sqrt((centered ** 2).sum(axis=1)).max()
        if scale < 1e-12:
            return 0.0
        pc = (pc - center) / scale
        vertices = (vertices - center) / scale

    if HAS_PCU:
        # pcu.closest_points_on_mesh 返回 (euclidean_distances, face_ids, barycentric_coords)
        dists, _, _ = pcu.closest_points_on_mesh(
            pc.astype(np.float32), vertices.astype(np.float32), mesh_f
        )
        return float((dists ** 2).mean())
    else:
        # 回退：用网格顶点做近似 P2S（不够精确，但无需额外依赖）
        tree = cKDTree(vertices)
        dists, _ = tree.query(pc, k=1)
        return float((dists ** 2).mean())


# ======================== 评分映射 ========================

def metric_to_score(val_pred, val_noisy):
    """score = clamp(100 * (1 - val_pred / val_noisy), 0, 100)"""
    if val_noisy < 1e-15:
        return 100.0 if val_pred < 1e-15 else 0.0
    score = 100.0 * (1.0 - val_pred / val_noisy)
    return max(0.0, min(100.0, score))


# ======================== 文件扫描 ========================

def find_samples(base_dir, filename):
    """扫描目录，返回 {relative_key: filepath}。"""
    samples = {}
    for path in sorted(glob.glob(os.path.join(base_dir, '**', filename), recursive=True)):
        rel = os.path.relpath(os.path.dirname(path), base_dir)
        samples[rel] = path
    return samples


def find_meshes(mesh_dir, data_name='models/model_normalized.obj'):
    """扫描网格目录。"""
    meshes = {}
    for path in sorted(glob.glob(os.path.join(mesh_dir, '**', data_name), recursive=True)):
        # key = relative path from mesh_dir to the parent of data_name
        parts = data_name.split('/')
        depth = len(parts)
        p = path
        for _ in range(depth):
            p = os.path.dirname(p)
        rel = os.path.relpath(p, mesh_dir)
        meshes[rel] = path
    return meshes


# ======================== 单样本评测（用于并行） ========================

def evaluate_single(args_tuple):
    """评测单个样本，返回 (key, cd_pred, cd_noisy, cd_score, p2s_pred, p2s_noisy, p2s_score)"""
    key, pred_path, gt_path, noisy_path, mesh_path = args_tuple

    pc_pred = load_pointcloud(pred_path)
    pc_gt = load_pointcloud(gt_path)
    pc_noisy = load_pointcloud(noisy_path)

    cd_pred = chamfer_distance(pc_pred, pc_gt, normalize=True)
    cd_noisy = chamfer_distance(pc_noisy, pc_gt, normalize=True)
    cd_s = metric_to_score(cd_pred, cd_noisy)

    p2s_pred_val = None
    p2s_noisy_val = None
    p2s_s = None
    if mesh_path is not None:
        mv, mf = load_mesh_vf(mesh_path)
        if mv is not None:
            p2s_pred_val = point_to_surface_distance(pc_pred, mv, mf, normalize_ref_pc=pc_gt)
            p2s_noisy_val = point_to_surface_distance(pc_noisy, mv, mf, normalize_ref_pc=pc_gt)
            if p2s_pred_val is not None and p2s_noisy_val is not None:
                p2s_s = metric_to_score(p2s_pred_val, p2s_noisy_val)

    return (key, cd_pred, cd_noisy, cd_s, p2s_pred_val, p2s_noisy_val, p2s_s)


# ======================== 主流程 ========================

def main():
    parser = argparse.ArgumentParser(description='点云降噪评测脚本')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='选手提交的降噪结果目录')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='真实干净点云目录')
    parser.add_argument('--noisy_dir', type=str, required=True,
                        help='含噪点云目录')
    parser.add_argument('--mesh_dir', type=str, default='',
                        help='原始网格目录（用于 P2S，可选）')
    parser.add_argument('--mesh_data_name', type=str, default='models/model_normalized.obj',
                        help='网格文件相对于 model_id 的路径')
    parser.add_argument('--pred_filename', type=str, default='denoised.npy')
    parser.add_argument('--gt_filename', type=str, default='clean.npy')
    parser.add_argument('--noisy_filename', type=str, default='noisy.npy')
    parser.add_argument('--workers', type=int, default=0,
                        help='并行进程数 (0=自动检测 CPU 核数)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    use_p2s = bool(args.mesh_dir)
    if use_p2s and not HAS_PCU:
        print("提示: point-cloud-utils 未安装，P2S 将用顶点近似。pip install point-cloud-utils 以获得精确结果。")

    n_workers = args.workers if args.workers > 0 else min(cpu_count(), 16)

    # 扫描文件
    pred_samples = find_samples(args.pred_dir, args.pred_filename)
    gt_samples = find_samples(args.gt_dir, args.gt_filename)
    noisy_samples = find_samples(args.noisy_dir, args.noisy_filename)
    mesh_samples = find_meshes(args.mesh_dir, args.mesh_data_name) if use_p2s else {}

    print(f"加速后端: CD=scipy.cKDTree, P2S={'pcu (BVH)' if HAS_PCU else 'cKDTree (vertex approx)'}, workers={n_workers}")

    common_keys = sorted(set(pred_samples.keys()) & set(gt_samples.keys()) & set(noisy_samples.keys()))

    if not common_keys:
        print("错误：未找到匹配的测试样本。")
        print(f"  pred_dir: {len(pred_samples)} 个, gt_dir: {len(gt_samples)} 个, noisy_dir: {len(noisy_samples)} 个")
        sys.exit(1)

    missing_pred = set(gt_samples.keys()) - set(pred_samples.keys())
    if missing_pred:
        print(f"警告：{len(missing_pred)} 个测试样本缺少预测结果，将记为 0 分。")

    # 构建任务列表
    tasks = []
    for key in common_keys:
        mesh_path = mesh_samples.get(key) if use_p2s else None
        tasks.append((key, pred_samples[key], gt_samples[key], noisy_samples[key], mesh_path))

    print(f"开始评测 {len(tasks)} 个样本...")
    t0 = time.time()

    # 并行评测
    if n_workers > 1 and len(tasks) > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(evaluate_single, tasks)
    else:
        results = [evaluate_single(t) for t in tasks]

    # 汇总
    cd_scores = []
    p2s_scores = []
    cd_preds = []
    cd_noisys = []
    p2s_preds = []
    p2s_noisys = []

    for key, cd_pred, cd_noisy, cd_s, p2s_pred, p2s_noisy, p2s_s in results:
        cd_scores.append(cd_s)
        cd_preds.append(cd_pred)
        cd_noisys.append(cd_noisy)
        if p2s_s is not None:
            p2s_scores.append(p2s_s)
            p2s_preds.append(p2s_pred)
            p2s_noisys.append(p2s_noisy)
        if args.verbose:
            msg = f"  {key}  CD_score={cd_s:.2f}"
            if p2s_s is not None:
                msg += f"  P2S_score={p2s_s:.2f}"
            print(msg)

    # 为缺失样本记 0 分
    for key in missing_pred:
        cd_scores.append(0.0)
        if use_p2s:
            p2s_scores.append(0.0)

    total_samples = len(common_keys) + len(missing_pred)
    mean_cd_score = np.mean(cd_scores) if cd_scores else 0.0

    has_p2s = len(p2s_scores) > 0
    mean_p2s_score = np.mean(p2s_scores) if has_p2s else 0.0

    if has_p2s:
        final_score = 0.5 * mean_cd_score + 0.5 * mean_p2s_score
    else:
        final_score = mean_cd_score

    elapsed = time.time() - t0

    # 输出
    print("\n" + "=" * 65)
    print("  点云降噪评测结果")
    print("=" * 65)
    print(f"  评测样本总数:       {total_samples}")
    print(f"  有效预测数:         {len(common_keys)}")
    print(f"  缺失预测数:         {len(missing_pred)}")
    print(f"  并行进程数:         {n_workers}")
    print(f"  评测耗时:           {elapsed:.1f}s")
    print("-" * 65)
    print(f"  平均 CD_pred:       {np.mean(cd_preds):.8f}" if cd_preds else "")
    print(f"  平均 CD_noisy:      {np.mean(cd_noisys):.8f}" if cd_noisys else "")
    print(f"  CD 得分:            {mean_cd_score:.2f} / 100.00")
    if has_p2s:
        print(f"  平均 P2S_pred:      {np.mean(p2s_preds):.8f}")
        print(f"  平均 P2S_noisy:     {np.mean(p2s_noisys):.8f}")
        print(f"  P2S 得分:           {mean_p2s_score:.2f} / 100.00")
        print("-" * 65)
        print(f"  最终得分 (0.5×CD + 0.5×P2S):  {final_score:.2f} / 100.00")
    else:
        print("-" * 65)
        print(f"  最终得分 (CD):      {final_score:.2f} / 100.00")
        if not use_p2s:
            print("  (未提供 mesh_dir，P2S 指标未计算)")
    print("=" * 65)

    return final_score


if __name__ == '__main__':
    score = main()
