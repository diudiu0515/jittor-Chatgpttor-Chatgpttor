"""Shared TTA helpers for Track 2 point cloud denoising."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree

from src.model.vm import patch_based_denoise


def parse_angles(value: str | Sequence[float]) -> List[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def rot_matrix_y(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def bilateral_filter(points: np.ndarray, k: int = 20, sigma_s: float = 0.01, sigma_r: float = 0.02) -> np.ndarray:
    if k <= 0:
        return points
    tree = cKDTree(points)
    filtered = np.empty_like(points)
    query_k = min(k + 1, len(points))

    for i, point in enumerate(points):
        dists, indices = tree.query(point, k=query_k)
        dists = np.atleast_1d(dists)[1:]
        indices = np.atleast_1d(indices)[1:]
        if len(indices) == 0:
            filtered[i] = point
            continue

        neighbors = points[indices]
        w_s = np.exp(-(dists ** 2) / (2.0 * sigma_s ** 2))
        diff_norm = np.linalg.norm(neighbors - point, axis=1)
        w_r = np.exp(-(diff_norm ** 2) / (2.0 * sigma_r ** 2))
        weights = w_s * w_r
        weights = weights / (weights.sum() + 1e-8)
        filtered[i] = (neighbors * weights[:, None]).sum(axis=0)
    return filtered


def denoise_once(model, pc_noisy: np.ndarray, patch_size: int, seed_k: int, seed_k_alpha: int) -> np.ndarray:
    with jt.no_grad():
        pc_denoised = patch_based_denoise(
            model=model,
            pcl_noisy=jt.array(pc_noisy.astype(np.float32)),
            patch_size=patch_size,
            seed_k=seed_k,
            seed_k_alpha=seed_k_alpha,
        )
    if pc_denoised is None:
        raise RuntimeError("patch_based_denoise returned None")
    return pc_denoised.detach().numpy().astype(np.float32)


def iterative_denoise(
    model,
    pc_noisy: np.ndarray,
    iterations: int = 2,
    alpha: float = 0.5,
    alpha_decay: float = 0.9,
    patch_size: int = 1000,
    seed_k: int = 6,
    seed_k_alpha: int = 1,
) -> np.ndarray:
    pc_current = pc_noisy.astype(np.float32).copy()
    curr_alpha = alpha
    for _ in range(max(1, iterations)):
        pc_denoised = denoise_once(model, pc_current, patch_size, seed_k, seed_k_alpha)
        residual = pc_denoised - pc_current
        pc_current = pc_current + curr_alpha * residual
        curr_alpha *= alpha_decay
    return pc_current.astype(np.float32)


def tta_denoise(
    model,
    pc_noisy: np.ndarray,
    angles: Iterable[float] = (0.0, 120.0, 240.0),
    iterations: int = 2,
    alpha: float = 0.5,
    alpha_decay: float = 0.9,
    patch_size: int = 1000,
    seed_k: int = 6,
    seed_k_alpha: int = 1,
    use_bilateral: bool = False,
    bilateral_k: int = 20,
    bilateral_sigma_s: float = 0.01,
    bilateral_sigma_r: float = 0.02,
) -> np.ndarray:
    pc_noisy = pc_noisy.astype(np.float32)
    results = []
    for angle in angles:
        rot = rot_matrix_y(angle)
        pc_rot = pc_noisy @ rot.T
        pc_denoised_rot = iterative_denoise(
            model=model,
            pc_noisy=pc_rot,
            iterations=iterations,
            alpha=alpha,
            alpha_decay=alpha_decay,
            patch_size=patch_size,
            seed_k=seed_k,
            seed_k_alpha=seed_k_alpha,
        )
        results.append(pc_denoised_rot @ rot)

    fused = np.mean(np.stack(results, axis=0), axis=0).astype(np.float32)
    if use_bilateral:
        fused = bilateral_filter(
            fused,
            k=bilateral_k,
            sigma_s=bilateral_sigma_s,
            sigma_r=bilateral_sigma_r,
        ).astype(np.float32)
    return fused