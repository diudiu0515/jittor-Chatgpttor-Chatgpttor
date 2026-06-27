from math import ceil
from typing import Dict, List

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree

from .feature import FeatureExtraction, Decoder, EnhancedFeatureExtractor
from .spec import ModelSpec

from ..data.asset import Asset

def get_random_indices(n, m):
    assert m < n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()

class VelocityModule(ModelSpec):
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        # geometry
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']
        
        # score-matching
        self.dsm_sigma = cfg['dsm_sigma']
        
        # networks
        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg['feat_embedding_dim']
        )
        
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )
    
    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        """
        pcl_noisy: (B, N, 3)
        pcl_clean: (B, N, 3)
        """
        B, N_noisy, d = pc_mix.shape
        
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)
        
        # Feature extraction
        feat = self.encoder(pc_mix)  # (B, N, F)
        F_dim = feat.shape[2]
        
        # gather
        feat = feat[:, pnt_idx, :]
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]
        
        # target
        grad_dir_t_target = pc_clean - pc_noisy
        
        # decoder
        pred_dir = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d) # type: ignore
        
        loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        
        return loss

    def _decay_factor(self, step: int, num_steps: int, decay: str):
        if decay == "none":
            return 1.0
        if decay == "linear":
            return 1.0 - float(step) / max(1, num_steps)
        raise ValueError(f"unsupported predict_step_decay: {decay}")

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=4, step_size: float=1.0, momentum: float=0.0, step_decay: str="none"):
        """
        pcl_noisy: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            velocity = pcl_next * 0.0
            for it in range(num_steps):
                feat = self.encoder(pcl_next)  # (B, N, F)
                F_dim = feat.shape[2]
                
                pred_dir = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)

                velocity = momentum * velocity + pred_dir
                step_scale = (step_size / num_steps) * self._decay_factor(it, num_steps, step_decay)
                pcl_next = pcl_next + step_scale * velocity
        return pcl_next, None
    
    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
        )
        return {"loss": loss}
    
    def execute(self, **kwargs) -> Dict: # type: ignore
        return self.training_step(**kwargs)
    
    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        
        num_steps = 1
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_next = pc_noisy
            for it in range(num_steps):
                pc_next = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=1000,
                    seed_k=6,
                    seed_k_alpha=1,
                )
            pc_denoised = pc_next.detach().numpy()
            res.append({"pc_denoised": pc_denoised})
        return res
    
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy": b.meta['pc_noisy'], # (num_patches, patch_size, 3)
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy, # (N, 3)
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res

def farthest_point_sampling(pcls, num_pnts):
    """
    pcls: (B, N, 3)
    return:
        sampled: (B, num_pnts, 3)
        indices: (B, num_pnts)
    """
    B, N, _ = pcls.shape
    sampled = []
    indices = []
    for b in range(B):
        pts = pcls[b]  # (N, 3)
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for i in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]  # (3,)
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = int(farthest.numpy())
        idx = jt.array(selected).int32()
        sampled.append(pts[idx][None, ...])
        indices.append(idx[None, ...])
    sampled = jt.concat(sampled, dim=0)
    indices = jt.concat(indices, dim=0)
    return sampled, indices

def knn_points(x, y, k):
    """
    x: (B, P, 3)
    y: (B, N, 3)
    return:
        dist: (B, P, k)
        idx:  (B, P, k)
        nn:   (B, P, k, 3)
    """
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    B = x.shape[0]
    nn = []
    for b in range(B):
        nn.append(y[b][idx[b]])
    nn = jt.stack(nn, dim=0)
    return dist_k, idx, nn

def farthest_point_sampling_np(points: np.ndarray, num_pnts: int) -> np.ndarray:
    """Deterministic NumPy FPS for inference-time patch centers."""
    n = points.shape[0]
    if num_pnts <= 0:
        raise ValueError("num_pnts must be positive")
    selected = np.empty(num_pnts, dtype=np.int64)
    dist = np.full(n, np.inf, dtype=np.float64)
    farthest = 0
    for i in range(num_pnts):
        selected[i] = farthest
        delta = points - points[farthest]
        dist = np.minimum(dist, np.einsum("ij,ij->i", delta, delta))
        farthest = int(np.argmax(dist))
    return selected

def build_patches_np(points: np.ndarray, patch_size: int, seed_k: int):
    n, d = points.shape
    num_patches = max(1, int(seed_k * n / patch_size))
    seed_idx = farthest_point_sampling_np(points, num_patches)
    seed_points = points[seed_idx]
    k = min(patch_size, n)
    dists, point_idxs = cKDTree(points).query(seed_points, k=k)
    if k == 1:
        dists = dists[:, None]
        point_idxs = point_idxs[:, None]
    patches = points[point_idxs]
    seed_expand = seed_points[:, None, :]
    patches_centered = patches - seed_expand
    norm = dists[:, -1:] + 1e-8
    patch_dists = dists / norm
    patch_weights = np.exp(-patch_dists).astype(np.float32)[..., None]
    return patches_centered.astype(np.float32), seed_expand.astype(np.float32), point_idxs.astype(np.int64), patch_weights

def patch_based_denoise(
    model: VelocityModule,
    pcl_noisy,
    patch_size=1000,
    seed_k=6,
    seed_k_alpha=1,
    patch_backend="numpy",
    patch_step=None,
    denoise_steps=4,
    step_size=1.0,
    momentum=0.0,
    step_decay="none",
    blend=1.0,
) -> jt.Var:
    """
    pcl_noisy: (N, 3)
    """
    assert len(pcl_noisy.shape) == 2
    
    N, d = pcl_noisy.shape
    original_pcl = pcl_noisy
    if patch_backend == "numpy":
        original_pcl_np = np.asarray(pcl_noisy.numpy(), dtype=np.float32)
        patches_np, seed_expand_np, point_idxs_np, patch_weights_np = build_patches_np(
            original_pcl_np,
            patch_size=patch_size,
            seed_k=seed_k,
        )
        patches = jt.array(patches_np)
        num_patches = patches_np.shape[0]
    else:
        num_patches = int(seed_k * N / patch_size)
        pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
        seed_pnts, seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
        patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)
        patches = patches[0]              # (P, M, 3)
        patch_dists = patch_dists[0]      # (P, M)
        point_idxs = point_idxs[0]        # (P, M)
        seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
        patches = patches - seed_expand
        patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
        all_dists = jt.ones((num_patches, N)) * 1e10
        for i in range(num_patches):
            all_dists[i][point_idxs[i]] = patch_dists[i]
        weights = jt.exp(-all_dists)
        best_weights_idx, _ = jt.argmax(weights, dim=0)
        patch_weights_np = jt.exp(-patch_dists).unsqueeze(-1).numpy()
        seed_expand_np = seed_expand.numpy()
        point_idxs_np = point_idxs.numpy()
        original_pcl_np = original_pcl.numpy()
    patches_denoised = []
    
    i = 0
    if patch_step is None:
        patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    else:
        patch_step = int(patch_step)
    assert patch_step > 0
    while i < num_patches:
        curr = patches[i:i+patch_step]
        try:
            out, _ = model.denoise_langevin_dynamics(
                curr,
                num_steps=denoise_steps,
                step_size=step_size,
                momentum=momentum,
                step_decay=step_decay,
            )
        except Exception as e:
            print("Denoise error:", e)
            return None
        patches_denoised.append(out.detach().numpy())
        i += patch_step
    
    patches_denoised_np = np.concatenate(patches_denoised, axis=0)
    patches_denoised_np = patches_denoised_np + seed_expand_np
    pcl_sum = np.zeros((N, d), dtype=np.float32)
    weight_sum = np.zeros((N, 1), dtype=np.float32)
    for patch_id in range(num_patches):
        idx = point_idxs_np[patch_id]
        weight = patch_weights_np[patch_id]
        np.add.at(pcl_sum, idx, patches_denoised_np[patch_id] * weight)
        np.add.at(weight_sum, idx, weight)
    pcl_out = pcl_sum / (weight_sum + 1e-8)
    missing = weight_sum[:, 0] <= 1e-8
    if np.any(missing):
        pcl_out[missing] = original_pcl_np[missing]
    if blend != 1.0:
        pcl_out = original_pcl_np + blend * (pcl_out - original_pcl_np)
    pcl_out = jt.array(pcl_out)
    assert pcl_out.shape[0] == N
    return pcl_out

class EnhancedVelocityModule(VelocityModule):

    def __init__(self, model_config, transform_config):
        ModelSpec.__init__(self, model_config, transform_config)

        cfg = self.model_config
        self.frame_knn_list = cfg.get('frame_knn_list', [16, 32, 48])
        self.num_train_points = cfg['num_train_points']
        self.dsm_sigma = cfg['dsm_sigma']

        self.encoder = EnhancedFeatureExtractor(
            k_list=self.frame_knn_list,
            input_dim=3,
            embedding_dim=cfg['feat_embedding_dim'],
        )

        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=4, step_size: float=1.05, momentum: float=0.0, step_decay: str="none"):
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            velocity = pcl_next * 0.0
            for it in range(num_steps):
                feat = self.encoder(pcl_next)
                F_dim = feat.shape[2]

                pred_dir = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)

                velocity = momentum * velocity + pred_dir
                step_scale = (step_size / num_steps) * self._decay_factor(it, num_steps, step_decay)
                pcl_next = pcl_next + step_scale * velocity
        return pcl_next, None

class EnhancedVelocityModuleCDRepulsion(EnhancedVelocityModule):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.cd_weight = cfg.get('cd_weight', 0.0)
        self.repulsion_weight = cfg.get('repulsion_weight', 0.0)
        self.displacement_weight = cfg.get('displacement_weight', 0.0)
        self.num_cd_points = cfg.get('num_cd_points', self.num_train_points)
        self.centroid_weight = cfg.get('centroid_weight', 0.1)
        self.covariance_weight = cfg.get('covariance_weight', 0.05)

    def chamfer_loss(self, pc_pred, pc_clean):
        dist = ((pc_pred.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2).sum(-1)
        pred_to_clean = jt.min(dist, dim=2)
        clean_to_pred = jt.min(dist, dim=1)
        return pred_to_clean.mean() + clean_to_pred.mean()

    def repulsion_loss(self, pc_pred):
        dist = ((pc_pred.unsqueeze(2) - pc_pred.unsqueeze(1)) ** 2).sum(-1)
        B, N, _ = dist.shape
        eye = jt.array(np.eye(N, dtype=np.float32)).reshape(1, N, N).broadcast((B, N, N))
        radius2 = self.repulsion_radius ** 2
        offdiag = 1.0 - eye
        repel = jt.exp(-dist / max(radius2, 1e-8)) * offdiag
        return repel.sum() / (offdiag.sum() * B + 1e-8)

    def distribution_loss(self, pc_pred, pc_clean):
        pred_center = pc_pred.mean(dim=1, keepdims=True)
        clean_center = pc_clean.mean(dim=1, keepdims=True)
        pred_delta = pc_pred - pred_center
        clean_delta = pc_clean - clean_center

        centroid = ((pred_center - clean_center) ** 2).sum(dim=-1).mean()
        pred_cov = jt.matmul(pred_delta.transpose(0, 2, 1), pred_delta) / pc_pred.shape[1]
        clean_cov = jt.matmul(clean_delta.transpose(0, 2, 1), clean_delta) / pc_clean.shape[1]
        covariance = ((pred_cov - clean_cov) ** 2).mean()
        return self.centroid_weight * centroid + self.covariance_weight * covariance

    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        B, N_noisy, d = pc_mix.shape

        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        feat = self.encoder(pc_mix)
        F_dim = feat.shape[2]

        feat = feat[:, pnt_idx, :]
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]

        grad_dir_t_target = pc_clean - pc_noisy
        pred_dir = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d) # type: ignore

        mse_loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        pc_pred = pc_noisy + pred_dir
        loss = mse_loss

        if self.cd_weight > 0:
            cd_points = min(self.num_cd_points, len(pnt_idx))
            if cd_points < len(pnt_idx):
                cd_idx = get_random_indices(len(pnt_idx), cd_points)
                pc_pred_cd = pc_pred[:, cd_idx, :]
                pc_clean_cd = pc_clean[:, cd_idx, :]
            else:
                pc_pred_cd = pc_pred
                pc_clean_cd = pc_clean
            loss = loss + self.cd_weight * self.chamfer_loss(pc_pred_cd, pc_clean_cd)

        if self.repulsion_weight > 0:
            loss = loss + self.repulsion_weight * self.repulsion_loss(pc_pred)

        if self.centroid_weight > 0 or self.covariance_weight > 0:
            loss = loss + self.distribution_loss(pc_pred, pc_clean)

        if self.displacement_weight > 0:
            loss = loss + self.displacement_weight * (pred_dir ** 2).sum(dim=-1).mean()

        return loss

class EnhancedVelocityModuleStructureLoss(EnhancedVelocityModule):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.edge_weight = cfg.get('edge_weight', 0.08)
        self.centroid_weight = cfg.get('centroid_weight', 0.02)
        self.displacement_weight = cfg.get('displacement_weight', 0.0005)
        self.edge_stride = cfg.get('edge_stride', 4)

    def edge_length_loss(self, pc_pred, pc_clean):
        stride = max(1, self.edge_stride)
        pred_a = pc_pred[:, :-stride, :]
        pred_b = pc_pred[:, stride:, :]
        clean_a = pc_clean[:, :-stride, :]
        clean_b = pc_clean[:, stride:, :]
        pred_len = ((pred_a - pred_b) ** 2).sum(dim=-1)
        clean_len = ((clean_a - clean_b) ** 2).sum(dim=-1)
        return ((pred_len - clean_len) ** 2).mean()

    def centroid_loss(self, pc_pred, pc_clean):
        pred_center = pc_pred.mean(dim=1)
        clean_center = pc_clean.mean(dim=1)
        return ((pred_center - clean_center) ** 2).sum(dim=-1).mean()

    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        B, N_noisy, d = pc_mix.shape
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        feat = self.encoder(pc_mix)
        F_dim = feat.shape[2]

        feat = feat[:, pnt_idx, :]
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]

        grad_dir_t_target = pc_clean - pc_noisy
        pred_dir = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d) # type: ignore

        mse_loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        pc_pred = pc_noisy + pred_dir
        loss = mse_loss

        if self.edge_weight > 0:
            loss = loss + self.edge_weight * self.edge_length_loss(pc_pred, pc_clean)
        if self.centroid_weight > 0:
            loss = loss + self.centroid_weight * self.centroid_loss(pc_pred, pc_clean)
        if self.displacement_weight > 0:
            loss = loss + self.displacement_weight * (pred_dir ** 2).sum(dim=-1).mean()
        return loss

class EnhancedVelocityModuleScaleOffsetLoss(EnhancedVelocityModuleCDRepulsion):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.scale_weight = cfg.get('scale_weight', 0.0)
        self.scale_stride = cfg.get('scale_stride', 4)
        self.offset_weight = cfg.get('offset_weight', 0.0)
        self.offset_max_ratio = cfg.get('offset_max_ratio', 1.2)

    def scale_consistency_loss(self, pc_pred, pc_noisy):
        stride = max(1, self.scale_stride)
        pred_delta = pc_pred[:, stride:, :] - pc_pred[:, :-stride, :]
        noisy_delta = pc_noisy[:, stride:, :] - pc_noisy[:, :-stride, :]
        pred_len = jt.sqrt((pred_delta ** 2).sum(dim=-1) + 1e-8)
        noisy_len = jt.sqrt((noisy_delta ** 2).sum(dim=-1) + 1e-8)
        ratio = pred_len / (noisy_len + 1e-8)
        lower = jt.maximum(0.0, 0.96 - ratio)
        upper = jt.maximum(0.0, ratio - 1.04)
        return ((lower + upper) ** 2).mean()

    def offset_limit_loss(self, pc_noisy, pred_dir):
        stride = max(1, self.scale_stride)
        noisy_delta = pc_noisy[:, stride:, :] - pc_noisy[:, :-stride, :]
        local_scale = jt.sqrt((noisy_delta ** 2).sum(dim=-1) + 1e-8).mean(dim=1, keepdims=True)
        offset_norm = jt.sqrt((pred_dir ** 2).sum(dim=-1) + 1e-8)
        limit = local_scale * self.offset_max_ratio
        excess = jt.maximum(0.0, offset_norm - limit)
        return (excess ** 2).mean()

    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        B, N_noisy, d = pc_mix.shape
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        feat = self.encoder(pc_mix)
        F_dim = feat.shape[2]

        feat = feat[:, pnt_idx, :]
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]

        grad_dir_t_target = pc_clean - pc_noisy
        pred_dir = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d) # type: ignore

        mse_loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        pc_pred = pc_noisy + pred_dir
        loss = mse_loss

        if self.cd_weight > 0:
            cd_points = min(self.num_cd_points, len(pnt_idx))
            if cd_points < len(pnt_idx):
                cd_idx = get_random_indices(len(pnt_idx), cd_points)
                pc_pred_cd = pc_pred[:, cd_idx, :]
                pc_clean_cd = pc_clean[:, cd_idx, :]
            else:
                pc_pred_cd = pc_pred
                pc_clean_cd = pc_clean
            loss = loss + self.cd_weight * self.chamfer_loss(pc_pred_cd, pc_clean_cd)

        if self.centroid_weight > 0 or self.covariance_weight > 0:
            loss = loss + self.distribution_loss(pc_pred, pc_clean)
        if self.displacement_weight > 0:
            loss = loss + self.displacement_weight * (pred_dir ** 2).sum(dim=-1).mean()
        if self.scale_weight > 0:
            loss = loss + self.scale_weight * self.scale_consistency_loss(pc_pred, pc_noisy)
        if self.offset_weight > 0:
            loss = loss + self.offset_weight * self.offset_limit_loss(pc_noisy, pred_dir)
        return loss
