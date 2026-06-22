吧 # TTA (Test-Time Augmentation) 优化方案

## 概述

本分支实现了基于 Test-Time Augmentation 的点云去噪增强方案，通过旋转增强、迭代去噪和双边滤波后处理，在不重新训练模型的情况下提升推理质量。

## 实现方法

### 1. 核心策略

**多旋转增强 (Rotation Augmentation)**
- 对输入点云进行多角度旋转
- 每个旋转角度独立推理
- 旋转结果平均融合

**迭代去噪 (Iterative Denoising)**
- 在每个旋转视角下进行多次迭代去噪
- 使用残差更新策略（alpha衰减）
- 逐步细化去噪结果

**双边滤波后处理 (Bilateral Filtering)**
- 基于空间距离和点云差异的加权平均
- 保留几何特征的同时平滑噪声
- 使用 cKDTree 加速近邻搜索

### 2. 实现文件

#### `predict_ultimate.py` - 验证版本 (62分)
```
配置: 3个旋转 × 2次迭代
- 旋转角度: 0°, 120°, 240°
- 迭代次数: 2次，alpha=0.5 (衰减0.9)
- 双边滤波: k=15, σ_s=0.01, σ_r=0.02
- 处理时间: ~1.7小时 (200样本)
- 验证分数: 62分
```

#### `predict_moderate_tta.py` - 适中版本 (运行中)
```
配置: 4个旋转 × 2次迭代
- 旋转角度: 0°, 90°, 180°, 270°
- 迭代次数: 2次
- 双边滤波: k=20, σ_s=0.01, σ_r=0.02
- 预计时间: ~5-6小时
- 预期分数: 65-70分
```

### 3. 工具脚本

#### `check_progress.sh` - 进度监控
自动显示：
- 完成样本数量和百分比
- GPU使用率和显存
- 预计剩余时间
- 最新处理样本

#### `eval_local.py` - 本地评估
快速评估去噪效果：
- Chamfer Distance (CD)
- Point-to-Surface Distance (P2S)
- 分数映射计算

#### `check_quality.py` - 质量检查
检测潜在问题：
- NaN 值
- 点云偏移
- 极端值

#### `kill_containers.sh` - 容器清理
清理卡住的Docker容器

## 使用方法

### 运行推理

```bash
# 适中版本（推荐）
CUDA_VISIBLE_DEVICES=4 scripts/run_in_runtime_container.sh python predict_moderate_tta.py

# 监控进度
watch -n 30 './check_progress.sh'
```

### 本地评估

```bash
# 快速评估
python eval_local.py --pred results_moderate_tta --mesh dataset_train

# 质量检查
python check_quality.py results_moderate_tta
```

## 性能对比

| 方案 | 旋转 | 迭代 | 时间 | 分数 | 说明 |
|------|------|------|------|------|------|
| Baseline | 0 | 1 | ~50m | 52 | 原始模型 |
| Ultimate | 3 | 2 | ~1.7h | 62 | 已验证 |
| Moderate | 4 | 2 | ~5-6h | 65-70 | 运行中 |

## 技术细节

### 旋转矩阵（Y轴）

```python
def rot_matrix_y(deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s], 
                     [0, 1, 0], 
                     [-s, 0, c]])
```

### 迭代去噪

```python
def iterative_denoise(model, pc, num_iterations=2, alpha=0.5):
    for i in range(num_iterations):
        pc_denoised = model.predict(pc)
        residual = pc_denoised - pc
        pc = pc + alpha * residual
        alpha *= 0.9  # 衰减
    return pc
```

### 双边滤波

```python
def bilateral_filter(points, k=20, sigma_s=0.01, sigma_r=0.02):
    # 空间权重: exp(-d^2 / (2*σ_s^2))
    # 范围权重: exp(-||p_i - p_j||^2 / (2*σ_r^2))
    # 加权平均: Σ(w_s * w_r * p_neighbor) / Σ(w_s * w_r)
```

## 优化空间

当前方案是**推理时优化**，无需重新训练。进一步提升需要：

1. **模型架构改进**（方案B）
   - EnhancedFeatureExtractor
   - 更深的网络
   - 注意力机制
   - 预期: 75-85分

2. **训练策略优化**
   - 更多数据增强
   - 困难样本挖掘
   - 损失函数改进

3. **TTA进一步优化**（边际收益递减）
   - 更多旋转角度（收益<1分）
   - 自适应参数（实现复杂）

## 提交记录

- **2026-06-22**: 实现 TTA 推理增强
  - 添加 predict_ultimate.py (62分验证)
  - 添加 predict_moderate_tta.py (65-70分预期)
  - 添加完整工具链（监控、评估、清理）
  - 文档化实现细节

## 作者

TTA优化: [你的名字]
基础代码: diudiu0515 & team
