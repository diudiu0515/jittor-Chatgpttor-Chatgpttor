# 赛道二增强模型训练、筛选与本地评测说明

本文档记录 `track2_fresh_opt` 工作区当前可复用的完整流程：在 Jittor Docker 镜像中训练增强模型，在本地验证集上筛选最佳 checkpoint，以及用本地 CD/P2S 指标做评测。

## 0. 基本约定

所有命令默认从工作区根目录执行：

```bash
cd /villa/mwq24-srt/track2_fresh_opt
```

Jittor 相关训练、验证、推理都通过容器入口运行，不直接使用宿主机 Python/Jittor：

```bash
scripts/run_in_runtime_container.sh <command>
```

当前容器入口的默认设置在 `scripts/run_in_runtime_container.sh` 中：

- 镜像：`jittor-denoise:runtime`
- 工作目录挂载到容器内：`/workspace`
- 默认 GPU：由 `CUDA_VISIBLE_DEVICES` 控制，未设置时脚本默认用 `4`
- Jittor cache：挂载到 `../repo_check/.jittor_cache`

如需指定 GPU，可以这样运行：

```bash
CUDA_VISIBLE_DEVICES=4 scripts/run_in_runtime_container.sh python run.py --task configs/task/debug.yaml
```

## 1. 在镜像里训练增强模型

当前增强模型训练任务是：

- task 配置：`configs/task/train_enhanced_vm.yaml`
- data 配置：`configs/data/train_enhanced.yaml`
- model 配置：`configs/model/enhanced_vm.yaml`
- 输出目录：`experiments/enhanced_vm/`
- 日志目录：`logs/`

推荐直接用封装脚本训练：

```bash
bash scripts/train_enhanced_container.sh 123
```

其中 `123` 是随机种子。脚本内部实际执行：

```bash
scripts/run_in_runtime_container.sh \
  python run.py --task configs/task/train_enhanced_vm.yaml --seed 123
```

训练日志会写到：

```text
logs/train_enhanced_container_seed_123.log
```

checkpoint 会保存在：

```text
experiments/enhanced_vm/checkpoint_0.pkl
experiments/enhanced_vm/checkpoint_1.pkl
...
experiments/enhanced_vm/checkpoint_99.pkl
```

当前训练配置默认训练 100 个 epoch，见 `configs/task/train_enhanced_vm.yaml`：

```yaml
trainer:
  epochs: 100
```

训练 dataloader worker 数在 `configs/data/train_enhanced.yaml` 中，目前为：

```yaml
num_workers: 8
```

## 2. 本地跑 CD/P2S 评测

测试集没有 GT，不能直接在官方测试集上算榜单分数。当前本地评测使用 `dataset_train` 中的 mesh 重新采样干净点云，再加噪声，调用模型降噪，然后计算：

- CD score：Chamfer Distance 相对 noisy 的改善分
- P2S score：Point-to-Surface Distance 相对 noisy 的改善分
- final score：`0.5 * CD score + 0.5 * P2S score`

评测脚本是：

```text
validate_metrics.py
```

单个 checkpoint 评测命令：

```bash
scripts/run_in_runtime_container.sh python validate_metrics.py \
  --checkpoint experiments/enhanced_vm/checkpoint_99.pkl \
  --data_config configs/data/train_enhanced.yaml \
  --transform_config configs/transform/vm.yaml \
  --model_config configs/model/enhanced_vm.yaml \
  --limit 50 \
  --output_csv experiments/enhanced_vm/validation_metrics.csv
```

参数说明：

- `--checkpoint`：要评测的模型权重。
- `--limit`：从验证列表前面取多少个样本做本地评测。建议快速筛选用 `20`，复核用 `50` 或更大。
- `--output_csv`：逐样本指标输出路径。

输出 CSV 每行是一个验证样本，字段包括：

```text
sample, cd_pred, cd_noisy, cd_score, p2s_pred, p2s_noisy, p2s_score, final_score
```

终端末尾会打印平均结果，例如：

```text
Validation Results
checkpoint: experiments/enhanced_vm/checkpoint_99.pkl
samples: 50
mean_cd_score: ...
mean_p2s_score: ...
final_score: ...
```

也可以用训练后评测脚本：

```bash
LIMIT=50 bash scripts/evaluate_train_after_training.sh experiments/enhanced_vm/checkpoint_99.pkl
```

该脚本会等待 checkpoint 出现，然后自动调用 `validate_metrics.py`。

## 3. 本地选择最好的模型

用于筛 checkpoint 的脚本是：

```text
scripts/screen_checkpoints.py
```

它会依次调用 `validate_metrics.py`，读取每个 checkpoint 的逐样本 CSV，计算平均 CD/P2S/final，并按 `final_score` 降序排序。

快速筛选示例：

```bash
scripts/run_in_runtime_container.sh python scripts/screen_checkpoints.py \
  --checkpoints \
    experiments/enhanced_vm/checkpoint_80.pkl \
    experiments/enhanced_vm/checkpoint_85.pkl \
    experiments/enhanced_vm/checkpoint_90.pkl \
    experiments/enhanced_vm/checkpoint_95.pkl \
    experiments/enhanced_vm/checkpoint_99.pkl \
  --limit 20 \
  --out_csv experiments/enhanced_vm/checkpoint_screen_metrics_limit20.csv \
  --best_out experiments/enhanced_vm/checkpoint_best_screened.pkl
```

复核建议把 `--limit` 提高到 50：

```bash
scripts/run_in_runtime_container.sh python scripts/screen_checkpoints.py \
  --checkpoints \
    experiments/enhanced_vm/checkpoint_80.pkl \
    experiments/enhanced_vm/checkpoint_90.pkl \
    experiments/enhanced_vm/checkpoint_99.pkl \
  --limit 50 \
  --out_csv experiments/enhanced_vm/checkpoint_screen_metrics_limit50.csv \
  --best_out experiments/enhanced_vm/checkpoint_best_screened.pkl
```

筛选完成后会生成：

```text
experiments/enhanced_vm/checkpoint_screen_metrics_limit50.csv
experiments/enhanced_vm/checkpoint_best_screened.pkl
```

其中 `checkpoint_best_screened.pkl` 是复制出来的最佳模型，后续推理和提交都优先使用它。

注意：Jittor/CUDA 偶尔会出现 transient segfault。`screen_checkpoints.py` 对某个 checkpoint 评测失败时会跳过它，不会中断整轮筛选。重要 checkpoint 可以单独重跑或提高 `--limit` 复核。

## 4. 生成测试集提交结果

官方 `run.py predict` 路径在当前环境下可能在 0/200 处段错误。为了稳定生成结果，当前使用直接逐样本推理脚本：

```text
scripts/predict_submit_direct.py
```

用筛选后的最佳模型推理并打包：

```bash
scripts/run_in_runtime_container.sh python scripts/predict_submit_direct.py \
  --checkpoint experiments/enhanced_vm/checkpoint_best_screened.pkl
```

输出：

```text
results/dataset_test_noisy/shapenet/<synset_id>/<model_id>/denoised.npy
results/result.zip
```

该脚本会逐个读取：

```text
dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
```

然后保存同 shape、`float32` 的：

```text
results/dataset_test_noisy/shapenet/<synset_id>/<model_id>/denoised.npy
```

如果中途失败，可加 `--resume` 继续跳过已完成输出：

```bash
scripts/run_in_runtime_container.sh python scripts/predict_submit_direct.py \
  --checkpoint experiments/enhanced_vm/checkpoint_best_screened.pkl \
  --resume
```

## 5. 校验提交包

提交前必须校验 zip 结构、文件数量、dtype 和 shape：

```bash
scripts/run_in_runtime_container.sh python scripts/validate_result_zip.py \
  --zip results/result.zip \
  --noisy_root dataset_test_noisy/shapenet
```

通过时输出应类似：

```text
zip: results/result.zip
denoised files: 200
expected files: 200
OK
```

提交包路径：

```text
results/result.zip
```

zip 内部必须是：

```text
shapenet/<synset_id>/<model_id>/denoised.npy
```

## 6. 2026-06-24 优化记录与下一步方向

本轮优化的目标是从已崩溃的 v2 训练继续提升榜单分数，并形成后续可复用路线。

### 6.1 本轮有效方案

原 v2 训练在 `experiments/enhanced_vm_v2/checkpoint_80.pkl` 之后发生 Jittor/CUDA 崩溃。后续采用从第 80 轮权重继续训练的方案：

- 续训 task：`configs/task/train_enhanced_vm_v2_resume80.yaml`
- 续训 data：`configs/data/train_enhanced_v2_workers4.yaml`
- 续训 system：`configs/system/enhanced_vm_v2_resume80.yaml`
- 续训脚本：`scripts/train_enhanced_v2_resume80_container.sh`
- 加载权重：`experiments/enhanced_vm_v2/checkpoint_80.pkl`
- 输出目录：`experiments/enhanced_vm_v2_resume80/`
- dataloader workers：从 8 降到 4，降低 Jittor/CUDA 崩溃概率
- 续训轮数：70 epoch
- 续训学习率：`2e-5`

续训完成后，没有直接使用最后一轮，而是批量筛 checkpoint。`limit=50` 复核结果中最佳模型是：

```text
experiments/enhanced_vm_v2_resume80/checkpoint_50.pkl
```

已复制为：

```text
experiments/enhanced_vm_v2_resume80/checkpoint_best_screened.pkl
```

本地 `limit=50` 复核分数：

```text
final_score: 66.1003
CD_score:    53.0492
P2S_score:   79.1514
```

网站提交分数从上一版提升到：

```text
score:     70.95
CD_score:  58.96
P2S_score: 82.93
```

上一版网站分数为：

```text
score:     66.35
CD_score:  54.21
P2S_score: 78.48
```

本轮网站提升：

```text
score:     +4.60
CD_score:  +4.75
P2S_score: +4.45
```

结论：继续训练、降低 worker、筛选中间 checkpoint 是有效路线。网站上 CD 和 P2S 同时提升，说明 v2 数据增强和续训方向是正收益。

### 6.2 本轮无效或收益较小的方案

测试过 TTA，但效果明显下降，不用于提交。

moderate TTA 和 light TTA 都显著低于非 TTA，说明当前模型对旋转/迭代式测试增强不敏感，或者反变换与迭代更新破坏了点分布。

后续还测试了非 TTA 推理参数扫描：

```text
patch_size: 800, 1000, 1200, 1500
seed_k:     4, 6, 8
seed_k_alpha: 1
```

已完成的 `limit=20` 结果中，最好的组合是：

```text
patch_size=800, seed_k=8, seed_k_alpha=1
CD_score:    54.76
P2S_score:   79.69
final_score: 67.23
```

默认附近组合 `patch_size=1000, seed_k=6, seed_k_alpha=1` 为：

```text
CD_score:    54.51
P2S_score:   79.31
final_score: 66.91
```

推理参数只带来约 `+0.3` 的本地提升，且后续更大 patch 组合出现 Jittor segfault 或收益下降。因此推理参数扫描不是冲高 10 分的主要方向。

注意：本轮扫描已停止，但停止终端后可能残留容器内验证进程。若发现 `validate_metrics.py ... patch1500_seed8_alpha1_limit20` 还在运行，应等待其自然退出后再启动新训练，避免多个 Jittor 容器抢同一张 GPU。

### 6.3 打包脚本修复

预测阶段已经完整生成 200 个测试样本，但旧脚本最后使用系统命令 `zip` 打包，在当前环境中报错：

```text
zip: command not found
```

该错误只影响最后压缩，不影响预测结果。已将 `scripts/predict_enhanced_container_and_pack.sh` 改为 Python `zipfile` 打包，后续会直接生成：

```text
results/result.zip
```

当前有效提交包为：

```text
/villa/mwq24-srt/track2_fresh_opt/results/result.zip
```

其中包含 200 个条目，结构为：

```text
shapenet/<synset_id>/<model_id>/denoised.npy
```

### 6.4 当前新分支：v3 strong

为了继续冲分，已创建 `v3_strong` 分支。该分支不覆盖当前 70.95 的模型和提交包。

新增文件：

```text
configs/transform/vm_v3_strong.yaml
configs/system/enhanced_vm_v3_strong.yaml
configs/task/train_enhanced_vm_v3_strong.yaml
scripts/train_enhanced_v3_strong_container.sh
```

设计思路：从当前网站有效的最佳 checkpoint 继续微调，略增强训练噪声，用更低学习率做稳定适配。

关键配置：

```text
load_ckpt: experiments/enhanced_vm_v2_resume80/checkpoint_best_screened.pkl
train noise: 0.006 - 0.026
lr: 1e-5
epochs: 50
num_workers: 4
output: experiments/enhanced_vm_v3_strong/
```

启动命令：

```bash
bash scripts/train_enhanced_v3_strong_container.sh 123
```

启动前建议确认没有残留验证/训练进程占用同一张 GPU：

```bash
ps -ef | grep -E 'validate_metrics.py|run.py --task|scan_inference_params' | grep -v grep || true
```

### 6.5 后续优化优先级

如果目标是继续提升约 10 分，单纯 TTA 或推理参数基本不够，应优先做训练和模型层面的更大改动。

优先级 1：训练多个差异化分支并分别提交网站。

- `v3_strong`：当前已创建，噪声更强，目标是适配更重噪声测试样本。
- `v3_conservative`：建议新增，噪声范围收窄到 `0.006-0.022`，目标是保护点分布和 CD。
- `v3_seed456`：同 v3 strong 或 v2 配置换 seed 训练，目标是制造模型多样性，用于网站选择或后续 ensemble。

优先级 2：每个分支都筛 checkpoint，不默认使用最后一轮。

建议训练完成后先用 `limit=20` 快筛，再用 `limit=50` 复核 top checkpoint。每个分支都输出一个：

```text
checkpoint_best_screened.pkl
```

优先级 3：网站提交做真实选择。

本地验证只适合相对筛选，最终仍以网站分数为准。每个分支最佳 checkpoint 都应生成 `result.zip` 并提交网站记录：

```text
branch, checkpoint, local_limit50_score, website_score, CD_score, P2S_score
```

优先级 4：模型容量或损失函数改造。

如果多个训练分支仍无法接近目标，需要考虑更大改动：

- 增大 `enhanced_vm.yaml` 中特征维度或 KNN 多尺度配置。
- 加 repulsion/uniformity loss，重点改善点分布覆盖和 CD。
- 加局部多尺度一致性 loss，减少局部聚团或漏点。

这类改动有机会带来更大提升，但训练风险和验证成本更高，建议在 v3 分支跑出结果后再动。

### 6.6 当前建议执行顺序

1. 等残留参数扫描容器自然结束。
2. 启动 `v3_strong` 训练。
3. 训练完成后筛 `experiments/enhanced_vm_v3_strong/checkpoint_*.pkl`。
4. 用最佳 checkpoint 生成新 `results/result.zip`。
5. 提交网站，记录 CD/P2S/score。
6. 若 v3 strong 未明显提升，创建 `v3_conservative` 分支继续训练。

不要多包一层 `results/` 或 `dataset_test_noisy/`。

## 6. 推荐完整流程

从训练到提交的一套常用命令如下：

```bash
cd /villa/mwq24-srt/track2_fresh_opt

# 1. 训练增强模型
bash scripts/train_enhanced_container.sh 123

# 2. 快速筛选候选 checkpoint
scripts/run_in_runtime_container.sh python scripts/screen_checkpoints.py \
  --checkpoints \
    experiments/enhanced_vm/checkpoint_80.pkl \
    experiments/enhanced_vm/checkpoint_85.pkl \
    experiments/enhanced_vm/checkpoint_90.pkl \
    experiments/enhanced_vm/checkpoint_95.pkl \
    experiments/enhanced_vm/checkpoint_99.pkl \
  --limit 20 \
  --out_csv experiments/enhanced_vm/checkpoint_screen_metrics_limit20.csv \
  --best_out experiments/enhanced_vm/checkpoint_best_screened.pkl

# 3. 对最有希望的 checkpoint 做更大样本复核
scripts/run_in_runtime_container.sh python scripts/screen_checkpoints.py \
  --checkpoints \
    experiments/enhanced_vm/checkpoint_80.pkl \
    experiments/enhanced_vm/checkpoint_90.pkl \
    experiments/enhanced_vm/checkpoint_99.pkl \
  --limit 50 \
  --out_csv experiments/enhanced_vm/checkpoint_screen_metrics_limit50.csv \
  --best_out experiments/enhanced_vm/checkpoint_best_screened.pkl

# 4. 用最佳 checkpoint 跑测试集推理并打包
scripts/run_in_runtime_container.sh python scripts/predict_submit_direct.py \
  --checkpoint experiments/enhanced_vm/checkpoint_best_screened.pkl

# 5. 校验 result.zip
scripts/run_in_runtime_container.sh python scripts/validate_result_zip.py \
  --zip results/result.zip \
  --noisy_root dataset_test_noisy/shapenet
```
