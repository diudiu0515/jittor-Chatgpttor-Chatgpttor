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
