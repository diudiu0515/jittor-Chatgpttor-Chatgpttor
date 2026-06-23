# 赛道二重新开始实验工作区

这个目录是新的干净起点，来自官方 `starter_code.zip`，不复用旧的 `jittor/`、`repo_check/`、`thuhrw_repo/` 失败工程。

## 目录

- `configs/`: 官方配置，训练和预测入口都从这里读参数。
- `src/`: 官方 baseline 模型和数据代码。
- `datalist/`: 官方 train/validate/test 列表。
- `scripts/`: 本工作区新增的便捷脚本。
- `experiments/`: 训练权重输出目录。
- `results/`: 推理结果和提交包输出目录。
- `logs/`: 训练日志输出目录。

## 1. 准备环境

建议使用 Python 3.9，并确保 gcc/g++ 不高于 10。依赖安装：

```bash
cd /villa/mwq24-srt/track2_fresh_opt
source scripts/activate_env.sh
python -m pip install -r requirements.txt
python -m pip install jittor numpy trimesh scipy omegaconf point-cloud-utils tqdm
```

如果 Jittor 首次编译失败，优先检查 CUDA、gcc/g++ 版本和 `~/.cache/jittor` 权限。

当前工作区已经创建了 `.venv`，并在 `.toolchain/gcc10` 放置了本地 `gcc/g++ 10`。每次训练、预测前先执行：

```bash
cd /villa/mwq24-srt/track2_fresh_opt
source scripts/activate_env.sh
```

## 2. 准备数据

根目录已经有 `dataset_train.tar.gz` 和 `dataset_test_noisy.zip`。执行：

```bash
cd /villa/mwq24-srt/track2_fresh_opt
bash scripts/prepare_data.sh
```

脚本会优先链接旧目录里已经解压好的数据；如果找不到，就从根目录压缩包解压。

## 3. 跑通 baseline 训练

先用官方配置跑一版 baseline：

```bash
cd /villa/mwq24-srt/track2_fresh_opt
bash scripts/train_baseline.sh 123
```

默认训练 `configs/task/train_vm.yaml`，权重保存在 `experiments/vm/`，日志在 `logs/train_seed_123.log`。

## 4. 生成提交包

训练结束后用最佳 checkpoint 推理并打包：

```bash
cd /villa/mwq24-srt/track2_fresh_opt
bash scripts/predict_and_pack.sh experiments/vm/checkpoint_99.pkl
```

输出文件：`results/result.zip`。

## 5. 推荐优化顺序

1. 先保持官方代码不变，确认训练和提交包能完整跑通。
2. 调小或调大 `configs/task/train_vm.yaml` 的 `lr`、`epochs`，建立稳定 baseline。
3. 改 `configs/data/train.yaml` 的 `num_files`、`batch_size` 做速度和效果权衡。
4. 再动 `src/model/vm.py` 和 `configs/transform/vm.yaml`，每次只改一个变量并记录日志。
5. 每个有效实验保留对应 checkpoint、配置副本和 `logs/` 日志。

## 注意

官方测试集没有 GT，本地无法得到最终榜单分数。能做的是用 `datalist/validate.txt` 监控训练行为，并用提交结果格式检查保证可上传。
