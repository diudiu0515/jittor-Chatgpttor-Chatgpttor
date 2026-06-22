# 点云降噪赛题 Baseline

## 环境说明
本项目默认使用已经固化依赖的 Docker 镜像 `jittor-denoise:runtime` 运行。

当前方案有两个关键点：
- Python 依赖已经固化在镜像内，不需要每次重新 `pip install`
- Jittor 首次运行产生的 CUDA 运行包和编译缓存复用 `./.jittor_cache`

如果你是在这台已经配置过的机器上继续实验，通常不需要再重复做完整环境安装。

## 首次准备
第一次在当前机器上跑这个实验时，先确认以下内容存在：
- 代码目录：`repo_check/`
- 训练数据目录：`dataset_train/`
- 测试数据目录：`dataset_test_noisy/`
- 运行镜像：`jittor-denoise:runtime`

如果你只是想检查镜像是否存在，可以运行：
```bash
cd /villa/mwq24-srt/repo_check
sg docker -c 'docker images | grep jittor-denoise'
```

## 数据准备
如果数据还没有解压，执行以下命令：
```bash
cd /villa/mwq24-srt/repo_check
tar xzf dataset_train.tar.gz
unzip -oq dataset_test_noisy.zip
```

解压后目录应为：
- `dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj`
- `dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy`

可以用下面命令快速检查：
```bash
cd /villa/mwq24-srt/repo_check
test -d dataset_train && echo 'dataset_train ok'
test -d dataset_test_noisy && echo 'dataset_test_noisy ok'
head -n 1 datalist/train.txt
head -n 1 datalist/test.txt
```

## 首次缓存预热
第一次运行前，建议先预热一次 Jittor 缓存。这个步骤会完成：
- Jittor 运行时检查
- CUDA 包下载
- 首次编译缓存生成

由于容器内运行用户和宿主机用户的 UID/GID 不一致，建议每次首次预热前先重建缓存目录并放开目录权限。执行：
```bash
cd /villa/mwq24-srt/repo_check
rm -rf .jittor_cache
mkdir -p .jittor_cache
chmod 777 .jittor_cache
```

执行命令：
```bash
cd /villa/mwq24-srt/repo_check
sg docker -c 'docker run --rm --gpus all --network host \
  -e HTTP_PROXY=$HTTP_PROXY \
  -e HTTPS_PROXY=$HTTPS_PROXY \
  -e ALL_PROXY=$ALL_PROXY \
  -e http_proxy=$HTTP_PROXY \
  -e https_proxy=$HTTPS_PROXY \
  -e all_proxy=$ALL_PROXY \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e JITTOR_CACHE_DIR=/home/user/.cache/jittor \
  -v /villa/mwq24-srt/repo_check:/workspace \
  -v /villa/mwq24-srt/repo_check/.jittor_cache:/home/user/.cache/jittor \
  -w /workspace \
  jittor-denoise:runtime bash scripts/setup_container_env.sh'
```

如果这一步成功，后续通常不需要重复下载 Jittor 的大文件。复用条件是：
- 继续使用同一个项目目录下的 `./.jittor_cache`
- 不手动删除 `./.jittor_cache`
- 继续使用同一套 Jittor / CUDA 版本

## 日常运行入口
后续统一使用这个脚本进入固定运行环境：
```bash
cd /villa/mwq24-srt/repo_check
scripts/run_in_runtime_container.sh
```

这个脚本会自动：
- 使用 `jittor-denoise:runtime`
- 挂载项目目录到容器内 `/workspace`
- 挂载 `./.jittor_cache` 到容器内 Jittor 缓存目录
- 传递代理环境变量
- 默认使用 `CUDA_VISIBLE_DEVICES=1`

## 调试
先跑最小调试任务，确认数据读取和数据流没问题：
```bash
cd /villa/mwq24-srt/repo_check
scripts/run_in_runtime_container.sh
```

上面的默认命令等价于：
```bash
cd /villa/mwq24-srt/repo_check
scripts/run_in_runtime_container.sh python run.py --task configs/task/debug.yaml
```

## 训练
确认 `debug` 正常后，再启动训练：
```bash
cd /villa/mwq24-srt/repo_check
scripts/run_in_runtime_container.sh python run.py --task configs/task/train_vm.yaml
```

训练权重保存在 `experiments/` 目录下。

## 推理（生成提交文件）
先修改 `configs/task/predict_vm.yaml` 中的 `load_ckpt`，填入你的最佳权重路径。

然后执行：
```bash
cd /villa/mwq24-srt/repo_check
scripts/run_in_runtime_container.sh python run.py --task configs/task/predict_vm.yaml
```

降噪结果保存在 `results/` 目录下，格式为 `.npy`，类型为 `float32`，形状为 `(N, 3)`。

## 打包提交
预测完成后执行：
```bash
cd /villa/mwq24-srt/repo_check
cd results/dataset_test_noisy
zip -r ../../result.zip shapenet/
```

最终提交文件应为：`result.zip`

## 提交格式
每个测试样本一个 `denoised.npy`，目录结构与测试集一致，打包为 `result.zip`：
```
result.zip
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy    # np.float32, shape (N, 3)
```

## 常用命令汇总
首次运行：
```bash
cd /villa/mwq24-srt/repo_check
tar xzf dataset_train.tar.gz
unzip -oq dataset_test_noisy.zip
rm -rf .jittor_cache
mkdir -p .jittor_cache
chmod 777 .jittor_cache
sg docker -c 'docker run --rm --gpus all --network host \
  -e HTTP_PROXY=$HTTP_PROXY \
  -e HTTPS_PROXY=$HTTPS_PROXY \
  -e ALL_PROXY=$ALL_PROXY \
  -e http_proxy=$HTTP_PROXY \
  -e https_proxy=$HTTPS_PROXY \
  -e all_proxy=$ALL_PROXY \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e JITTOR_CACHE_DIR=/home/user/.cache/jittor \
  -v /villa/mwq24-srt/repo_check:/workspace \
  -v /villa/mwq24-srt/repo_check/.jittor_cache:/home/user/.cache/jittor \
  -w /workspace \
  jittor-denoise:runtime bash scripts/setup_container_env.sh'
scripts/run_in_runtime_container.sh
```

日常训练：
```bash
cd /villa/mwq24-srt/repo_check
scripts/run_in_runtime_container.sh python run.py --task configs/task/train_vm.yaml
```

日常预测：
```bash
cd /villa/mwq24-srt/repo_check
scripts/run_in_runtime_container.sh python run.py --task configs/task/predict_vm.yaml
```

## 旧环境安装方式
如果你不使用当前的固定镜像方案，也可以按下面方式手动准备原始 Python 环境：
```bash
# 安装计图
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 -y # 确保gcc、g++版本不高于10
conda install -c conda-forge libgomp -y # 确保OpenMP runtime存在

# 安装依赖
python -m pip install -r requirements.txt
pip install jittor numpy trimesh scipy omegaconf point-cloud-utils
```

## 本地评测（需要 GT 数据，仅组委会持有）
```bash
python evaluate.py \
    --pred_dir ./results/dataset_test_noisy \
    --gt_dir ./test_gt \
    --noisy_dir ./dataset_test_noisy \
    --mesh_dir ./dataset_train \
    --workers 8
```
