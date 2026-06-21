# 项目训练、评测与优化流程补充

这份文档把项目的完整实验流程合在一起：如何跑 baseline、如何选择最好的 checkpoint、如何本地比较模型优劣、如何生成提交文件，以及如何记录每次优化实验。

## 1. 准备数据

按照原来的md解压就可以

## 2. 跑实验

主要流程：
```text
修改少量参数
  -> 修改 checkpoint 输出目录
  -> 训练
  -> select_best_checkpoint.py 选 best pkl
  -> validate_metrics.py 算本地 CD/P2S
  -> 记录结果
  -> 修改 predict_vm.yaml 的 load_ckpt
  -> 推理测试集
  -> 打包 result.zip
  -> 线上提交
  -> 记录线上分数
  -> 决定下一组优化
```

### 训练

```bash
python run.py --task configs/task/train_vm.yaml
```
baseline 默认训练 100 轮。每一轮结束后都会保存一个模型，训练时项目会使用 `configs/data/train.yaml` 里的 `validate_dataset` 做验证。它会从 `dataset_train` 读取干净 mesh，采样干净点云，再添加模拟噪声，然后计算验证 loss。

### 选择最好的 checkpoint

默认配置是：
```yaml
load_ckpt: experiments/vm/checkpoint_99.pkl
```
但其实需要选择最好。

训练结束后运行：

```bash
python select_best_checkpoint.py
```
这个脚本会：

1. 扫描 `experiments/vm/checkpoint_*.pkl`
2. 逐个加载 checkpoint
3. 在验证集上计算平均 validation loss
4. 打印每个 checkpoint 的 loss
5. 把验证 loss 最低的模型复制为：

```text
experiments/vm/checkpoint_best_manual.pkl
```

同时生成：

```text
experiments/vm/checkpoint_validation.csv
```

`checkpoint_validation.csv` 里会记录每个 checkpoint 的验证 loss，这个指标用于同一次训练内部选择最好的 pkl。validation loss 越低，说明这个 checkpoint 在模拟验证集上的训练目标越好。

### 修改推理配置

打开：

```text
configs/task/predict_vm.yaml
```

把 `load_ckpt` 改成你选出来的最佳权重。例如：

```yaml
load_ckpt: experiments/vm/checkpoint_best_manual.pkl
```

如果你比较的是某个优化实验，例如 `vm_patch4`：

```yaml
load_ckpt: experiments/vm_patch4/checkpoint_best_manual.pkl
```

同时确认 writer 配置满足提交格式。赛题要求输出文件名必须是 `denoised.npy`，所以建议改成：

```yaml
writer:
  __target__: vm
  save_dir: results
  save_name: denoised
```

这样推理结果会保存为：

```text
results/dataset_test_noisy/shapenet/<synset_id>/<model_id>/denoised.npy
```

### 对测试集推理

configs/task/predict_vm.yaml 中确认：
```yaml
  transform: predict
```

configs/data/predict.yaml 中确认预测阶段使用单 worker，避免多进程 dataloader 漏样本：
```yaml
  num_workers: 0
```

同时确认 writer 配置：
```yaml
writer:
  __target__: vm
  save_dir: results
  save_name: denoised
```

运行：

```bash
python run.py --task configs/task/predict_vm.yaml
find results/dataset_test_noisy/shapenet -name denoised.npy | wc -l
```


```text
每个 .npy 必须是 float32
shape 必须是 (N, 3)
点数 N 必须和对应 noisy.npy 完全一致
```

### 计算CD和P2S分数

```bash
python validate_metrics.py --checkpoint experiments/vm/checkpoint_best_manual.pkl --limit 50 --output_csv experiments/vm/validation_metrics.csv
```

这个脚本会在验证集上模拟比赛评测，输出：

```text
mean_cd_score
mean_p2s_score
final_score
```

其中：

```text
final_score = 0.5 * mean_cd_score + 0.5 * mean_p2s_score
```

这更接近赛题指标。分数越高越好。

## 3. 本地评测指标

第一类是 validation loss：它适合用于选择同一次训练中的最佳 checkpoint。第二类是本地模拟 CD/P2S 分数。

指标优先级建议这样看：

```text
线上分数 > 本地 CD/P2S final_score > validation loss
```
## 4. baseline 和优化实验对比

做优化实验时，建议换一个 checkpoint 输出目录，避免覆盖 baseline，maybe每次跑可以重新设一个目录。比如把 `configs/system/vm.yaml` 改成：

```yaml
__target__: vm
ckpt_save_dir: experiments/vm_patch4
ckpt_save_name: checkpoint
```

然后重新训练：

```bash
python run.py --task configs/task/train_vm.yaml
```

选择优化实验的 best checkpoint：

```bash
python select_best_checkpoint.py --ckpt_dir experiments/vm_patch4 --log_csv experiments/vm_patch4/checkpoint_validation.csv
```

计算优化实验的本地 CD/P2S：

```bash
python validate_metrics.py --checkpoint experiments/vm_patch4/checkpoint_best_manual.pkl --limit 50 --output_csv experiments/vm_patch4/validation_metrics.csv
```

对比方式：

```text
baseline:
  best val loss:
  mean_cd_score:
  mean_p2s_score:
  local final_score:

patch4:
  best val loss:
  mean_cd_score:
  mean_p2s_score:
  local final_score:
```

如果优化后的 `best val loss` 更低，并且 `local final_score` 更高，这个优化大概率值得拿去线上提交。

## 5. 打包提交文件

```bash
cd results/dataset_test_noisy
zip -r ../../result.zip shapenet/
```

最终提交：

```text
result.zip
```

压缩包结构应该是：

```text
result.zip
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy
```
