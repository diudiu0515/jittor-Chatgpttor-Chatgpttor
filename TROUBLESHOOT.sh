#!/bin/bash
# 清理和重启指南

echo "╔════════════════════════════════════════════════════════╗"
echo "║            Docker容器冲突解决方案                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

echo "当前问题: 多个Docker容器卡在X server初始化阶段"
echo ""

echo "═══ 方案1: 等待自然超时 (推荐) ═══"
echo "容器会在10-30分钟后自动退出"
echo "运行以下命令监控:"
echo "  watch -n 60 'ps aux | grep predict_balanced | wc -l'"
echo ""

echo "═══ 方案2: 重启后运行 ═══"
echo "如果有服务器重启权限，重启后运行:"
echo "  cd /villa/mwq24-srt/repo_check"
echo "  CUDA_VISIBLE_DEVICES=4 scripts/run_in_runtime_container.sh python predict_moderate_tta.py"
echo ""

echo "═══ 方案3: 使用现有62分结果 ═══"
echo "已有完整200个样本的结果(62分):"
echo "  results/dataset_test_noisy/shapenet/"
echo ""
echo "可以继续方案B: 改进模型架构并重新训练"
echo "  • 实现EnhancedFeatureExtractor"
echo "  • 重新训练模型(需要1-2天)"
echo "  • 预期提升至75-85分"
echo ""

echo "═══ 当前状态 ═══"
stuck=$(ps aux | grep "predict_balanced" | grep -v grep | wc -l)
echo "卡住的进程数: $stuck"

running_docker=$(ps aux | grep "docker run.*predict" | grep -v grep | wc -l)
echo "Docker容器数: $running_docker"

echo ""
echo "建议: 选择方案3，专注于模型架构改进(方案B)"
echo "      或等待30分钟后容器自动清理，再尝试运行TTA"
