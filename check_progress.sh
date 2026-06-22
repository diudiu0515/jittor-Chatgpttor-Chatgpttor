#!/bin/bash
# 快速检查TTA进度

echo "╔══════════════════════════════════════════╗"
echo "║     折中版TTA进度 - $(date '+%H:%M:%S')     ║"
echo "╚══════════════════════════════════════════╝"

# 完成数量
completed=$(find results_balanced_tta -name "denoised.npy" 2>/dev/null | wc -l)
percent=$((completed * 100 / 200))
echo "📊 进度: $completed / 200 ($percent%)"

# 进度条
filled=$((completed / 4))
bar=$(printf "%${filled}s" | tr ' ' '█')
empty=$(printf "%$((50 - filled))s" | tr ' ' '░')
echo "   [$bar$empty]"

# GPU状态
gpu=$(nvidia-smi --id=4 --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null)
if [ -n "$gpu" ]; then
    echo "🎮 GPU: $gpu"
fi

# 进程状态
if ps aux | grep -q "predict_balanced_tta.py" | grep -v grep; then
    pid=$(ps aux | grep "docker run.*predict_balanced_tta.py" | grep -v grep | awk 'NR==1{print $2}')
    if [ -n "$pid" ]; then
        runtime=$(ps -p $pid -o etimes= 2>/dev/null | tr -d ' ')
        if [ -n "$runtime" ] && [ "$completed" -gt 0 ]; then
            avg=$((runtime / completed))
            remain=$((200 - completed))
            eta=$((avg * remain))
            eta_h=$((eta / 3600))
            eta_m=$(((eta % 3600) / 60))
            echo "⏱️  预计剩余: ${eta_h}h ${eta_m}m (平均${avg}s/样本)"
        fi
    fi
    echo "✅ 状态: 运行中"
else
    echo "❌ 状态: 未运行"
fi

# 最新文件
if [ "$completed" -gt 0 ]; then
    latest=$(find results_balanced_tta -name "denoised.npy" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
    if [ -n "$latest" ]; then
        sample=$(dirname "$latest" | xargs basename)
        echo "📝 最新: $sample"
    fi
fi

echo ""
echo "💡 持续监控: watch -n 30 ./check_progress.sh"
