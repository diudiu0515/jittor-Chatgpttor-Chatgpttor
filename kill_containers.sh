#!/bin/bash
# 强制清除卡住的Docker容器

echo "正在清除卡住的容器..."

# 1. Kill所有相关的宿主机进程
pkill -9 -f "docker run.*predict"
pkill -9 -f "run_in_runtime_container.sh"

# 2. 使用sg docker（有权限）停止容器
sg docker -c "docker ps -q --filter ancestor=jittor-denoise:runtime | xargs -r docker stop" 2>/dev/null

# 3. 等待清理
sleep 3

# 4. 强制删除所有相关容器
sg docker -c "docker ps -a -q --filter ancestor=jittor-denoise:runtime | xargs -r docker rm -f" 2>/dev/null

# 5. 清理X server进程
pkill -9 Xvfb 2>/dev/null

sleep 2

echo "清理完成！"
echo "当前状态:"
echo "  Docker进程: $(ps aux | grep 'docker run.*predict' | grep -v grep | wc -l)"
echo "  Python进程: $(ps aux | grep 'python predict' | grep -v grep | wc -l)"
