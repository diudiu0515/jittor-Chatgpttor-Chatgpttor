#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JTCUDA_DIR="/villa/mwq24-srt/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux"

export PATH="$ROOT_DIR/.venv/bin:$ROOT_DIR/.toolchain/gcc10/usr/bin:$JTCUDA_DIR/bin:$PATH"
export CC="$ROOT_DIR/.toolchain/gcc10/usr/bin/gcc-10"
export CXX="$ROOT_DIR/.toolchain/gcc10/usr/bin/g++-10"
export CUDAHOSTCXX="$ROOT_DIR/.toolchain/gcc10/usr/bin/g++-10"
export JT_CC="$ROOT_DIR/.toolchain/gcc10/usr/bin/g++-10"
export cc_path="$ROOT_DIR/.toolchain/gcc10/usr/bin/g++-10"
export cuda_home="$JTCUDA_DIR"
export CUDA_HOME="$JTCUDA_DIR"
export nvcc_path="$JTCUDA_DIR/bin/nvcc"
export LD_LIBRARY_PATH="$JTCUDA_DIR/lib64:$ROOT_DIR/.toolchain/gcc10/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export use_cutt=0
export JITTOR_HOME="/villa/mwq24-srt"
export JITTOR_CACHE_PATH="$JITTOR_HOME/.cache/jittor"

echo "Python: $(command -v python)"
echo "CC: $CC"
echo "CXX: $CXX"
echo "cc_path: $cc_path"
echo "cuda_home: $cuda_home"
echo "use_cutt: $use_cutt"
echo "JITTOR_HOME: $JITTOR_HOME"
echo "JITTOR_CACHE_PATH: $JITTOR_CACHE_PATH"