#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-jittor-denoise:runtime}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-1}"
HOST_JITTOR_CACHE="${HOST_JITTOR_CACHE:-$ROOT_DIR/.jittor_cache}"

if [ "$#" -eq 0 ]; then
	set -- python run.py --task configs/task/debug.yaml
fi

mkdir -p "$HOST_JITTOR_CACHE"

cmd="docker run --rm --gpus all --network host \
	-e HTTP_PROXY=${HTTP_PROXY:-} \
	-e HTTPS_PROXY=${HTTPS_PROXY:-} \
	-e ALL_PROXY=${ALL_PROXY:-} \
	-e http_proxy=${HTTP_PROXY:-} \
	-e https_proxy=${HTTPS_PROXY:-} \
	-e all_proxy=${ALL_PROXY:-} \
	-e CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
	-e JITTOR_CACHE_DIR=/home/user/.cache/jittor \
	-v ${ROOT_DIR}:/workspace \
	-v ${HOST_JITTOR_CACHE}:/home/user/.cache/jittor \
	-w /workspace \
	${IMAGE_NAME}"

for arg in "$@"; do
	cmd+=" $(printf '%q' "$arg")"
done

sg docker -c "$cmd"