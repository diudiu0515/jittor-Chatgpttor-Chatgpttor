#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/../repo_check}"
IMAGE_NAME="${IMAGE_NAME:-jittor-denoise:runtime}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-4}"
HOST_JITTOR_CACHE="${HOST_JITTOR_CACHE:-$ROOT_DIR/../repo_check/.jittor_cache}"

if [ "$#" -eq 0 ]; then
	set -- python run.py --task configs/task/debug.yaml
fi

mkdir -p "$HOST_JITTOR_CACHE"

cmd="docker run --rm --gpus all --network host --entrypoint /bin/bash \
	-e HTTP_PROXY=${HTTP_PROXY:-} \
	-e HTTPS_PROXY=${HTTPS_PROXY:-} \
	-e ALL_PROXY=${ALL_PROXY:-} \
	-e http_proxy=${HTTP_PROXY:-} \
	-e https_proxy=${HTTPS_PROXY:-} \
	-e all_proxy=${ALL_PROXY:-} \
	-e CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
	-e JITTOR_CACHE_DIR=/home/user/.cache/jittor \
	-v ${ROOT_DIR}:/workspace \
	-v ${DATA_ROOT}:${DATA_ROOT}:ro \
	-v ${HOST_JITTOR_CACHE}:/home/user/.cache/jittor \
	-w /workspace \
	${IMAGE_NAME} \
	-lc"

inner_cmd=""
for arg in "$@"; do
	inner_cmd+=" $(printf '%q' "$arg")"
done
cmd+=" $(printf '%q' "${inner_cmd# }")"

sg docker -c "$cmd"