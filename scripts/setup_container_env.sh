#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export JITTOR_CACHE_DIR="$HOME/.cache/jittor"

mkdir -p "$JITTOR_CACHE_DIR/cutt"

if [ ! -s "$JITTOR_CACHE_DIR/cutt/cutt-1.2.zip" ]; then
	curl -L --retry 5 --retry-delay 2 \
		https://codeload.github.com/Jittor/cutt/zip/v1.2 \
		-o "$JITTOR_CACHE_DIR/cutt/cutt-1.2.zip"
fi

python - <<'PY'
import jittor as jt
jt.flags.use_cuda = 1
print('jittor_version=', jt.__version__)
print('has_cuda=', jt.has_cuda)
a = jt.ones((2, 3))
print('tensor_sum=', a.sum().item())
PY