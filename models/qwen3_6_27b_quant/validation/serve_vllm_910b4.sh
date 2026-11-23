#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Explicit service arguments for the validated TP4/PP1 B4 MTP5 protocol.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
action=${1:-serve}
case "$action" in
  -h|--help|help)
    cat <<'HELP'
Usage: bash serve_vllm_910b4.sh [serve|--print-command]
Required: MODEL, VLLM_LAUNCHER, RUN_DIR, ASCEND_RT_VISIBLE_DEVICES (four idle cards).
VLLM_LAUNCHER is the validated executable Python environment wrapper, not vllm.
BACKEND=pto (default) also requires PTO_TARGET_SOURCE, PTO_BUILD_ROOT,
SIMPLER_RUNTIME_DIR and PTO_WEIGHT_REFERENCE. No historical status JSON is needed.
BASE_URL defaults to http://127.0.0.1:8253. PYTHON defaults to python3.
--print-command validates paths/configuration without NPU access or service launch.
serve validates idle cards, saves the effective argv and starts in foreground.
This entry requires an already installed environment and compiled PTO bundle.
HELP
    exit 0 ;;
  serve|--print-command) ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
: "${MODEL:?Set MODEL to the mixed checkpoint directory}"
: "${VLLM_LAUNCHER:?Set VLLM_LAUNCHER to the validated environment wrapper}"
: "${RUN_DIR:?Set RUN_DIR to a new absolute result directory}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select four currently idle physical cards}"
export BACKEND=${BACKEND:-pto} BASE_URL=${BASE_URL:-http://127.0.0.1:8253}
export MODEL RUN_DIR VLLM_LAUNCHER ASCEND_RT_VISIBLE_DEVICES
python=${PYTHON:-python3}
[[ -x "$VLLM_LAUNCHER" ]] || { echo "VLLM_LAUNCHER is not executable" >&2; exit 2; }
case "$BACKEND" in
  pto)
    mode=0
    : "${PTO_TARGET_SOURCE:?Set the frozen integration source directory}"
    : "${PTO_BUILD_ROOT:?Set the validated compiled target bundle directory}"
    : "${SIMPLER_RUNTIME_DIR:?Set the Simpler AICPU runtime library directory}"
    : "${PTO_WEIGHT_REFERENCE:?Set the validated physical weight reference JSON}"
    export PTO_TARGET_SOURCE PTO_BUILD_ROOT SIMPLER_RUNTIME_DIR PTO_WEIGHT_REFERENCE
    ;;
  native) mode=3 ;;
  *) echo "BACKEND must be pto or native" >&2; exit 2 ;;
esac
# The launcher configures NZ, official custom ops, imports and the selected stack.
# These are service flags, not options reconstructed from an old status file.
args=(
  "$VLLM_LAUNCHER" -m vllm.entrypoints.cli.main serve "$MODEL"
  --host 127.0.0.1 --port 8253
  --served-model-name qwen36-mixed
  --dtype bfloat16
  --quantization ascend
  --tensor-parallel-size 4
  --pipeline-parallel-size 1
  --max-model-len 45056
  --max-num-seqs 4
  --max-num-batched-tokens 2048
  --async-scheduling
  --enable-chunked-prefill
  --gpu-memory-utilization 0.65
  --no-enable-prefix-caching
  --language-model-only
  --seed 0
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"mode\":$mode,\"cudagraph_capture_sizes\":[6,12,18,24]}"
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"enforce_eager":false}'
  --kv-cache-memory-bytes 6442450944
  --mamba-cache-mode align
)
if [[ "$BACKEND" == pto ]]; then
  extra=$("$python" - "$here/pto_target_config.json" <<'PYCONFIG'
import json
import os
from pathlib import Path
import sys
config = json.loads(Path(sys.argv[1]).read_text())
for key, value in config['qwen38_pto_mpmd'].items():
    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
        config['qwen38_pto_mpmd'][key] = os.environ[value[2:-1]]
config['qwen38_pto_mpmd']['session_dir'] = str(Path(os.environ['RUN_DIR']).resolve() / 'runtime')
reference = Path(config['qwen38_pto_mpmd']['weight_validation_reference'])
if not reference.is_file():
    raise FileNotFoundError(reference)
print(json.dumps(config))
PYCONFIG
)
  args+=(--worker-extension-cls qwen38_runtime.mode0_worker.Mode0PtoWorker
         --additional-config "$extra")
fi
# Use the same protocol validator, idle-card gate and effective-command record.
export SERVER_COMMAND_JSON
SERVER_COMMAND_JSON=$("$python" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${args[@]}")
unset SERVER_COMMAND_FILE
if [[ "$action" == --print-command ]]; then
  exec "$python" "$here/e2e_pypto_vllm_profile.py" env
fi
exec "$python" "$here/e2e_pypto_vllm_profile.py" serve
