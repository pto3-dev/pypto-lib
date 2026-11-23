#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Edit machine paths once here or in the ignored serving.local.sh.
# Environment values take precedence over the local examples.
config_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
local_config=${SERVING_LOCAL_CONFIG:-$config_dir/serving.local.sh}
if [[ -f "$local_config" ]]; then
  source "$local_config"
fi
export BACKEND=${BACKEND:-pto} PYTHON=${PYTHON:-python3}
case "$BACKEND" in
  pto)
    export VLLM_LAUNCHER=${VLLM_LAUNCHER:-${PTO_ENV:-}}
    export BASE_URL=${BASE_URL:-http://127.0.0.1:8253}
    ;;
  native)
    export VLLM_LAUNCHER=${VLLM_LAUNCHER:-${NATIVE_ENV:-}}
    export BASE_URL=${BASE_URL:-http://127.0.0.1:8254}
    ;;
  *) echo "BACKEND must be pto or native" >&2; return 2 ;;
esac
export MODEL=${MODEL:-} CORPUS_ROOT=${CORPUS_ROOT:-} SCENE=${SCENE:-zh_qa}
case "$SCENE" in
  zh_qa|en_technical|code_review) ;;
  *) echo "SCENE must be zh_qa, en_technical or code_review" >&2; return 2 ;;
esac
export CORPUS=${CORPUS:-${CORPUS_ROOT:+$CORPUS_ROOT/$SCENE.jsonl}}
export RESULTS_ROOT=${RESULTS_ROOT:-$config_dir/results}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}
export PTO_TARGET_SOURCE=${PTO_TARGET_SOURCE:-} PTO_BUILD_ROOT=${PTO_BUILD_ROOT:-}
export SIMPLER_RUNTIME_DIR=${SIMPLER_RUNTIME_DIR:-} PTO_WEIGHT_REFERENCE=${PTO_WEIGHT_REFERENCE:-}
export OUTPUT_LEN=${OUTPUT_LEN:-10240} PROFILE_STEPS=${PROFILE_STEPS:-50}
export SERVICE_LATEST_LINK="$RESULTS_ROOT/$BACKEND/service.latest"
