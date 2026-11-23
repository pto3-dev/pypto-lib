#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# One-command profile against the service recorded by serve_vllm_910b4.sh.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "${1:-}" in
  -h|--help|help)
    echo "Usage: bash profile_vllm_910b4.sh [pto|native] [zh_qa|en_technical|code_review]"
    echo "Start the matching service first. Paths come from serving.local.sh."
    exit 0 ;;
esac
if (( $# > 2 )); then echo "Too many arguments" >&2; exit 2; fi
export BACKEND=${1:-${BACKEND:-pto}} SCENE=${2:-${SCENE:-zh_qa}}
if (( $# == 2 )); then unset CORPUS; fi
source "$here/serving_config.sh"
: "${VLLM_LAUNCHER:?Fill PTO_ENV and NATIVE_ENV in serving.local.sh once}"
[[ -x "$VLLM_LAUNCHER" ]] || { echo "Environment launcher is not executable" >&2; exit 2; }
# The client and exporter run in the same installed Python environment as vLLM.
exec "$VLLM_LAUNCHER" "$here/serving_workflow.py" profile
