#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Qwen3.6 mixed-W8A8 TP4 serving, QA, benchmark and outer CANN trace entry.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
command=${1:-help}
case "$command" in
  help|-h|--help)
    cat <<'HELP'
For explicit service flags use: bash serve_vllm_910b4.sh help
Usage: bash e2e_pypto_vllm_910b4.sh COMMAND [arguments]
  env       Show selected interpreter, frozen command and protocol
  check     Check explicit checkpoint/corpus, command and four idle cards
  serve     Start the frozen command in foreground on four verified idle cards
  qa        Four exact32768-token context-extraction requests on an existing service
  qa-run    Own server -> exact32K QA -> stop only this server; no10K benchmark
  chat      Historical short-context four-conversation two-turn check
  bench     Official vLLM bench serve, without enabling the profiler
  window    Custom SSE/counter benchmark: CORPUS NEW_OUTPUT [--output-len N]
  capture   Profile a separate bounded B4 window on an existing service
  analyse   Export existing raw four-rank CANN profiles offline
  timeline  Read exported full-target Verify and MTP-start Decode boundaries
  run       Own server -> QA -> warmup -> unprofiled bench -> profile -> stop -> export

Run inside the existing CANN 9.1/vLLM 0.23 execution environment.
SERVER_COMMAND_FILE: frozen status JSON with server_command, or a JSON argv list.
BACKEND: pto (default) or native. No framework/configuration patch is installed.
RUN_DIR: new absolute directory; BASE_URL: service URL (default http://127.0.0.1:8253).
MODEL: checkpoint directory; CORPUS: one four-prompt 32768-token custom JSONL.
ASCEND_RT_VISIBLE_DEVICES: four explicitly selected, currently idle physical cards.
OUTPUT_LEN=10240; PROFILE_STEPS=50; PYTHON=python3 by default.
Use qa/bench/window/capture for an existing service; they never stop that service.
Use a fresh RUN_DIR for each backend and round. Model compilation is separate.
HELP
    exit 0
    ;;
esac
export BACKEND=${BACKEND:-pto} BASE_URL=${BASE_URL:-http://127.0.0.1:8253}
export OUTPUT_LEN=${OUTPUT_LEN:-10240} PROFILE_STEPS=${PROFILE_STEPS:-50}
case "$command" in
  window) shift; exec "${PYTHON:-python3}" "$here/e2e_vllm_window.py" "$@" ;;
  timeline) shift; exec "${PYTHON:-python3}" "$here/e2e_pypto_vllm_profile.py" timeline "$@" ;;
  env|check|serve|qa|chat|qa-run|bench|capture|analyse|run)
    exec "${PYTHON:-python3}" "$here/e2e_pypto_vllm_profile.py" "$command" "${@:2}" ;;
  *) echo "Unknown command: $command" >&2; exit 2 ;;
esac
