#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/g00893677/pto-dev/qwen3_8_27b-perf/work/nnal_9.0.0/set_env.sh --cxx_abi=1
export QWEN38_RUNTIME_ROOT=/home/g00893677/pto-dev/qwen3_6_27b-inner-w8a8/work/restore_16ms_20261030
export PTOAS_ROOT=/home/g00893677/pto-dev/qwen3_8_27b-perf/work/upstream_l2_bypass_20261015/ptoas/.venv
export PYTHONPATH="$QWEN38_RUNTIME_ROOT/bootstrap:$QWEN38_RUNTIME_ROOT/src:/home/g00893677/pto-dev/qwen3_8_27b-perf/work/hybrid_static_schedule_20261015/pypto/python:/usr/local/Ascend/cann-9.0.0/python/site-packages:/home/g00893677/pto-dev/qwen3_8_27b-perf/work/aic_fixed_reuse_20261017/simpler/python:/home/g00893677/pto-dev/qwen3_8_27b-perf/work/aic_fixed_reuse_20261017/simpler:/home/g00893677/pto-dev/qwen3_8_27b-pto/vllm-ascend:/home/g00893677/pto-dev/vllm"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="/home/g00893677/pto-dev/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export PYTHONSAFEPATH=1
exec /home/g00893677/pto-dev/qwen3_8_27b-perf/work/inprocess_native_nz_20261023/.venv/bin/python -u "$@"
