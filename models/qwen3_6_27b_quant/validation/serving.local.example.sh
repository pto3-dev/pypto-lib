#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Copy to serving.local.sh and fill in installed paths once.
# Never commit a machine-specific serving.local.sh.
export MODEL=${MODEL:-/installed/checkpoint}
export CORPUS_ROOT=${CORPUS_ROOT:-/installed/four-prompt-corpora}
export PTO_ENV=${PTO_ENV:-/installed/environment/pto_env.sh}
export NATIVE_ENV=${NATIVE_ENV:-/installed/environment/native_env.sh}
export PTO_TARGET_SOURCE=${PTO_TARGET_SOURCE:-/installed/frozen-integration}
export PTO_BUILD_ROOT=${PTO_BUILD_ROOT:-/installed/validated-target-bundle}
export SIMPLER_RUNTIME_DIR=${SIMPLER_RUNTIME_DIR:-/installed/simpler/build/lib/a2a3}
export PTO_WEIGHT_REFERENCE=${PTO_WEIGHT_REFERENCE:-/installed/weight-reference.json}
export RESULTS_ROOT=${RESULTS_ROOT:-/installed/serving-results}
# Example only. Startup refuses cards with existing processes.
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
