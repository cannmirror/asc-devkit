#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

CASE_REL=01_simd_cpp_api/02_features/00_framework/04_aclgraph
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../../_case_entry.sh"
presmoke_case_init "$CASE_REL"

scenario_build_dir() {
    printf '%s_scenario%s\n' "$BUILD_DIR" "$1"
}

case_build() {
    local scenario_num
    local scenario_dir
    for scenario_num in 1 2; do
        scenario_dir="$(scenario_build_dir "$scenario_num")"
        mkdir -p "$scenario_dir"
        (
            cd "$scenario_dir"
            SCENARIO_NUM=$scenario_num soc_version=$SOC_VERSION bash -lc \
                'cmake .. -DCMAKE_ASC_ARCHITECTURES="$ARCH" -DSCENARIO_NUM=$SCENARIO_NUM $RUN_MODE_ARG'
        )
        (
            cd "$scenario_dir"
            SCENARIO_NUM=$scenario_num soc_version=$SOC_VERSION bash -lc 'make -j'
        )
    done
}

case_run() {
    local scenario_num
    local scenario_dir
    for scenario_num in 1 2; do
        scenario_dir="$(scenario_build_dir "$scenario_num")"
        mkdir -p "$scenario_dir"
        (
            cd "$scenario_dir"
            SCENARIO_NUM=$scenario_num soc_version=$SOC_VERSION bash -lc './demo'
        )
    done
}

case_verify() {
    mkdir -p "$BUILD_DIR"
    :
}

case_clean() {
    rm -rf "$BUILD_DIR" "$(scenario_build_dir 1)" "$(scenario_build_dir 2)"
}

presmoke_case_main "$@"
