#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
export IS_PERF="0"

# if use split cann run, need to set the following environment variables
ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/latest
export ASCEND_HOME_DIR=$ASCEND_CANN_PACKAGE_PATH

SHORT=r:,v:,p:,
LONG=run-mode:,soc-version:,perf:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        (-r | --run-mode )
            RUN_MODE="$2"
            shift 2;;
        (-v | --soc-version )
            SOC_VERSION="$2"
            shift 2;;
        (-p | --perf )
            IS_PERF="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done
current_dir=$(dirname "$(realpath "\$0")")
echo "current_dir directory is :$current_dir"
cd ../../
act_dir=$(dirname "$(realpath "\$0")")
echo "act_dir directory is :$act_dir"
cd examples/01_misplace_core_matmul/

bishengcc "$current_dir/main.cpp" -arch $SOC_VERSION -I$act_dir
rm -rf build

rm -rf input
mkdir input
rm -rf output
mkdir output

mkdir -p bin
cd bin
# ascenc_matmul_bbit only
mv ../a.out ./ascendc_matmul_bbit

# npu only
if [ "${IS_PERF}" = "1" ]; then
    python3 -u ../../scripts/exec_test.py npu "perf"
else
    python3 -u ../../scripts/exec_test.py npu "normal"
fi
