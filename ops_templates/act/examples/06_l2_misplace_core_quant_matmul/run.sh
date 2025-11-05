#!/bin/bash
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

# replace params
sed -i 's/[[:space:]]*KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)/KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)/g' ../../include/matmul/device/device_matmul.h
sed -i 's/IS_TRANS_B[[:space:]]*=[[:space:]]*False/IS_TRANS_B = True/g' ../scripts/exec_test.py
sed -i 's/DATA_TYPE_STR[[:space:]]*=[[:space:]]*"[^"]*"/DATA_TYPE_STR = "quant_int8_bf16"/g' ../scripts/exec_test.py

cd ../../
act_dir=$(dirname "$(realpath "\$0")")
echo "act_dir directory is :$act_dir"
cd examples/06_l2_misplace_core_quant_matmul/

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

# restore params
cd $current_dir
sed -i 's/[[:space:]]*KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)/KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)/g' ../../include/matmul/device/device_matmul.h
sed -i 's/IS_TRANS_B[[:space:]]*=[[:space:]]*True/IS_TRANS_B = False/g' ../scripts/exec_test.py
sed -i 's/DATA_TYPE_STR[[:space:]]*=[[:space:]]*"[^"]*"/DATA_TYPE_STR = "float16"/g' ../scripts/exec_test.py
