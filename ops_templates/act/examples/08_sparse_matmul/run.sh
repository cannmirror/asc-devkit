#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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

TRANSPOSE_A=false
# replace params
if [ "$TRANSPOSE_A" = "true" ]; then
    sed -i 's/using[[:space:]]*LayoutA[[:space:]]*=[[:space:]]*layout::RowMajor/using LayoutA = layout::ColumnMajor/g' ./main.cpp
    sed -i 's/IS_TRANS_A[[:space:]]*=[[:space:]]*False/IS_TRANS_A = True/g' ../scripts/exec_test.py
else
    sed -i 's/using[[:space:]]*LayoutA[[:space:]]*=[[:space:]]*layout::ColumnMajor/using LayoutA = layout::RowMajor/g' ./main.cpp
    sed -i 's/IS_TRANS_A[[:space:]]*=[[:space:]]*True/IS_TRANS_A = False/g' ../scripts/exec_test.py
fi
sed -i 's/IS_TRANS_B[[:space:]]*=[[:space:]]*False/IS_TRANS_B = True/g' ../scripts/exec_test.py
sed -i 's/IS_BIAS[[:space:]]*=[[:space:]]*True/IS_BIAS = False/g' ../scripts/exec_test.py
sed -i 's/IS_SPARSE[[:space:]]*=[[:space:]]*False/IS_SPARSE = True/g' ../scripts/exec_test.py
sed -i 's/DATA_TYPE_STR[[:space:]]*=[[:space:]]*"[^"]*"/DATA_TYPE_STR = "int8_int32"/g' ../scripts/exec_test.py

cd ../../
act_dir=$(dirname "$(realpath "\$0")")
echo "act_dir directory is :$act_dir"
cd examples/08_sparse_matmul/

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
