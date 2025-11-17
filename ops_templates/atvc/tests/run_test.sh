#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

if command -v bishengcc; then
    COMPILE_TOOL=bishengcc
elif command -v ascc; then
    COMPILE_TOOL=ascc
else
    echo "Error: Cannot find bishengcc/ascc compiling tool, please check cann package version or set up envrionment first."
    exit 1
fi

ATVC_HOME_DIR=$CURRENT_DIR/../
TEST_CASE_LIST=$(ls $ATVC_HOME_DIR/examples|xargs)
if [ $# -ne 1 ]; then
    echo "This script takes only one test case name as input. Execution example: 'bash run_test.sh [$TEST_CASE_LIST]'"
    exit 1
fi
TEST_NAME=$1

if [[ " $TEST_CASE_LIST " == *" ${TEST_NAME} "* ]]; then
    cd $ATVC_HOME_DIR/examples/$TEST_NAME
    rm -rf ./$TEST_NAME
    ${COMPILE_TOOL} -arch Ascend910B1 $TEST_NAME.cpp -o $TEST_NAME --include-path ${ATVC_HOME_DIR}/include
    if [ ! -f ./$TEST_NAME ]; then
        echo "Error: Cannot find file ./${TEST_NAME} due to compilation error, please check error message."
        exit 1
    fi
    ./$TEST_NAME
    if [ $? -eq 0 ]; then
        echo "Sample ${TEST_NAME} passed!"
    else
        echo "Sample ${TEST_NAME} failed!"
    fi
    cd ${ATVC_HOME_DIR}
else
    echo "Error: Cannot find '$TEST_NAME' in ${ATVC_HOME_DIR}examples. Execution example: 'bash run_test.sh [$TEST_CASE_LIST]'"
    exit 1
fi