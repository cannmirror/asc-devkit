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

set -e

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=0

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

# 导出环境变量
HEIGHT=$1
WIDTH=$2
if [ ! $ASCEND_INSTALL_PATH ]; then
    ASCEND_INSTALL_PATH=/usr/local/Ascend/latest
    source $ASCEND_INSTALL_PATH/bin/setenv.bash
fi

export DDK_PATH=$ASCEND_INSTALL_PATH
arch=$(uname -m)
export NPU_HOST_LIB=$ASCEND_INSTALL_PATH/${arch}-linux/lib64

function main() {
    # 1. 生成输入数据和真值数据
    cd $CURRENT_DIR/scripts
    python3 gen_data.py $HEIGHT $WIDTH
    if [ $? -ne 0 ]; then
        echo "ERROR: generate input data failed!"
        return 1
    fi
    echo "INFO: generate input data success!"

    # 2. 编译acl可执行文件
    cd $CURRENT_DIR; rm -rf build; mkdir -p build; cd build
    cmake ../src
    if [ $? -ne 0 ]; then
        echo "ERROR: cmake failed!"
        return 1
    fi
    echo "INFO: cmake success!"
    make
    if [ $? -ne 0 ]; then
        echo "ERROR: make failed!"
        return 1
    fi
    echo "INFO: make success!"

    # 3. 运行可执行文件
    cd $CURRENT_DIR/run_out
    ./execute_add_template_op $HEIGHT $WIDTH
    if [ $? -ne 0 ]; then
        echo "ERROR: acl executable run failed! please check your project!"
        return 1
    fi
    echo "INFO: acl executable run success!"

    # 4. 比较真值文件
    cd $CURRENT_DIR
    python3 $CURRENT_DIR/scripts/verify_result.py       \
        $CURRENT_DIR/run_out/input_0.bin \
        $CURRENT_DIR/run_out/input_1.bin \
        $CURRENT_DIR/run_out/output_0.bin
    if [ $? -ne 0 ]; then
        echo "ERROR: compare golden data failed! the result is wrong!"
        return 1
    fi
    echo "INFO: compare golden data success!"
}

main
