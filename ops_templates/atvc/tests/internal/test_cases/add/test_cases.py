#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import os
import sys
import numpy as np

TEST_FRAMEWORK_PATH = "{}/../../".format(os.getcwd())
sys.path.append(TEST_FRAMEWORK_PATH)
from build_test_case import run_test_case


def gen_golden_add(x: np.ndarray, y: np.ndarray):
    return x + y


TEST_CASES = {
    # 测试用例名: {
    #    "kernel_func": 函数文件中编写的kernel计算函数
    #    "golden_func": 当前文件中的真值函数: 入参严格按照inputs顺序排列, 返回值严格按照outputs顺序排列
    #    输入tensor信息描述：
    #    "inputs": [{"name": 0号输入名, "dtype": 0号输入数据类型, "data_range": 0号输入数据范围, "shape": 0号输入shape},
    #               {"name": 1号输入名, "dtype": 1号输入数据类型, "data_range": 1号输入数据范围, "shape": 1号输入shape}, ...,],
    #    输出tensor信息描述：
    #    "outputs": [{"name": 0号输出名, "dtype": 0号输出类型}, ...]
    # },
    "AddTestFloat": {
        "kernel_func": "AddCustomFloat",
        "golden_func": gen_golden_add,
        "inputs": [{"name": "a", "dtype": "float", "data_range": [-100, 100], "shape": [4096, 2048]},
                   {"name": "b", "dtype": "float", "data_range": [-200, 200], "shape": [4096, 2048]}],
        "outputs": [{"name": "c", "dtype": "float"}],
    },
    "AddTestInt": {
        "kernel_func": "AddCustomInt",
        "golden_func": gen_golden_add,
        "inputs": [{"name": "a", "dtype": "int32_t", "data_range": [-100, 100], "shape": [4096, 2048]},
                   {"name": "b", "dtype": "int32_t", "data_range": [-200, 200], "shape": [4096, 2048]}],
        "outputs": [{"name": "c", "dtype": "int32_t"}],
    },
}

if __name__ == "__main__":
    run_test_case(TEST_CASES)