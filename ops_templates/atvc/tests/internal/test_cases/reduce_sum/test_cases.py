#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
import sys
import numpy as np

TEST_FRAMEWORK_PATH = "{}/../../".format(os.getcwd())
sys.path.append(TEST_FRAMEWORK_PATH)
from build_test_case import run_test_case


def gen_golden_reduce(x: np.ndarray, reduce_dim=(0,)):
    return np.sum(x, axis=reduce_dim).astype(x.dtype)


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
    "ReduceSumTest": {
        "kernel_func": "ReduceSumCustom",
        "golden_func": gen_golden_reduce,
        "inputs": [{"name": "x", "dtype": "float", "data_range": [-100, 100], "shape": [48, 2]}],
        "outputs": [{"name": "y", "dtype": "float", "shape": [1, 2]}],
        "reduce_dim": [0],
    },

}

if __name__ == "__main__":
    run_test_case(TEST_CASES)