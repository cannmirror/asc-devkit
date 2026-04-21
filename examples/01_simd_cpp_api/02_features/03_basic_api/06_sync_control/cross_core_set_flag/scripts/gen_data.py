#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import argparse
import numpy as np
np.random.seed(9)

def leaky_relu(x, alpha=0.001):
    # 计算Leaky ReLU
    return np.maximum(alpha * x, x)

def gen_golden_data():
    # 需要将左、右矩阵的K轴切分8次，分配到8个aicore中
    M = 32
    K = 32
    N = 64
    num_Blocks = 8
    input_type = np.uint8
    temp_type = np.half
    output_type = np.float32
    x1_gm = np.random.uniform(0, 1, [M, K]).astype(input_type)
    x2_gm = np.random.uniform(0, 1, [K, N]).astype(input_type)
    
    x11_gm = np.zeros_like(x1_gm).astype(temp_type)
    x22_gm = np.zeros_like(x2_gm).astype(temp_type)

    golden = (np.matmul(x1_gm.astype(output_type), x2_gm.astype(output_type))).astype(output_type)

    golden = leaky_relu(golden)

    golden = golden.astype(output_type)
    os.system("mkdir -p input")
    os.system("mkdir -p output")

    # 将K轴外移
    x1_gm = x1_gm.reshape(M, num_Blocks, K//num_Blocks).transpose(1, 0, 2)

    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    x11_gm.tofile("./input/x11_gm.bin")
    x22_gm.tofile("./input/x22_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()