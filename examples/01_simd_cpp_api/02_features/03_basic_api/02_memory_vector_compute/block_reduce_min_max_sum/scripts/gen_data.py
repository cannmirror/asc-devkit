#!/usr/bin/python3
# coding=utf-8

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
import argparse
import numpy as np
np.random.seed(9)


def half_binary_tree_sum(data):
    """
    模拟硬件BlockReduceSum的二叉树两两相加过程，每一步都在float16精度下进行
    """
    arr = list(data.astype(np.float16))
    while len(arr) > 1:
        new_arr = []
        for i in range(0, len(arr), 2):
            if i + 1 < len(arr):
                new_arr.append(np.float16(arr[i] + arr[i + 1]))
            else:
                new_arr.append(arr[i])
        arr = new_arr
    return arr[0]


def gen_golden_data(scenarioNum=1):
    """
    根据场景编号生成输入数据和Golden数据
    场景1：BlockReduceMax，输入[1, 128]，输出[1, 64]（前8个为有效值），每个datablock内求最大值
    场景2：BlockReduceMin，输入[1, 128]，输出[1, 64]（前8个为有效值），每个datablock内求最小值
    场景3：BlockReduceSum，输入[1, 128]，输出[1, 64]（前8个为有效值），每个datablock内求和
    """
    input_type = np.dtype("float16")
    output_type = input_type
    block_length = 128
    one_data_block_items = 32 // input_type.itemsize
    block_num = block_length // one_data_block_items
    dst_length = 64

    input_x = np.random.uniform(-10, 10, [1, block_length]).astype(input_type)
    golden = np.zeros([1, dst_length]).astype(output_type)

    for i in range(block_num):
        block_data = input_x[0, i * one_data_block_items:(i + 1) * one_data_block_items]
        if scenarioNum == 1:
            golden[0, i] = np.max(block_data)
        elif scenarioNum == 2:
            golden[0, i] = np.min(block_data)
        elif scenarioNum == 3:
            golden[0, i] = half_binary_tree_sum(block_data)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-scenarioNum', type=int, default=1, choices=range(1, 4))
    args = parser.parse_args()
    gen_golden_data(args.scenarioNum)
