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
import sys
import numpy as np


def get_range_by_dtype(input_type):
    try:
        if input_type == np.float16 or input_type == np.float32 or input_type == np.float64:
            return np.finfo(input_type).min, np.finfo(input_type).max
        else:
            return np.iinfo(input_type).min, np.iinfo(input_type).max
    except ValueError:
        print(f"Unsupported data type:{input_type}")


def gen_golden_data(scenario_num):
    """
    根据场景编号生成输入数据和Golden数据：
    场景1：普通转置，对[16, 16]的二维矩阵进行转置
    场景2：增强转置，[N,C,H,W]与[N,H,W,C]两个数据格式互相转换
    """
    
    if scenario_num == 1:
        input_x = np.arange(1, 129).astype(np.uint16)
        gdst = np.arange(2, 129, 2).astype(np.uint16)
        gzero = np.zeros(64).astype(np.uint16)
        golden = np.concatenate([gdst, gzero]).astype(np.uint16)
        os.makedirs("input", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        input_x.tofile("./input/input_x.bin")
        golden.tofile("./output/golden.bin")
    elif scenario_num == 2:
        gen_golden_data_custom()


def gen_golden_data_custom():
    one_repeat_size = 256
    data_block_size = 32
    input_type = np.dtype("uint32")
    output_type = input_type
    type_size = input_type.itemsize
    block_length = 256
    mask = 70
    src0_block_stride = 1
    repeat_times = 2
    src0_repeat_stride = 4
    src1_repeat_stride = 0
    one_data_block_items = data_block_size // type_size

    min_val, max_val = get_range_by_dtype(input_type)
    input_x_shape = [block_length]
    input_y_shape = [block_length // 8]
    input_x = np.random.uniform(min_val, max_val, input_x_shape).astype(input_type)
    input_y = 0x7E7C00A5 * np.ones(input_y_shape).astype(input_type)
    input_mask = np.unpackbits(input_y.view(np.uint8), bitorder='little').astype(bool)
    golden = np.zeros(input_x_shape).astype(input_type)
    for i in range(repeat_times):
        base = i * (src0_repeat_stride // src0_block_stride) * one_data_block_items
        base_m = i * src1_repeat_stride * one_data_block_items
        src0_iter = input_x[base : base + mask]
        mask_slice = input_mask[base_m : base_m + mask]
        selected = src0_iter[mask_slice]
        base_g = i * selected.size
        golden[base_g : base_g + selected.size] = selected
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-scenario_num', type=int, default=1, choices=range(1, 3))
    args = parser.parse_args()
    gen_golden_data(args.scenario_num)
