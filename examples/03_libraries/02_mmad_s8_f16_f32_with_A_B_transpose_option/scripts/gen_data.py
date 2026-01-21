#!/usr/bin/python3
# coding=utf-8

# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import argparse
import numpy as np
np.random.seed(9)


def gen_golden_data(tilingKey=1, M=0, K=0, N=0):
    if tilingKey == 1:
        x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.int8)
        x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.int8)
        golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.int32)
    elif tilingKey == 2:
        x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.int8)
        x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.int8)
        golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.int32)
        x2_gm = x2_gm.transpose()
    elif tilingKey == 3:
        x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.int8)
        x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.int8)
        golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.int32)
        x1_gm = x1_gm.transpose()
    elif tilingKey == 4:
        x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.int8)
        x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.int8)
        golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.int32)
        x1_gm = x1_gm.transpose()
        x2_gm = x2_gm.transpose()
    elif tilingKey == 5:
        x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.float16)
        x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.float16)
        golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
    elif tilingKey == 6:
        x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.float16)
        x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.float16)
        golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
        x2_gm = x2_gm.transpose()
    elif tilingKey == 7:
        x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.float16)
        x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.float16)
        golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
        x1_gm = x1_gm.transpose()
    elif tilingKey == 8:
        x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.float16)
        x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.float16)
        golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
        x1_gm = x1_gm.transpose()
        x2_gm = x2_gm.transpose()
    elif tilingKey == 9:
        x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.float32)
        x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.float32)
        golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
    elif tilingKey == 10:
        x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.float32)
        x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.float32)
        golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
        x2_gm = x2_gm.transpose()
    elif tilingKey == 11:
        x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.float32)
        x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.float32)
        golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
        x1_gm = x1_gm.transpose()
    elif tilingKey == 12:
        x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.float32)
        x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.float32)
        golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
        x1_gm = x1_gm.transpose()
        x2_gm = x2_gm.transpose()
    if tilingKey <= 4:
        golden = golden.astype(np.int32)
    else:
        golden = golden.astype(np.float32)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-key', type=int, default=1, choices=range(1, 13))
    parser.add_argument('-m', type=int)
    parser.add_argument('-k', type=int)
    parser.add_argument('-n', type=int)
    args = parser.parse_args()
    gen_golden_data(args.key, args.m, args.k, args.n)