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
import numpy as np


def gen_golden_data():
    m = 32
    n = 32
    k = 32
    x1_gm = np.random.uniform(1, 10, [m, k]).astype(np.float16)
    x2_gm = np.random.uniform(1, 10, [k, n]).astype(np.float16)
    golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
    os.makedirs("input_cube", exist_ok=True)
    os.makedirs("output_cube", exist_ok=True)
    x1_gm.tofile("./input_cube/x1_gm.bin")
    x2_gm.tofile("./input_cube/x2_gm.bin")
    golden.tofile("./output_cube/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
