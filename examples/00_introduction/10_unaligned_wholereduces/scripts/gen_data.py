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

import numpy as np
import os

def gen_golden_data_simple():
    datatype = np.float16
    shape = [13, 123]
    input_x = np.random.uniform(1, 100, shape).astype(datatype)
    golden = np.sum(input_x, axis=1).astype(datatype)
    tiling = [shape[0] * shape[1]] + shape
    tiling = np.array(tiling, dtype=np.uint32)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    tiling.tofile("./input/tiling.bin")
    golden.tofile("./output/golden.bin")
    

if __name__ == "__main__":
    gen_golden_data_simple()