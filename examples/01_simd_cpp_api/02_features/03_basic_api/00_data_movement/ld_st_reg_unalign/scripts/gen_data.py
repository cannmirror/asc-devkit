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
import sys
import numpy as np

def gen_golden_data():
    total_length = 1024
    data_type = np.float32
    
    input_x = np.zeros(total_length, dtype=data_type)
    input_x[1:129] = 1.0
    input_x = input_x.reshape(1, total_length)
    
    input_y = np.zeros(total_length, dtype=data_type)
    input_y[2:130] = 1.0
    input_y = input_y.reshape(1, total_length)
    
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    input_x.tofile('./input/input_x.bin')
    input_y.tofile('./input/input_y.bin')
    
    golden = np.zeros(total_length, dtype=data_type).reshape(1, total_length)
    golden[0, 3:131] = input_x[0, 1:129] + input_y[0, 2:130]
    
    golden.tofile('./output/golden.bin')

if __name__ == "__main__":
    gen_golden_data()