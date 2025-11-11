#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ==============================================================================

import numpy as np
import os

def gen_golden_data_simple():

    dtype = np.float32

    input_shape = [7, 2024]
    input_x = np.random.uniform(-100, 100, input_shape).astype(dtype)

    golden = np.sum(input_x[:, :-1], axis=1).astype(dtype)


    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

    print(f" golden: {golden}")
    print(f" input: {input_x}")

if __name__ == '__main__':
    gen_golden_data_simple()