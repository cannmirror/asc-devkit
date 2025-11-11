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

    datatype = np.float32
    input_shape = [4096, 1]
    output_shape = [4096, 3]

    input_x = np.full(input_shape, 0, dtype=datatype)

    golden = np.broadcast_to(input_x, output_shape).copy().astype(datatype)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == '__main__':
    gen_golden_data_simple()