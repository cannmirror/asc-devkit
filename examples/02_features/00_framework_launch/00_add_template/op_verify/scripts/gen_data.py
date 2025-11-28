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

import sys
import numpy as np


def generate_input(input_size):
    a = np.random.randint(100, size=input_size).astype(np.float16)
    b = np.random.randint(100, size=input_size).astype(np.float16)

    a.tofile('../run_out/input_0.bin')
    a.tofile('../run_out/input_1.bin')


if __name__ == '__main__':
    height = sys.argv[1]
    width = sys.argv[2]
    generate_input(int(height) * int(width))


    