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


def get_data(scenarioNum=1):
    x = np.random.randn(1024).astype(np.float32)
    y = np.random.randn(1024).astype(np.float32)
    if scenarioNum == 1:
        z = x + y
    elif scenarioNum == 2:
        z = x - y
        z[:64] = z[-64:]
    else:
        raise ValueError(f"Unsupported scenarioNum: {scenarioNum}")
    return x, y, z


def gen_golden_data_simple(scenarioNum=1):
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x, y, z = get_data(scenarioNum)
    x.tofile("./input/input_x.bin")
    y.tofile("./input/input_y.bin")
    z.tofile("./output/golden.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-scenarioNum', type=int, default=1, choices=[1, 2],
                        help='Scenario number: 1=Add, 2=Sub')
    args = parser.parse_args()
    gen_golden_data_simple(args.scenarioNum)
