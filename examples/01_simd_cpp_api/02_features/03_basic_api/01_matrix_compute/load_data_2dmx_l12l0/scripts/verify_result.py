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

import sys
import numpy as np


def verify_result(scenario_num, output_file, golden_file):
    m, n = 40, 50

    output_data = np.fromfile(output_file, dtype=np.float32)
    golden_data = np.fromfile(golden_file, dtype=np.float32)

    output_data = output_data.reshape(m, n)
    golden_data = golden_data.reshape(m, n)

    diff = np.abs(output_data - golden_data)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    threshold = 0.01
    if max_diff < threshold:
        print("test pass!")
        return True
    else:
        print(f"test failed! max_diff={max_diff}, mean_diff={mean_diff}")
        return False


if __name__ == "__main__":
    scenario_num = 1
    output_file = "output/output.bin"
    golden_file = "output/golden.bin"

    if len(sys.argv) > 1:
        args = sys.argv[1:]
        for arg in args:
            if "scenarioNum" in arg:
                scenario_num = int(arg.split("=")[1])
            elif arg.endswith(".bin"):
                if "output" in arg:
                    output_file = arg
                elif "golden" in arg:
                    golden_file = arg

    verify_result(scenario_num, output_file, golden_file)