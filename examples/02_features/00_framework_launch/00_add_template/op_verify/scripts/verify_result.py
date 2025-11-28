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
import math
import numpy as np


def data_compare(input1_file, input2_file, output_file):
    """
    Verify that the data are the same
    """
    input1 = np.fromfile(input1_file, dtype=np.float16)
    input2 = np.fromfile(input2_file, dtype=np.float16)
    golden = input1 + input2
    output = np.fromfile(output_file, dtype=np.float16)

    different_element_results = np.isclose(
        output, golden,
        rtol=1e-3,
        atol=1e-8,
        equal_nan=True)
    different_element_indexes = np.where(
        different_element_results != np.array((True,)))[0]
    return 0 if different_element_indexes.size == 0 else 1


if __name__ == '__main__':
    try:
        input1_file = sys.argv[1]
        input2_file = sys.argv[2]
        output_file = sys.argv[3]
        cmp_result = data_compare(input1_file, input2_file, output_file)
        if (cmp_result == 0):
            print("test pass")
        else:
            raise ValueError("[ERROR] result error")
    except Exception as e:
        print(e)
        sys.exit(1)
