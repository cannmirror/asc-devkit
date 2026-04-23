#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import sys
import numpy as np


RELATIVE_TOL = 1e-5
ABSOLUTE_TOL = 1e-8
ERROR_TOL = 1e-3


def verify_result(output_file, golden_file):
    """
    验证输出结果与真值是否一致
    
    参数：
    - output_file: 输出数据文件路径
    - golden_file: 真值数据文件路径
    
    返回：
    - True: 验证通过
    - False: 验证失败
    """
    output_type = np.float32
    output = np.fromfile(output_file, dtype=output_type).reshape(-1)
    golden = np.fromfile(golden_file, dtype=output_type).reshape(-1)
    
    assert output.size >= golden.size, f"output size {output.size} < expected {golden.size}"
    
    output = output[:golden.size]
    
    # 使用 isclose 进行逐元素比较
    different_element_results = np.isclose(output,
                                           golden,
                                           rtol=RELATIVE_TOL,
                                           atol=ABSOLUTE_TOL,
                                           equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]
    
    # 打印前100个不一致的元素
    for index in range(min(len(different_element_indexes), 100)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print("data index: %06d, expected: %-.9f, actual: %-.9f" %
              (real_index, golden_data, output_data))
    
    error_ratio = float(different_element_indexes.size) / golden.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, ERROR_TOL))
    return error_ratio <= ERROR_TOL


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 verify_result.py output.bin golden.bin")
        sys.exit(1)
    
    try:
        res = verify_result(sys.argv[1], sys.argv[2])
        if not res:
            raise ValueError("[ERROR] result error")
        print("test pass!")
    except Exception as e:
        print(e)
        sys.exit(1)