#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
import json
import numpy as np


def gen_demo_code(testcase, input_path, output_path):
    M, N, K = testcase["shape"]
    single_M, single_N, single_K = testcase["single_shape"]
    a_pos, b_pos, c_pos, bias_pos = testcase["position"]
    a_format, b_format, c_format, bias_format = testcase["format"]
    a_type, b_type, c_type, bias_type = testcase["dtype"]
    a_is_trans, b_is_trans = testcase["is_transpose"]
    is_bias = testcase["is_bias"]

    base_M, base_N, base_K = testcase["base_shape"]
    step_M, step_N, step_Ka, step_Kb = testcase["step_num"]

    lines = []
    with open(input_path, "r") as f:
        lines = f.readlines()

    config_code = f"""
struct MatmulConfig {{
    uint32_t m = {M};
    uint32_t n = {N};
    uint32_t k = {K};

    uint32_t singleCoreM = {single_M};
    uint32_t singleCoreN = {single_N};
    uint32_t singleCoreK = {single_K};

    uint32_t baseM = {base_M};
    uint32_t baseN = {base_N};
    uint32_t baseK = {base_K};

    uint32_t stepM = {step_M};
    uint32_t stepN = {step_N};
    uint32_t stepKa = {step_Ka};
    uint32_t stepKb = {step_Kb};

    uint32_t isTransposeA = {a_is_trans};
    uint32_t isTransposeB = {b_is_trans};
    uint32_t isBias = {is_bias};
}};

constexpr MatmulConfig MATMUL_CONFIG;
    """

    insert_code = f"""
    using A_TYPE = {a_type};
    using B_TYPE = {b_type};
    using C_TYPE = {c_type};
    using BIAS_TYPE = {bias_type};

    constexpr uint32_t m = {M};
    constexpr uint32_t n = {N};
    constexpr uint32_t k = {K};

    constexpr uint32_t singleCoreM = {single_M};
    constexpr uint32_t singleCoreN = {single_N};
    constexpr uint32_t singleCoreK = {single_K};

    constexpr uint32_t baseM = {base_M};
    constexpr uint32_t baseN = {base_N};
    constexpr uint32_t baseK = {base_K};

    constexpr uint32_t stepM = {step_M};
    constexpr uint32_t stepN = {step_N};
    constexpr uint32_t stepKa = {step_Ka};
    constexpr uint32_t stepKb = {step_Kb};

    constexpr uint32_t isTransposeA = {a_is_trans};
    constexpr uint32_t isTransposeB = {b_is_trans};
    constexpr uint32_t isBias = {is_bias};
    """

    for idx, line in enumerate(lines):
        if "insert_code" in line:
            lines[idx] = insert_code
        if "matmul_config" in line:
            lines[idx] = config_code
    
    with open(output_path, "w") as f:
        for line in lines:
            f.write(line)