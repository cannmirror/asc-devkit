#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import subprocess
import numpy as np

LOSS = 1e-3  # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
MINIMUM = 10e-10
logging.basicConfig(level=logging.INFO, filename='result.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_np_dtype(input_type: str):
    dtype_map = {
        "float": np.float32,
        "int8_t": np.int8,
        "int16_t": np.int16,
        "int32_t": np.int32,
        "int64_t": np.int64,
        "uint8_t": np.uint8,
        "uint16_t": np.uint16,
        "uint32_t": np.uint32,
        "uint64_t": np.uint64,
        "int": np.int32,
        "float16_t": np.float16,
        "float64_t": np.float64,
        "half": np.float16
    }
    return dtype_map.get(input_type, np.float32)


def verify_result(real_result, golden):
    try:
        result = np.abs(real_result - golden)  # 计算运算结果和预期结果偏差
        deno = np.maximum(np.abs(real_result), np.abs(golden))  # 获取最大值并组成新数组
        result_atol = np.less_equal(result, LOSS)  # 计算绝对误差
        result_rtol = np.less_equal(
            result / np.add(deno, MINIMUM), LOSS)  # 计算相对误差
        if not result_rtol.all() and not result_atol.all():
            if np.sum(result_rtol == False) > real_result.size * LOSS and \
                    np.sum(result_atol == False) > real_result.size * LOSS:
                logging.error(f"Result error, result_rtol = {np.sum(result_rtol == False)}")
                logging.error(f"Result error, resut_atol = {np.sum(result_atol == False)}")
                logging.info(f"golden:{golden}")
                logging.info(f"real_result: {real_result}")
                false_positions = np.where(result_rtol == False)
                logging.info(f"pos: {false_positions[:16]}")
                return False
    except Exception as e:
        logging.error(f"{e}")
        return False
    return True


def run_cmds(cmds_str: str):
    ret = subprocess.run(cmds_str, shell=True)
    if ret.returncode != 0:
        raise RuntimeError(f"Execute {cmds_str} failed!")
