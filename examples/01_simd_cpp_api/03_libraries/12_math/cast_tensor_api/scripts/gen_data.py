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
import argparse
import numpy as np


# ============================================================
# Main data generator
# ============================================================

def gen_golden_data_simple(scenario_num):
    total_length = 256
    np.random.seed(42)

    if scenario_num == 1:
        # half -> int32_t: floor, no saturation
        src_data_type = np.float16
        dst_data_type = np.int32
        x = np.random.uniform(-100, 100, total_length).astype(src_data_type)
        golden = np.floor(x).astype(dst_data_type)

    elif scenario_num == 2:
        # float -> int16_t: round, saturation
        src_data_type = np.float32
        dst_data_type = np.int16
        x = np.random.uniform(-100, 100, total_length).astype(src_data_type)
        golden = np.clip(np.round(x),
                         np.iinfo(dst_data_type).min,
                         np.iinfo(dst_data_type).max).astype(dst_data_type)

    elif scenario_num == 3:
        # int8_t -> int32_t: widening, no saturation
        src_data_type = np.int8
        dst_data_type = np.int32
        x = np.random.randint(-128, 128, total_length).astype(src_data_type)
        golden = x.astype(dst_data_type)

    elif scenario_num == 4:
        # int32_t -> uint8_t: floor, saturation to [0, 255]
        src_data_type = np.int32
        dst_data_type = np.uint8
        x = np.random.randint(-128, 384, total_length).astype(src_data_type)
        golden = np.clip(np.floor(x.astype(np.float64)),
                         0, 255).astype(dst_data_type)

    elif scenario_num == 5:
        # bfloat16_t -> float: widening (exact)
        # bfloat16: upper 16 bits of float32 (1 sign, 8 exp, 7 mantissa)
        x_f32 = np.random.uniform(-100, 100, total_length).astype(np.float32)
        # truncate to bfloat16: keep upper 16 bits
        x_uint32 = x_f32.view(np.uint32)
        x_bf16 = (x_uint32 >> 16).astype(np.uint16)
        # golden: bfloat16 widened back to float32 (pad lower 16 bits with 0)
        golden_uint32 = x_bf16.astype(np.uint32) << 16
        golden = golden_uint32.view(np.float32)
        x = x_bf16

    else:  # scenario_num == 6
        # float -> bfloat16_t: narrowing with round-to-nearest-even
        x = np.random.uniform(-100, 100, total_length).astype(np.float32)
        x_uint32 = x.view(np.uint32)
        lower = x_uint32 & np.uint32(0xFFFF)
        upper = (x_uint32 >> 16).astype(np.uint32)
        # round to nearest even: carry when lower > 0x8000 or (lower == 0x8000 and odd)
        carry = (lower > np.uint32(0x8000)) | \
                ((lower == np.uint32(0x8000)) & ((upper & np.uint32(1)) != np.uint32(0)))
        result = upper + carry.astype(np.uint32)
        # clamp overflow (e.g. 0xFFFF + 1 → 0xFFFF)
        result = np.where(result > np.uint32(0xFFFF), np.uint32(0xFFFF), result)
        golden = result.astype(np.uint16)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x.tofile('./input/input_x.bin')
    golden.tofile('./output/golden.bin')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-scenarioNum', type=int, default=1,
                        choices=[1, 2, 3, 4, 5, 6],
                        help='1:f16→i32 | 2:f32→i16 | 3:i8→i32 | 4:i32→u8 | 5:bf16→f32 | 6:f32→bf16')
    args = parser.parse_args()
    gen_golden_data_simple(args.scenarioNum)
