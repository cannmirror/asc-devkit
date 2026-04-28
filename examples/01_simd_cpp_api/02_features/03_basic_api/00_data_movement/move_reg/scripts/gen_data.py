#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import os
import numpy as np

def gen_golden_data_simple():
    total_length = 64
    
    cond = np.zeros([1, total_length], dtype=np.uint32)
    cond[0, 0] = 0x00000001
    cond[0, 1] = 0xFFFFFFFF

    cond_bytes = cond.flatten().view(np.uint8)
    src_bytes = cond_bytes[0:8]
    src_bits = []
    for byte in src_bytes:
        for bit in range(8):
            src_bits.append((byte >> bit) & 1)

    mask_bytes = [0] * 32
    for src_bit_idx in range(64):
        src_bit_val = src_bits[src_bit_idx]
        mask_bit_start = src_bit_idx * 4
        for j in range(4):
            mask_bit_idx = mask_bit_start + j
            mask_byte_idx = mask_bit_idx // 8
            mask_bit_in_byte = mask_bit_idx % 8
            if src_bit_val:
                mask_bytes[mask_byte_idx] |= (1 << mask_bit_in_byte)

    golden = np.array(mask_bytes, dtype=np.uint8)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    cond.tofile('./input/input_cond.bin')
    golden.tofile('./output/golden.bin')

if __name__ == "__main__":
    gen_golden_data_simple()