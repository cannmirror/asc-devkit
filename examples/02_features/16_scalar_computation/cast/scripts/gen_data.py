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
import numpy as np
import torch


def bf16_tofile(bf16_tensor, filename):
    """
    将PyTorch bfloat16 视为numpy int16保存
    """
    uint16_view = bf16_tensor.view(torch.int16)
    uint16_view.cpu().numpy().tofile(filename)


def gen_golden_data_simple():
    input_type = torch.bfloat16
    output_type = np.float32
    block_length = 1024

    input_shape = [block_length]
    output_shape = [block_length]

    tensor = (torch.rand(block_length) * 100)  # [0,1) -> [0,2000) -> [-1000,1000)
    bf16_tensor = tensor.to(input_type)
    f32_np = bf16_tensor.float().numpy()
    f32_scalar_value = f32_np[0]
    golden = f32_scalar_value * np.ones(output_shape, dtype=output_type)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    bf16_tofile(bf16_tensor, "./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
