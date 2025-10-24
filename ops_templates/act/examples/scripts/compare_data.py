#!/usr/bin/python3
# coding=utf-8

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import os
import sys
import logging
import math
from concurrent.futures import ProcessPoolExecutor

import torch
import numpy as np

IS_OUTPUT_TXT = True


def read_bfloat_file(file_path):
    with open(file_path, 'rb') as file:
        byte_data = file.read()
        numpy_array = np.frombuffer(byte_data, dtype=np.int16).copy()
        short_tensor = torch.from_numpy(numpy_array)
        recovered_tensor = short_tensor.view(torch.bfloat16).to(torch.float32)
        numpyt_tensor = recovered_tensor.numpy()
        return numpyt_tensor
    logging.info("[ERROR] can't get numpy tensor.")
    return np.zeros((1, 1))


def compare_chunk(chunk1_data, chunk2_data):
    wrong_num = 0
    eps = 1e-3
    for i in range(chunk1_data.shape[0]):
        a = chunk1_data[i]
        b = chunk2_data[i]
        ae = abs(a - b)
        re = 0 if math.isclose(b, 0) else ae / abs(b)
        if np.isnan(a) or np.isnan(b):
            wrong_num += 1
            continue
        if (ae > eps and re > eps):
            wrong_num += 1
    return wrong_num


def compare_data(work_dir, n, data_type_str):
    if not os.path.exists(work_dir + "/output/golden.bin"):
        logging.info("[ERROR] can't get golden bin file.")
        return -1
    if not os.path.exists(work_dir + "/output/output.bin"):
        logging.info("[ERROR] can't get output bin file.")
        return -1

    if data_type_str == "float16":
        golden_data = np.fromfile(work_dir + "/output/golden.bin", dtype="float16")
        output_data = np.fromfile(work_dir + "/output/output.bin", dtype="float16")
    elif data_type_str == "bfloat16":
        golden_data = read_bfloat_file(work_dir + "/output/golden.bin")
        output_data = read_bfloat_file(work_dir + "/output/output.bin")
    elif data_type_str == "quant_int8_bf16":
        golden_data = read_bfloat_file(work_dir + "/output/golden.bin")
        output_data = read_bfloat_file(work_dir + "/output/output.bin")
    elif data_type_str == "int8_int32":
        golden_data = np.fromfile(work_dir + "/output/golden.bin", dtype="int32")
        output_data = np.fromfile(work_dir + "/output/output.bin", dtype="int32")
    else:
        logging.info("[ERROR] unsupported data type %s" % (data_type_str))
        return -1

    if IS_OUTPUT_TXT:
        save_path = work_dir + "/output/output.txt"
        if data_type_str == "int8_int32":
            np.savetxt(save_path, output_data.astype(np.int32).flatten(), fmt='%d', newline='\n')
        else:
            np.savetxt(save_path, output_data.astype(np.float32).flatten(), fmt='%f', newline='\n')

    num_chunks = 32 # process numbers
    total_wrong_num = 0
    chunks1 = np.array_split(output_data, num_chunks)
    chunks2 = np.array_split(golden_data, num_chunks)
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(compare_chunk, chunk1, chunk2): (chunk1, chunk2)
                   for chunk1, chunk2 in zip(chunks1, chunks2)}
        for future in futures:
            total_wrong_num += future.result()

    return total_wrong_num
