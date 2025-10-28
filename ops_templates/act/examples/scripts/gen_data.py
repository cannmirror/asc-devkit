#!/usr/bin/python3
# coding=utf-8

# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
import sys
import logging

import torch
import numpy as np
import tensorflow as tf

IS_OUTPUT_TXT = False


class MatmulGenData:
    def __init__(self, m, n, k, b, is_trans_a, is_trans_b, is_bias, data_type_str):
        self.m = m
        self.n = n
        self.k = k
        self.b = b
        self.is_trans_a = is_trans_a
        self.is_trans_b = is_trans_b
        self.is_bias = is_bias
        self.data_type_str = data_type_str

    @staticmethod
    def write_bfloat_tensor(torch_tensor, file_path):
        with open(file_path, 'wb') as file:
            for element in torch_tensor.flatten():
                byte_data = element.view(torch.short).cpu().numpy().tobytes()
                file.write(byte_data)

    @staticmethod
    def due_overflow(data):
        data = np.maximum(data, -65504)
        data = np.minimum(data, 65504)
        return data

    def tf_matmul(self, x1_gm_fp32, x2_gm_fp32, bias_gm_fp32=None):
        tf.compat.v1.disable_eager_execution()
        x1 = tf.compat.v1.placeholder(np.float32, shape=x1_gm_fp32.shape)
        x2 = tf.compat.v1.placeholder(np.float32, shape=x2_gm_fp32.shape)
        res_tf = tf.matmul(x1, x2, transpose_a=self.is_trans_a, transpose_b=self.is_trans_b)
        if self.is_bias:
            bias = tf.compat.v1.placeholder(np.float32, shape=bias_gm_fp32.shape)
            res_tf = tf.add(res_tf, bias)
        with tf.compat.v1.Session() as sess:
            feed_dict = {
                x1: x1_gm_fp32,
                x2: x2_gm_fp32,
            }
            if self.is_bias:
                feed_dict[bias] = bias_gm_fp32
            res_tf = sess.run(res_tf, feed_dict=feed_dict)
        y_gm_fp32 = MatmulGenData.due_overflow(res_tf)
        return y_gm_fp32

    def tf_matmul_quant_int8_bf16(self, x1_gm_fp32, x2_gm_fp32, scale_gm, pertoken_scale_gm, bias_gm_fp32=None):
        tf.compat.v1.disable_eager_execution()
        x1 = tf.compat.v1.placeholder(np.float32, shape=x1_gm_fp32.shape)
        x2 = tf.compat.v1.placeholder(np.float32, shape=x2_gm_fp32.shape)
        res_tf = tf.matmul(x1, x2, transpose_a=self.is_trans_a, transpose_b=self.is_trans_b)
        if self.is_bias:
            bias = tf.compat.v1.placeholder(np.float32, shape=bias_gm.shape)
            res_tf = tf.add(res_tf, bias)
        scale = tf.compat.v1.placeholder(np.float32, shape=scale_gm.shape)
        pertoken_scale = tf.compat.v1.placeholder(np.float32, shape=pertoken_scale_gm.shape)
        res_tf = tf.multiply(res_tf, scale)
        res_tf = tf.multiply(res_tf, pertoken_scale)
        with tf.compat.v1.Session() as sess:
            feed_dict = {
                x1: x1_gm_fp32,
                x2: x2_gm_fp32,
                scale: scale_gm,
                pertoken_scale: pertoken_scale_gm
            }
            if self.is_bias:
                feed_dict[bias] = bias_gm_fp32
            res_tf = sess.run(res_tf, feed_dict=feed_dict)
        y_gm_fp32 = MatmulGenData.due_overflow(res_tf)
        return y_gm_fp32

    def torch_matmul(self, x1_torch_bf16, x2_torch_bf16, bias_torch_bf16=None):
        y_torch_bfloat16 = torch.matmul(x1_torch_bf16, x2_torch_bf16)
        if self.is_bias:
            y_torch_bfloat16 = y_torch_bfloat16 + bias_torch_bf16
        return y_torch_bfloat16

    def gen_golden_data_quant_int8_bf16(self, work_dir):
        data_type_a = np.int8
        data_type_b = np.int8
        data_type_c = np.float32
        data_type_out = torch.float16 # bfloat16
        data_type_scale = np.float32

        if self.is_trans_a:
            x1_shape = [self.k, self.m]
        else:
            x1_shape = [self.m, self.k]
        if self.is_trans_b:
            x2_shape = [self.n, self.k]
        else:
            x2_shape = [self.k, self.n]
        x1_gm = np.random.randint(-127, 128, x1_shape).astype(data_type_a)
        x1_gm_fp32 = x1_gm.astype(np.float32)
        x2_gm = np.random.randint(-127, 128, x2_shape).astype(data_type_b)
        x2_gm_fp32 = x2_gm.astype(np.float32)
        if self.is_bias:
            bias_gm = np.random.uniform(-1, 1, [1, self.n]).astype(data_type)
            bias_gm_fp32 = bias_gm.astype(np.float32)
        scale_gm = np.random.uniform(-0.01, 0.01, [1, self.n]).astype(data_type_scale)
        pertoken_scale_gm = np.random.uniform(-0.01, 0.01, [self.m, 1]).astype(data_type_scale)

        if self.is_bias:
            y_gm_fp32 = self.tf_matmul_quant_int8_bf16(x1_gm_fp32, x2_gm_fp32, scale_gm, pertoken_scale_gm,
                bias_gm_fp32)
        else:
            y_gm_fp32 = self.tf_matmul_quant_int8_bf16(x1_gm_fp32, x2_gm_fp32, scale_gm, pertoken_scale_gm)

        x1_gm.tofile(work_dir + "/input/x1_gm.bin")
        x2_gm.tofile(work_dir + "/input/x2_gm.bin")
        scale_gm.tofile(work_dir + "/input/scale_gm.bin")
        pertoken_scale_gm.tofile(work_dir + "/input/pertoken_scale_gm.bin")
        if self.is_bias:
            bias_gm.tofile(work_dir + "/input/bias_gm.bin")
        y_torch_bfloat16 = torch.from_numpy(y_gm_fp32).to(data_type_out)
        self.write_bfloat_tensor(y_torch_bfloat16, work_dir + "/output/golden.bin")

        if IS_OUTPUT_TXT:
            np.savetxt(work_dir + "/input/x1_gm.txt", x1_gm_fp32.flatten(), fmt='%f', newline='\n')
            np.savetxt(work_dir + "/input/x2_gm.txt", x2_gm_fp32.flatten(), fmt='%f', newline='\n')
            np.savetxt(work_dir + "/output/golden.txt", y_gm_fp32.flatten(), fmt='%f', newline='\n')
            np.savetxt(work_dir + "/input/scale_gm.txt", scale_gm.flatten(), fmt='%f', newline='\n')
            np.savetxt(work_dir + "/input/pertoken_scale_gm.txt", pertoken_scale_gm.flatten(), fmt='%f', newline='\n')
            if self.is_bias:
                np.savetxt(work_dir + "/input/bias_gm.txt", bias_gm_fp32.flatten(), fmt='%f', newline='\n')
        return 0

    def gen_golden_data_fp16(self, work_dir):
        data_type = np.float16
        if self.is_trans_a:
            x1_shape = [self.b, self.k, self.m]
        else:
            x1_shape = [self.b, self.m, self.k]
        if self.is_trans_b:
            x2_shape = [self.b, self.n, self.k]
        else:
            x2_shape = [self.b, self.k, self.n]
        x1_gm = np.random.uniform(-1, 1, x1_shape).astype(data_type)
        x1_gm_fp32 = x1_gm.astype(np.float32)
        x2_gm = np.random.uniform(-1, 1, x2_shape).astype(data_type)
        x2_gm_fp32 = x2_gm.astype(np.float32)
        if self.is_bias:
            bias_gm = np.random.uniform(-1, 1, [1, self.n]).astype(data_type)
            bias_gm_fp32 = bias_gm.astype(np.float32)

        if self.is_bias:
            y_gm_fp32 = self.tf_matmul(x1_gm_fp32, x2_gm_fp32, bias_gm_fp32)
        else:
            y_gm_fp32 = self.tf_matmul(x1_gm_fp32, x2_gm_fp32)
        y_gm = y_gm_fp32.astype(data_type)

        x1_gm.tofile(work_dir + "/input/x1_gm.bin")
        x2_gm.tofile(work_dir + "/input/x2_gm.bin")
        y_gm.tofile(work_dir + "/output/golden.bin")
        if self.is_bias:
            bias_gm.tofile(work_dir + "/input/bias_gm.bin")

        if IS_OUTPUT_TXT:
            np.savetxt(work_dir + "/input/x1_gm.txt", x1_gm_fp32.flatten(), fmt='%f', newline='\n')
            np.savetxt(work_dir + "/input/x2_gm.txt", x2_gm_fp32.flatten(), fmt='%f', newline='\n')
            np.savetxt(work_dir + "/output/golden.txt", y_gm_fp32.astype(np.float32).flatten(), fmt='%f', newline='\n')
            if self.is_bias:
                np.savetxt(work_dir + "/input/bias_gm.txt", bias_gm_fp32.flatten(), fmt='%f', newline='\n')
        return 0

    def gen_golden_data_bf16(self, work_dir):
        data_type = np.float32

        x1_shape = [self.b, self.m, self.k]
        x2_shape = [self.b, self.k, self.n]
        x1_gm_fp32 = np.random.uniform(-1, 1, x1_shape).astype(data_type)
        x2_gm_fp32 = np.random.uniform(-1, 1, x2_shape).astype(data_type)
        x1_torch_bf16 = torch.from_numpy(x1_gm_fp32).to(torch.bfloat16)
        x2_torch_bf16 = torch.from_numpy(x2_gm_fp32).to(torch.bfloat16)
        # cpu torch_matmul need high precision
        x1_torch_fp32 = x1_torch_bf16.to(torch.float32)
        x2_torch_fp32 = x2_torch_bf16.to(torch.float32)
        if self.is_bias:
            bias_gm_fp32 = np.random.uniform(-1, 1, [1, self.n]).astype(data_type)
            bias_torch_bf16 = torch.from_numpy(bias_gm_fp32).to(torch.bfloat16)
            bias_torch_fp32 = bias_torch_bf16.to(torch.float32)
        if self.is_bias:
            y_torch_fp32 = self.torch_matmul(x1_torch_fp32, x2_torch_fp32, bias_torch_fp32)
        else:
            y_torch_fp32 = self.torch_matmul(x1_torch_fp32, x2_torch_fp32)
        y_gm_fp32 = y_torch_fp32.numpy()
        y_torch_bfloat16 = y_torch_fp32.to(torch.bfloat16)

        if self.is_trans_a:
            x1_torch_bf16 = torch.transpose(x1_torch_bf16, 2, 1)
        if self.is_trans_b:
            x2_torch_bf16 = torch.transpose(x2_torch_bf16, 2, 1)
        self.write_bfloat_tensor(x1_torch_bf16, work_dir + "/input/x1_gm.bin")
        self.write_bfloat_tensor(x2_torch_bf16, work_dir + "/input/x2_gm.bin")
        self.write_bfloat_tensor(y_torch_bfloat16, work_dir + "/output/golden.bin")
        if self.is_bias:
            self.write_bfloat_tensor(bias_torch_bf16, work_dir + "/input/bias_gm.bin")

        if IS_OUTPUT_TXT:
            np.savetxt(work_dir + "/input/x1_gm.txt", x1_gm_fp32.flatten(), fmt='%f', newline='\n')
            np.savetxt(work_dir + "/input/x2_gm.txt", x2_gm_fp32.flatten(), fmt='%f', newline='\n')
            np.savetxt(work_dir + "/output/golden.txt", y_gm_fp32.flatten(), fmt='%f', newline='\n')
            if self.is_bias:
                np.savetxt(work_dir + "/input/bias_gm.txt", bias_gm_fp32.flatten(), fmt='%f', newline='\n')
        return 0

    def gen_golden_data(self, work_dir):
        if self.data_type_str == "quant_int8_bf16":
            self.gen_golden_data_quant_int8_bf16(work_dir)
        elif self.data_type_str == "float16":
            self.gen_golden_data_fp16(work_dir)
        elif self.data_type_str == "bfloat16":
            self.gen_golden_data_bf16(work_dir)
        else:
            logging.info("[ERROR] can't support data type %s" % (self.data_type_str))
            return -1
        return 0

    def gen_fake_golden_data(self, work_dir):
        data_type_bytes_ab = 1 if self.data_type_str == "quant_int8_bf16" else 2
        data_type_bytes_c = 4 if self.data_type_str == "quant_int8_bf16" else 2

        file_byte = self.b * self.m * self.k * data_type_bytes_ab
        with open(work_dir + "/input/x1_gm.bin", 'wb') as file:
            file.truncate(file_byte)

        file_byte = self.b * self.k * self.n * data_type_bytes_ab
        with open(work_dir + "/input/x2_gm.bin", 'wb') as file:
            file.truncate(file_byte)

        if self.data_type_str == "quant_int8_bf16":
            data_type_bytes_scale = 4 # float32
            file_byte = self.n * data_type_bytes_scale
            with open(work_dir + "/input/scale_gm.bin", 'wb') as file:
                file.truncate(file_byte)
            file_byte = self.m * data_type_bytes_scale
            with open(work_dir + "/input/pertoken_scale_gm.bin", 'wb') as file:
                file.truncate(file_byte)

        if self.is_bias:
            file_byte = 1 * self.n * data_type_bytes_c
            with open(work_dir + "/input/bias_gm.bin", 'wb') as file:
                file.truncate(file_byte)
