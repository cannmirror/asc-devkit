#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import csv
import ast
import logging


logging.basicConfig(level=logging.INFO, filename='result.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')


# 该函数将 CSV 文件中的每一行存储为一个test case，并将所有结果返回。
# - testcase_name: 测试用例名称，字符串类型。
# - op_name: 算子名称，字符串类型。
# - input_names: 输入名称列表或元组，存储输入的名称。
# - input_dtypes: 输入数据类型列表或元组，存储输入的数据类型。
# - input_shapes: 输入形状列表或元组，存储输入的形状。
# - data_ranges: 数据范围列表或元组，存储输入或输出的数据范围。
# - output_names: 输出名称列表或元组，存储输出的名称。
# - output_dtypes: 输出数据类型列表或元组，存储输出的数据类型。
# - output_shapes: 输出形状列表或元组（可选），存储输出的形状。
# - scalar_names: 标量名称列表或元组（可选），存储标量的名称。
# - scalar_dtypes: 标量数据类型列表或元组（可选），存储标量的数据类型。
# - scalar_values: 标量值列表或元组（可选），存储标量的值。
# - op_temps: 描述OpTraits临时buffer类型
# - kernel_so: 选用算子编译的kernel so
# - exec_bin: 选用算子调用的执行文件
def csv_to_testcase(file_path, golden_func):
    result_dict = dict()
    try:
        with open(file_path, mode='r', encoding="utf-8") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                test_case_info = dict()
                testcase_name = row["testcase_name"]
                test_case_info['kernel_func'] = row["op_name"]
                test_case_info['golden_func'] = golden_func
                if row['exec_bin']:
                    test_case_info['exec_bin'] = row['exec_bin']
                if row['kernel_so']:
                    test_case_info['kernel_so'] = row['kernel_so']
                input_infos = []
                output_infos = []
                scalar_infos = []
                input_names = ast.literal_eval(row['input_names'])
                input_dtypes = ast.literal_eval(row['input_dtypes'])
                input_shapes = ast.literal_eval(row['input_shapes'])
                data_ranges = ast.literal_eval(row['data_ranges'])
                output_names = ast.literal_eval(row['output_names'])
                output_dtypes = ast.literal_eval(row['output_dtypes'])
                temp_types = []
                if row['op_temps']:
                    temp_types = ast.literal_eval(row['op_temps'])
                if len(data_ranges) != len(input_names):
                    data_ranges = [data_ranges[0]] * len(input_names)
                if len(input_dtypes) != len(input_names):
                    input_dtypes = [input_dtypes[0]] * len(input_names)
                if len(output_dtypes) != len(output_names):
                    output_dtypes = [output_dtypes[0]] * len(output_names)
                zip_input = zip(input_names, input_dtypes, input_shapes, data_ranges)
                for name_in, dt_in, shape_in, range_in in zip_input:
                    input_info = dict()
                    input_info['name'] = name_in
                    input_info['dtype'] = dt_in
                    input_info['shape'] = shape_in
                    input_info['data_range'] = range_in
                    input_infos.append(input_info)
                zip_output = zip(output_names, output_dtypes)
                for i, (name_out, dt_out) in enumerate(zip_output):
                    output_info = dict()
                    output_info['name'] = name_out
                    output_info['dtype'] = dt_out
                    if row['output_shapes']:
                        output_info['shape'] = ast.literal_eval(row['output_shapes'])[i]
                    output_infos.append(output_info)
                if row['scalar_names']:
                    zip_scalar = zip(ast.literal_eval(row['scalar_names']),
                                     ast.literal_eval(row['scalar_dtypes']),
                                     ast.literal_eval(row['scalar_values']))
                    for scal_na, scal_dt, scal_val in zip_scalar:
                        scalar_info = dict()
                        scalar_info['name'] = scal_na
                        scalar_info['dtype'] = scal_dt
                        scalar_info['value'] = scal_val
                        scalar_infos.append(scalar_info)
                if 'reduce_dim' in row:
                    if row['reduce_dim'] == "((,))":
                        test_case_info["reduce_dim"] = None
                    else:
                        test_case_info["reduce_dim"] = ast.literal_eval(row['reduce_dim'])
                if 'broadcast' in row:
                    test_case_info["broadcast"] = bool(row['broadcast'])
                test_case_info["inputs"] = input_infos
                test_case_info["outputs"] = output_infos
                test_case_info["scalars"] = scalar_infos
                test_case_info["op_temps"] = temp_types
                result_dict[testcase_name] = test_case_info
    except FileNotFoundError:
        logging.error(f"文件 {file_path} 未找到，请检查文件路径。")
    except csv.Error as e:
        logging.error(f"读取 CSV 文件时发生错误: {e}")
    except Exception as e:
        logging.error(f"发生意外错误: {e}")
    return result_dict