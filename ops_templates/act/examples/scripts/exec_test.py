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
import sys
import csv
import time
import logging

import numpy as np

from gen_data import MatmulGenData
from sparse_gen_data import SparseMatmulGenData
from compare_data import compare_data

IS_TRANS_A = False
IS_TRANS_B = False
IS_BIAS = False
IS_SPARSE = False
# support float16 bfloat16 or "quant_int8_bf16" for quant mamtul or "int8_int32" for sparse matmul
DATA_TYPE_STR = "float16"

logging.basicConfig(level=logging.INFO)


def get_file_work_dir():
    current_path = os.getcwd()
    file_work_dir = os.path.dirname(current_path)
    return file_work_dir


def get_case_list():
    current_path = os.getcwd()
    case_dir = os.path.join(os.path.dirname(current_path), "testcase")
    if not os.path.exists(case_dir):
        logging.info("[ERROR] file path %s not exist!" % (case_dir))
        return None

    case_list = []
    for file_name in os.listdir(case_dir):
        if not file_name.endswith(".csv"):
            continue

        abs_file_name = os.path.join(case_dir, file_name)
        with open(abs_file_name, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_list = []
                item_list.append(int(row[0].lstrip("\ufeff")))
                item_list.append(row[1])
                item_list.append(int(row[2]))
                item_list.append(int(row[3]))
                item_list.append(int(row[4]))
                if len(row) > 5:
                    item_list.append(int(row[5]))
                else:
                    item_list.append(1)
                case_list.append(item_list)
    return case_list


def find_prof_file(perf_dir):
    for root, _, files in os.walk(perf_dir):
        for file in files:
            if file.startswith("OpBasicInfo"): # msprof op_summary_* / msprof op OpBasicInfo*
                return os.path.join(root, file)
    return ""


def get_perf_task_duration(prof_output_dir):
    task_duration = 0
    prof_file = find_prof_file(prof_output_dir)

    try:
        with open(prof_file, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                task_duration = row["Task Duration(us)"]
    except FileNotFoundError:
        logging.info("[WARNNING] can't find profiling file!")
        return 0

    return task_duration


def clear_file_cache(file_work_dir):
    rm_files = file_work_dir + "/input/*"
    os.system("rm -rf " + rm_files)
    rm_files = file_work_dir + "/output/*.txt"
    os.system("rm -rf " + rm_files)
    rm_files = file_work_dir + "/output/*.bin"
    os.system("rm -rf " + rm_files)
    rm_files = file_work_dir + "/build/prof_out/*"
    os.system("rm -rf " + rm_files)


def process_case(file_work_dir, case_name, m, n, k, b, is_perf):
    logging.info("[INFO] start process case[%s]" % (case_name))
    clear_file_cache(file_work_dir)

    if IS_SPARSE:
        matmul_gen_data = SparseMatmulGenData(m, n, k, b, IS_TRANS_A, IS_TRANS_B, IS_BIAS, DATA_TYPE_STR)
    else:
        matmul_gen_data = MatmulGenData(m, n, k, b, IS_TRANS_A, IS_TRANS_B, IS_BIAS, DATA_TYPE_STR)

    if is_perf:
        matmul_gen_data.gen_fake_golden_data(file_work_dir)

        os.system("msprof op --application=\"./ascendc_matmul_bbit %s %s %s %s\" --output=\"./prof_out\"" %
            (m, n, k, b))
    else:
        matmul_gen_data.gen_golden_data(file_work_dir)
        os.system("./ascendc_matmul_bbit %s %s %s %s" % (m, n, k, b))
    if is_perf:
        wrong_num = -1
    else:
        logging.info("[INFO] compare data case[%s]" % (case_name))
        wrong_num = compare_data(file_work_dir, n, DATA_TYPE_STR)
    res_data = []
    res_data.append(case_name)
    res_data.append(wrong_num)
    res_data.append(b * m * n)
    if wrong_num == -1:
        res_data.append("None")
    elif wrong_num / (b * m * n) > 0.001:
        res_data.append("Fail")
    else:
        res_data.append("Success")
    if is_perf:
        task_duration = get_perf_task_duration("./prof_out")
        res_data.append(task_duration)
    return res_data


def main():
    args_len = len(sys.argv) - 1
    if args_len != 2:
        logging.info("[ERROR] exec_test input params error!")
        return -1

    run_mode = sys.argv[1]
    if run_mode != "cpu" and run_mode != "npu":
        logging.info("[ERROR] run_mode [%s]!" % (run_mode))
        return -1

    file_work_dir = get_file_work_dir()
    if not os.path.exists(file_work_dir):
        logging.info("[ERROR] file path %s not exist!" % (file_work_dir))
        return -1

    is_perf = False
    if sys.argv[2] == "perf":
        is_perf = True

    if IS_SPARSE:
        if (not IS_TRANS_B) or (DATA_TYPE_STR != "int8_int32") or IS_BIAS:
            logging.info("[ERROR] sparse only support trans_b and int8, does not support bias")
            return -1

    case_list = get_case_list()
    res_list = [["case_name", "wrong_num", "total_num", "result", "task_duration"]]
    for is_process, case_name, m, n, k, b in case_list:
        if is_process == 1:
            res_data = process_case(file_work_dir, case_name, m, n, k, b, is_perf)
            res_list.append(res_data)

    timestamp = time.time()
    result_file_name = "result_" + str(timestamp) + ".csv"
    with open(os.path.join(file_work_dir, "output", result_file_name), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(res_list)

    logging.info("---------------RESULT---------------")
    for res in res_list:
        logging.info(res)
    return 0


if __name__ == "__main__":
    main()
