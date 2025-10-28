#!/usr/bin/env python3
# -*- coding:utf-8 -*-
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
import shutil
from dataclasses import dataclass
from functools import reduce
import numpy as np
from utils.main_template import build_main_file
from utils.common import verify_result, get_np_dtype, run_cmds
from utils.csv_parser import csv_to_testcase


SHOW_ELEMENT_NUM = 8

logging.basicConfig(level=logging.INFO, filename='result.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class ExecArgs:
    run_mode: str = "npu"
    enable_prof: bool = False
    prof_path: str = "../"
    enable_dump: bool = False


class TestCaseExecutor:
    def __init__(self, test_cases, exec_args: ExecArgs):
        self.test_cases = test_cases
        self.exec_args = exec_args
        

    @staticmethod
    def check_test_case(test_cases_info):
        def check_keys(dic, key_set):
            if not key_set.issubset(set(dic.keys())):
                logging.error(f"{dic} must contain {key_set}")
                return False
            return True
        mandatory_attrs = set(["kernel_func",
                            "golden_func", "inputs", "outputs"])
        mandatory_input_tensor_keys = set(["name", "dtype", "data_range", "shape"])
        mandatory_output_tensor_keys = set(["name", "dtype"])
        mandatory_scalar_keys = set(["name", "dtype", "value"])
        if not check_keys(test_cases_info, mandatory_attrs):
            return False
        for input_dict in test_cases_info["inputs"]:
            if not check_keys(input_dict, mandatory_input_tensor_keys):
                return False
        for output_dict in test_cases_info["outputs"]:
            if not check_keys(output_dict, mandatory_output_tensor_keys):
                return False
        for scalar_dict in test_cases_info.get("scalars", []):
            if not check_keys(scalar_dict, mandatory_scalar_keys):
                return False
        return True


    @staticmethod
    def normalize_test_case(test_case_info):
        for key_word in ["exec_bin", "kernel_so"]:
            if key_word in test_case_info:
                test_case_info[key_word] = os.path.abspath(test_case_info[key_word])


    @staticmethod
    def verify_results(case_name, test_case_info):
        output_name = [output_info["name"]
                    for output_info in test_case_info.get("outputs", [])]
        actual_results = [
            "./{}/output/output_{}.bin".format(case_name, i) for i in output_name]
        golden_results = [
            "./{}/output/golden_{}.bin".format(case_name, i) for i in output_name]
        if len(golden_results) != len(actual_results):
            logging.error("Actual result size is not same as golden.")
            return False

        zip_res = zip(actual_results, golden_results)
        for i, (act_res, golden_res) in enumerate(zip_res):
            dtype = get_np_dtype(test_case_info["outputs"][i]["dtype"])
            real_result = np.fromfile(
                act_res, dtype=dtype)  # 从bin文件读取实际运算结果
            logging.info(
                f"First {SHOW_ELEMENT_NUM} elements in actual output: \n {list(real_result[:SHOW_ELEMENT_NUM])}")
            golden = np.fromfile(golden_res, dtype=dtype)  # 从bin文件读取预期运算结果
            logging.info(f"First {SHOW_ELEMENT_NUM} elements in golden: \n {list(golden[:SHOW_ELEMENT_NUM])}")
            if not verify_result(real_result, golden):
                logging.error("test case: {} failed!".format(case_name))
                return False
        logging.info("test case: {} passes successfully!".format(case_name))
        return True


    @staticmethod
    def copy_compile_files(case_name):
        if os.path.exists("./" + case_name):
            shutil.rmtree("./{}".format(case_name))
        os.makedirs(case_name)
        os.makedirs("{}/input".format(case_name), exist_ok=True)
        os.makedirs("{}/output".format(case_name), exist_ok=True)
        base_path = os.path.dirname(os.path.abspath(__file__))
        run_cmds(" ".join(["cp", "-r", "./kernel.cpp", "../../utils/data_utils.h ", "./" + case_name + "/"]))
        logging.info("[INFO] copy cmake file {} success!".format(case_name))


    @staticmethod
    def copy_exec_files(case_name, test_case_info):
        os.makedirs(case_name, exist_ok=True)
        os.makedirs("{}/input".format(case_name), exist_ok=True)
        os.makedirs("{}/output".format(case_name), exist_ok=True)
        base_path = os.path.dirname(os.path.abspath(__file__))
        run_cmds(" ".join(["cp", "-r", test_case_info["kernel_so"],
            test_case_info["exec_bin"], "./" + case_name + "/"]))
        logging.info("[INFO] copy executable file {} success!".format(case_name))


    @staticmethod
    def gen_golden_data(case_name, test_case_info):
        inputs_data = [np.random.uniform(*input_info["data_range"],
                input_info["shape"]).astype(get_np_dtype(input_info["dtype"]))
                for input_info in test_case_info["inputs"]]
        golden_func = test_case_info["golden_func"]
        if "reduce_dim" in test_case_info:
            inputs_data.append(tuple(test_case_info["reduce_dim"]))
        if "broadcast" in test_case_info:
            inputs_data.append(tuple(test_case_info["outputs"][0]["shape"]))
        outputs_data = golden_func(*inputs_data)
        if isinstance(outputs_data, np.ndarray) or np.isscalar(outputs_data):
            outputs_data = [outputs_data]
        os.system(f"mkdir -p ./{case_name}/input")
        os.system(f"mkdir -p ./{case_name}/output")
        for input_idx in range(len(test_case_info["inputs"])):
            logging.info(f"The first {SHOW_ELEMENT_NUM} elements in input data "
                         f"{test_case_info['inputs'][input_idx]['name']} is: "
                         f"{list(inputs_data[input_idx].flatten()[:SHOW_ELEMENT_NUM])}")
            inputs_data[input_idx].tofile(
                f"./{case_name}/input/input_{test_case_info['inputs'][input_idx]['name']}.bin")
        for output_idx in range(len(test_case_info["outputs"])):
            logging.info(f"The first {SHOW_ELEMENT_NUM} elements"
                         f" in output data {test_case_info['outputs'][output_idx]['name']} is: "
                         f"{list(outputs_data[output_idx].flatten()[:SHOW_ELEMENT_NUM])}")
            outputs_data[output_idx].tofile(
                f"./{case_name}/output/golden_{test_case_info['outputs'][output_idx]['name']}.bin")


    def compile_kernel(self, case_name, test_case_info):
        case_dir = "./" + case_name
        atvc_home_path = os.path.abspath(os.path.dirname(__file__) + "/../../include")
        if "kernel_so" not in test_case_info:
            logging.info("Start compiling kernel library.")
            cmd_str = f"cd {case_dir}; bishengcc -shared kernel.cpp -arch Ascend910B1"\
                        f" --include-path {atvc_home_path} -o libkernel.so"
            if self.exec_args.enable_dump:
                cmd_str += " -DASCENDC_DUMP=1"
            else:
                cmd_str += " -DASCENDC_DUMP=0"
            logging.info(cmd_str)
            run_cmds(cmd_str)
        else:
            run_cmds(" ".join(["cp", "-r", test_case_info["kernel_so"], "./" + case_name + "/"]))
        kernel_lib_name = "kernel" if "kernel_so" not in test_case_info \
            else os.path.basename(test_case_info["kernel_so"])[3:-3]
        link_cmds = f"cd {case_dir}; export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH;"\
            f" bishengcc main.cpp -arch Ascend910B1 -L{os.getcwd() + '/' + case_name}"\
            f" -l{kernel_lib_name} -o main --include-path {atvc_home_path}"
        logging.info("Start compiling executable file.")
        logging.info(f"{link_cmds}")
        run_cmds(link_cmds)
        logging.info("Compile op success!")


    def execute_kernel(self, case_name, test_case_info):
        case_dir = "./" + case_name
        cwd = os.getcwd()
        ld_library_path = os.environ["LD_LIBRARY_PATH"]
        shape_sizes = [reduce(lambda x, y: x * y, inp["shape"]) for inp in test_case_info["inputs"]]
        kernel_lib_path = f"{cwd}/{case_name}/"
        exec_bin = "./main" if "exec_bin" not in test_case_info else test_case_info["exec_bin"]
        os.environ["LD_LIBRARY_PATH"] = f"{kernel_lib_path}:{ld_library_path}"
        exec_bin_args = ""
        if "reduce_dim" in test_case_info:
            input_shape_str = ",".join([str(i) for i in test_case_info["inputs"][0]["shape"]])
            output_shape_str = ",".join([str(i) for i in test_case_info["outputs"][0]["shape"]])
            dim_str = ",".join([str(i) for i in test_case_info["reduce_dim"]])
            dtype = "1" if "float" in test_case_info["inputs"][0]["dtype"] else "0"
            exec_bin_args = f"{input_shape_str} {output_shape_str} {dim_str} {dtype}"
        elif "broadcast" in test_case_info:
            input_shape_str = ",".join([str(i) for i in test_case_info["inputs"][0]["shape"]])
            output_shape_str = ",".join([str(i) for i in test_case_info["outputs"][0]["shape"]])
            dtype = "1" if "float" in test_case_info["inputs"][0]["dtype"] else "0"
            exec_bin_args = f"{input_shape_str} {output_shape_str} {dtype}"
        else:
            exec_bin_args = f"{max(shape_sizes)}"
        if self.exec_args.enable_prof:
            exec_bin_args += " 1"
        else:
            exec_bin_args += " 0"
        exec_cmds = f"{exec_bin} {exec_bin_args}"
        if self.exec_args.enable_prof and self.exec_args.run_mode == "npu":
            exec_cmds = f"msprof --ai-core=on --ascendcl=on --model-execution=on --runtime-api=on"\
                f" --task-time=on --application='{exec_bin} {exec_bin_args}' --output={self.exec_args.prof_path}"
        exec_cmds = f"cd ./{case_name}; {exec_cmds}"
        logging.info(exec_cmds)
        run_cmds(exec_cmds)
        logging.info("Execute op success!")


    def exec_single_case(self, case_name, test_case_info):
        logging.info("<<<<<<<<<<<<<<< START TESTING {} >>>>>>>>>>>>>>>".format(case_name))
        self.normalize_test_case(test_case_info)
        if "kernel_so" in test_case_info and "exec_bin" in test_case_info:
            self.copy_exec_files(case_name, test_case_info)
        else:
            self.copy_compile_files(case_name)
            # 构造Kernel外部调用的main函数
            build_main_file(case_name, test_case_info, self.exec_args)
            # 编译
            self.compile_kernel(case_name, test_case_info)
        # 调用测试用例函数生成真值
        self.gen_golden_data(case_name, test_case_info)
        self.execute_kernel(case_name, test_case_info)
        if self.exec_args.enable_prof:
            cmd_str = "rm -rf ./{}".format(case_name)
            run_cmds(cmd_str)
        else:
            self.verify_results(case_name, test_case_info)


    def run(self, case_names: str):
        if not case_names:
            for case, test_case_info in self.test_cases.items():
                self.exec_single_case(case, test_case_info)
            return
        case_name_list = [name.strip() for name in case_names.split(",") if name.strip()]
        for case_name in case_name_list:
            if case_name not in self.test_cases.keys():
                raise RuntimeError(f"Case name : {case_name} cannot be found!")
            self.exec_single_case(case_name, self.test_cases[case_name])


def run_test_case(test_cases, case_name="",
    exec_args=ExecArgs(run_mode="npu", enable_prof=False, prof_path="../", enable_dump=False)):
    executor = TestCaseExecutor(test_cases, exec_args)
    executor.run(case_name)


def run_test_case_with_csv(csv_path, golden_func, case_name="",
    exec_args=ExecArgs(run_mode="npu", enable_prof=False, prof_path="../", enable_dump=False)):
    test_cases = csv_to_testcase(csv_path, golden_func)
    executor = TestCaseExecutor(test_cases, exec_args)
    executor.run(case_name)