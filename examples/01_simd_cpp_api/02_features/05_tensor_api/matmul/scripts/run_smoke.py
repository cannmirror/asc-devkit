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
import logging

from gen_golden import gen_golden
from common import cpp_to_py_dtype
from gen_demo_code_by_case import gen_demo_code
from verify_result import verify_result


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

work_dir = os.getenv("WORK_DIR")


def build_demo(demo_dir):
    os.chdir(demo_dir)
    cmake_cmd = "cmake -S . -B ./build"
    os.system(cmake_cmd)

    make_cmd = "make -C ./build"
    os.system(make_cmd)


def run_demo(x, y, bias, quant, z):
    demo_exe = os.path.join(work_dir, "build/demo")
    run_cmd = f"{demo_exe} {x} {y} {bias} {quant} {z}"
    os.system(run_cmd)


def run():
    failed_cases = []
    
    logging.info("Step-0: Begin running...")
    input_cases_path = os.path.join(work_dir, "testcases/testcases.json")
    testcases = {}
    with open(input_cases_path, "r", encoding="utf-8") as f:
        testcases = json.load(f)
    logging.info("Step-1: gen_testcases finish...")

    for testcase in testcases["testcases"]:
        case_name = testcase["name"]
        golden_path = os.path.join(work_dir, "build", case_name)
        gen_golden(testcase, golden_path)
        logging.info(f"Step-2: case {case_name} gen_golden finish...")

        demo_dir = os.path.join(work_dir, "demo")
        gen_demo_code(
            testcase,
            os.path.join(work_dir, "matmul.asc"),
            os.path.join(work_dir, "matmul_custom.asc")
        )
        logging.info(f"Step-3: case {case_name} gen_demo_code finish...")

        build_demo(work_dir)
        x_path = os.path.join(golden_path, "input_x.bin")
        y_path = os.path.join(golden_path, "input_y.bin")
        bias_path = os.path.join(golden_path, "input_bias.bin")
        quant_path = os.path.join(golden_path, "input_quant.bin")
        z_path = os.path.join(golden_path, "output.bin")
        run_demo(x_path, y_path, bias_path, quant_path, z_path)
        logging.info(f"Step-4: case {case_name} run_demo finish...")

        c_type = cpp_to_py_dtype[testcase["dtype"][2]]
        res = verify_result(z_path, os.path.join(golden_path, "golden.bin"), c_type)
        logging.info(f"Step-5: case {case_name} verify_result finish...")

        if not res:
            logging.info(f"case {testcase['name']} failed!")
            failed_cases.append(case_name)
        else:
            logging.info(f"case {testcase['name']} success!")
    if failed_cases:
        logging.error(f"🔴 以下测试用例失败：")
        for case in failed_cases:
            logging.error(f"    - {case}")
        logging.error("❌ 请检查上述用例的日志和输出。")
    else:
        logging.info("🟢 所有测试用例均成功通过！")
    

if __name__ == "__main__":
    run()
    logging.info(f"All testcases finish...")
