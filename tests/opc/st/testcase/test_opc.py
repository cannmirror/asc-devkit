#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Opc test case
"""
import unittest
import os
import sys
import json
from opc import OpcOptionParser
import opc as opc
import opc_api as opc_api
from constant import (GraphDtype, OpcCompileMode, OpcOptions, CompileParam, SupportInfo)
from op_compilation import OpCompilation
from opc_common import get_int64_mode
from op_info_store import load_set_op_content
from graph_parser import GraphParser
from op_compile_info_check import check_op_optional_paramtype
import tbe.common.utils.log as logger


class TestOpc01(unittest.TestCase):
    def set_up(self):
        pass

    def tear_down(self):
        pass

    def test_dir_01(self):
        parser = OpcOptionParser()
        ret = parser.check_dir_valid("xxx")
        self.assertEqual(ret, False)

    def test_dir_02(self):
        parser = OpcOptionParser()
        ret = parser.check_dir_valid("C:\\Program Files\\JetBrains\\PyCharm 2019.2\\helpers\\")
        self.assertEqual(ret, False)


current_dir = os.path.abspath(os.getcwd())
class TestOpc02(unittest.TestCase):
    def set_up(self):
        pass

    def tear_down(self):
        pass


    def check_options(self, options, expect_options):
        '''
        "ne" means not exsit.
        '''
        op_path = options.get("op_path", "")
        input_param = options.get("input_param", "not_exist")
        main_func = options.get("main_func", "not_exist")
        soc_version = options.get("soc_version", "not_exist")
        output = options.get("output", "not_exist")
        debug_dir = options.get("debug_dir", "not_exist")
        h = options.get("h", "not_exist")
        log = options.get("log", "not_exist")
        core_type = options.get("core_type", "AiCore")
        aicore_num = options.get("aicore_num", "not_exist")
        mdl_bank_path = options.get("mdl_bank_path", "not_exist")
        op_bank_path = options.get("op_bank_path", "not_exist")
        op_debug_level = options.get("op_debug_level", "not_exist")
        impl_mode = options.get("impl_mode", "not exist")

        self.assertEqual(expect_options["op_path"], op_path)
        self.assertEqual(expect_options["input_param"], input_param)
        self.assertEqual(expect_options["main_func"], main_func)
        self.assertEqual(expect_options["soc_version"], soc_version)
        logger.info("%s output %s", expect_options["output"], output)
        self.assertEqual(expect_options["output"], output)
        debug_dir_check = expect_options["debug_dir"] in debug_dir
        logger.info("%s output %s, debug_dir_check %s", expect_options["debug_dir"], debug_dir, debug_dir_check)
        self.assertEqual(debug_dir_check, True)
        logger.info("%s output %s", expect_options["h"], h)
        self.assertEqual(expect_options["h"], h)
        logger.info("%s output %s", expect_options["log"], log)
        self.assertEqual(expect_options["log"], log)
        logger.info("%s output %s", expect_options.get("core_type"), core_type)


    def test_options_parser_01(self):
        logger.info(current_dir)
        logger.info("Start to execute ==============test_options_parser_01==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"

        sys.argv = ["xx", "--op_path=./op_impl/dynamic/mat_mul.py", "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : "--op_path=./op_impl/dynamic/mat_mul.py",
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : current_dir +"/output",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "null",
            "core_type" : "AiCore",
        }
        self.assertEqual(opt_parser.check_input_params(), False)
        self.check_options(options, expect_options)


    def test_options_parser_02(self):
        '''
        op path use relative path, which does not exist
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_02==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"

        sys.argv = ["xx", "../../stub/files/mat_mul.py", "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : "../../stub/files/mat_mul.py",
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : current_dir +"/output",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "null",
            "core_type" : "AiCore",
        }
        self.assertEqual(opt_parser.check_input_params(), False)
        self.check_options(options, expect_options)


    def test_options_parser_03(self):
        '''
        input param: matmul_test_false.json does not exist
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_03==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test_false.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"
        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output"]

        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : current_dir +"/output",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "null",
            "core_type" : "AiCore",
        }
        self.assertEqual(opt_parser.check_input_params(), False)
        self.check_options(options, expect_options)


    def test_options_parser_04(self):
        '''
        op path is empty, --main_func=mat_mul will be
        considered as op path
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_04==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"

        sys.argv = ["xx", "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : "",
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : current_dir +"/output",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "null",
            "core_type" : "AiCore",
        }

        self.assertEqual(opt_parser.check_input_params(), True)
        self.check_options(options, expect_options)


    def test_options_parser_05(self):
        '''
        correct case
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_05==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : current_dir + "/output",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "null",
            "core_type" : None,
        }
        self.assertEqual(opt_parser.check_input_params(), True)
        self.check_options(options, expect_options)


    def test_options_parser_06(self):
        '''
        soc version is empty
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_06==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path,
                    "--output=./output"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "",
            "output" : "./output",
            "debug_dir" : "./",
            "h" : "False",
            "log" : "null",
            "core_type" : "AiCore",
        }
        self.assertEqual(opt_parser.check_input_params(), False)
        self.check_options(options, expect_options)


    def test_options_parser_07(self):
        '''
        main_func is empty, which is allowed
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_07==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path,
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : None,
            "soc_version" : "Ascend910A",
            "output" : current_dir + "/output",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "null",
            "core_type" : None,
        }
        self.assertEqual(opt_parser.check_input_params(), True)
        self.check_options(options, expect_options)


    def test_options_parser_08(self):
        '''
        output is empty, which will use current dir as output dir
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_08==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : current_dir + "/output",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "null",
            "core_type" : None,
        }
        self.assertEqual(opt_parser.check_input_params(), True)
        self.check_options(options, expect_options)


    def test_options_parser_09(self):
        '''
        debug dir is empty, which will use current dir(./) as debug dir
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_09==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : current_dir + "/output",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "null",
            "core_type" : None,
        }
        self.assertEqual(opt_parser.check_input_params(), True)
        self.check_options(options, expect_options)


    def test_options_parser_09_01(self):
        '''
        debug dir is not empty
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_09_01==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--op_debug_config=dump_cce",
                    "--output=./output", "--debug_dir=" + test_root_dir]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "op_debug_config": "dump_cce",
            "output" : current_dir + "/output",
            "debug_dir" : test_root_dir,
            "h" : "False",
            "log" : "null",
            "core_type" : None,
        }
        self.assertEqual(opt_parser.check_input_params(), True)
        self.check_options(options, expect_options)


    def test_options_parser_09_02(self):
        '''
        output not exist, opc will create
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_09_02==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./outputtemp"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : current_dir + "/outputtemp",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "null",
            "core_type" : None
        }
        self.assertEqual(opt_parser.check_input_params(), True)
        self.check_options(options, expect_options)
        #remove current_dir + "/outputtemp" create by opc
        if os.path.exists(current_dir + "/outputtemp"):
            logger.info("Delete existed output dir created by opc.")
            os.rmdir(current_dir + "/outputtemp")


    def test_options_parser_10(self):
        '''
        log is not empty
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_10==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output", "--log=debug"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : current_dir + "/output",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "debug",
            "core_type" : None,
        }
        self.assertEqual(opt_parser.check_input_params(), True)
        self.check_options(options, expect_options)
        self.assertEqual(opt_parser.get_option("log"), "debug")


    def test_options_parser_11(self):
        '''
        core_type is not empty
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_11==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output", "--log=debug", "--core_type=VectorCore"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : current_dir + "/output",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "debug",
            "core_type" : "VectorCore",
        }
        self.assertEqual(opt_parser.check_input_params(), True)
        self.check_options(options, expect_options)


    def test_options_parser_12(self):
        '''
        core_type is not empty
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_12==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output", "--log=debug", "--core_type=VectorCore"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : current_dir + "/output",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "debug",
            "core_type" : "VectorCore",
        }
        self.assertEqual(opt_parser.check_input_params(), True)
        self.check_options(options, expect_options)

    def test_options_parser_13(self):
        '''
        no arguments
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_13==============")
        sys.argv = ["xx"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : "",
            "input_param" : None,
            "main_func" : None,
            "soc_version" : "",
            "output" : "",
            "debug_dir" : "",
            "h" : "False",
            "log" : "null",
            "core_type" : "AiCore",
        }
        self.assertEqual(opt_parser.check_input_params(), False)
        self.check_options(options, expect_options)

    def test_options_parser_14(self):
        '''
        -h and --help only
        :return:
        '''
        logger.info("Start to execute ==============test_options_parser_14==============")
        sys.argv = ["xx"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : "",
            "input_param" : None,
            "main_func" : None,
            "soc_version" : "",
            "output" : "",
            "debug_dir" : "",
            "h" : "False",
            "log" : "null",
            "core_type" : "AiCore",
        }
        self.assertEqual(opt_parser.check_input_params(), False)
        self.check_options(options, expect_options)

    def test_options_parser_15(self):
        logger.info("Start to execute ==============test_options_parser_15==============")
        op_type = "conv2d"
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        path = test_root_dir + "/stub/files/build.txt"
        debug_path = test_root_dir + "/debug_dir"
        output_path = test_root_dir + "/output"
        sys.argv = ["xx", "--graph=" + path, "--soc_version=Ascend910", "--output=" + output_path,
                    "--debug_dir=" + debug_path, "--bin_filename=add1", "--op_debug_level=1",
                    "--op_debug_config=dump_cce", "--core_type=AiCore"]
        ret, opt_parser = opc.parse_args()
        result = opc.op_compile_classify(opt_parser)
        op_compile_mode = opt_parser.get_option(OpcOptions.OP_COMPILE_MODE)
        logger.debug("op_compile_mode is {}".format(op_compile_mode))
        self.assertEqual(op_compile_mode, "single_op_compile_graph_mode")

        if op_compile_mode != "single_op_compile_config_file_mode":
            graph_parser = GraphParser(opt_parser.get_all_options())
            graph_parser.group_compile_param()
        json_path = test_root_dir + "/stub/files/ascend910.json"
        load_set_op_content(json_path)
        op_compile = OpCompilation(opt_parser.get_all_options())
        ret = op_compile.op_compilation()
        self.assertEqual(ret, True)

        logger.info("End to execute ==============test_options_parser_15==============")

    def test_options_parser_16(self):
        logger.info("Start to execute ==============test_options_parser_16==============")
        op_type = "leaky_relu"
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        path = test_root_dir + "/stub/files/build_1.txt"
        debug_path = test_root_dir + "/debug_dir"
        output_path = test_root_dir + "/output"
        sys.argv = ["xx", "--graph=" + path, "--soc_version=Ascend910A", "--output=" + output_path,
                    "--debug_dir=" + debug_path, "--bin_filename=add3", "--core_type=AiCore"]
        ret, opt_parser = opc.parse_args()
        result = opc.op_compile_classify(opt_parser)
        op_compile_mode = opt_parser.get_option(OpcOptions.OP_COMPILE_MODE)
        self.assertEqual(op_compile_mode, "fusion_op_compile_graph_mode")
        logger.debug("op_compile_mode is {}".format(op_compile_mode))

        if op_compile_mode != "single_op_compile_config_file_mode":
            graph_parser = GraphParser(opt_parser.get_all_options())
            graph_parser.group_compile_param()
        op_compile = OpCompilation(opt_parser.get_all_options())
        ret = op_compile.op_compilation()
        self.assertEqual(ret, True)
        logger.info("End to execute ==============test_options_parser_16==============")

    def test_options_parser_16_2(self):
        logger.info("Start to execute ==============test_options_parser_16_2==============")
        op_type = "leaky_relu"
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        path = test_root_dir + "/stub/files/build_1.txt"
        debug_path = test_root_dir + "/debug_dir"
        output_path = test_root_dir + "/output"
        sys.argv = ["xx", "--graph=" + path, "--soc_version=Ascend910A", "--output=" + output_path,
                    "--debug_dir=" + debug_path, "--bin_filename=add4"]
        ret, opt_parser = opc.parse_args()
        result = opc.op_compile_classify(opt_parser)
        op_compile_mode = opt_parser.get_option(OpcOptions.OP_COMPILE_MODE)
        self.assertEqual(op_compile_mode, "fusion_op_compile_graph_mode")
        logger.debug("op_compile_mode is %s", op_compile_mode)
        graph_parser = GraphParser(opt_parser.get_all_options())
        graph_parser.group_compile_param()
        op_compile = OpCompilation(opt_parser.get_all_options())
        ret = op_compile.op_compilation()
        self.assertEqual(ret, True)
        logger.info("End to execute ==============test_options_parser_16_2==============")

    def test_options_parser_17(self):
        logger.info("Start to execute ==============test_options_parser_17==============")
        op_type = "leaky_relu"
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        path = test_root_dir + "/stub/files/build_2.txt"
        debug_path = test_root_dir + "/debug_dir"
        output_path = test_root_dir + "/output"
        sys.argv = ["xx", "--graph=" + path, "--soc_version=Ascend910A", "--output=" + output_path,
                    "--debug_dir=" + debug_path, "--bin_filename=add2", "--core_type=AiCore"]
        ret, opt_parser = opc.parse_args()
        result = opc.op_compile_classify(opt_parser)
        op_compile_mode = opt_parser.get_option(OpcOptions.OP_COMPILE_MODE)
        self.assertEqual(op_compile_mode, "fusion_op_compile_graph_mode")
        logger.debug("op_compile_mode is {}".format(op_compile_mode))

        if op_compile_mode != "single_op_compile_config_file_mode":
            graph_parser = GraphParser(opt_parser.get_all_options())
            graph_parser.group_compile_param()
        op_compile = OpCompilation(opt_parser.get_all_options())
        ret = op_compile.op_compilation()
        self.assertEqual(ret, True)

    def test_options_parser_18(self):
        '''
        test invalid core_type: ai_vector
        '''
        logger.info("Start to execute ==============test_options_parser_18==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output", "--log=debug", "--core_type=ai_vector"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        expect_options = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : current_dir + "/output",
            "debug_dir" : current_dir,
            "h" : "False",
            "log" : "debug",
            "core_type" : "ai_vector"
        }
        self.assertEqual(opt_parser.check_input_params(), False)
        self.check_options(options, expect_options)


    def test_options_parser_19(self):
        '''
        test invalid aicore_num: sih
        '''
        logger.info("Start to execute ==============test_options_parser_19==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output", "--log=debug", "--aicore_num=sih"]
        try:
            ret, opt_parser = opc.parse_args()
        except SystemExit as e:
            logger.info("aicore_num only support digit.")
            self.assertEqual("1", str(e))


    def test_options_parser_20(self):
        '''
        test invalid mdl_bank_path
        '''
        logger.info("Start to execute ==============test_options_parser_20==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output", "--log=debug", "--mdl_bank_path=dffgfds"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        self.assertEqual(opt_parser.check_input_params(), True)

    def test_options_parser_21(self):
        '''
        test invalid op_bank_path
        '''
        logger.info("Start to execute ==============test_options_parser_21==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output", "--log=debug", "--op_bank_path=dffgfds"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        self.assertEqual(opt_parser.check_input_params(), True)

    def test_options_parser_22(self):
        '''
        test invalid op_debug_level
        '''
        logger.info("Start to execute ==============test_options_parser_22==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output", "--log=debug", "--op_debug_level=5"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        self.assertEqual(opt_parser.check_input_params(), False)


    def test_options_parser_23(self):
        '''
        test invalid impl_mode: ascend
        '''
        logger.info("Start to execute ==============test_options_parser_23==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output", "--log=debug", "--impl_mode=high_precision,enable_hi_float_32_execution"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        self.assertEqual(opt_parser.check_input_params(), True)


    def test_options_parse_24(self):
        '''
        test valid core_type, aicore_num, mdl_bank_path, op_bank_path, impl_mode
        '''
        logger.info("Start to execute ==============test_options_parse_24==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        path = test_root_dir + "/output"
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output", "--log=debug", "--core_type=AiCore",
                    "--aicore_num=1", "--mdl_bank_path=" + path, "--op_bank_path=" + path,
                    "--impl_mode=high_performance"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        self.assertEqual(opt_parser.check_input_params(), True)

    def test_options_parse_25(self):
        '''
        test valid core_type, aicore_num, mdl_bank_path, op_bank_path, impl_mode
        '''
        logger.info("Start to execute ==============test_options_parse_25==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        path = test_root_dir + "/output"
        input_param_path = test_root_dir + "/stub/files/matmul_test.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"

        sys.argv = ["xx", op_path, "--main_func=mat_mul",
                    "--input_param=" + input_param_path, "--soc_version=Ascend910A",
                    "--output=./output", "--log=debug", "--core_type=AiCore,VectorCore",
                    "--aicore_num=1", "--mdl_bank_path=" + path, "--op_bank_path=" + path,
                    "--impl_mode=high_performance", "--simplified_key_mode=0"]
        ret, opt_parser = opc.parse_args()
        opc.op_compile_classify(opt_parser)
        options = opt_parser.get_all_options()
        self.assertEqual(opt_parser.check_input_params(), True)

    def test_options_parser_26(self):
        logger.info("Start to execute ==============test_options_parser_26==============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        path = test_root_dir + "/stub/files/conv2d_2inputs_conv_relu_dynamic.txt"
        debug_path = test_root_dir + "/debug_dir"
        output_path = test_root_dir + "/output"
        sys.argv = ["xx", "--graph=" + path, "--soc_version=Ascend910A", "--output=" + output_path,
                    "--debug_dir=" + debug_path, "--bin_filename=add1", "--op_debug_level=1"]
        ret, opt_parser = opc.parse_args()
        result = opc.op_compile_classify(opt_parser)
        op_compile_mode = opt_parser.get_option(OpcOptions.OP_COMPILE_MODE)
        logger.debug("op_compile_mode is {}".format(op_compile_mode))

        graph_parser = GraphParser(opt_parser.get_all_options())
        graph_parser.group_compile_param()
        self.assertEqual("a_value_range" in str(opt_parser.get_all_options()), True)

    def test_check_op_optional_paramtype(self):
        """test check_op_optional_paramtype"""
        logger.debug("Start to execute ============ test_check_op_optional_paramtype ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/apply_adam_v2.json"
        op_path = test_root_dir + "/stub/files/apply_adam_v2.py"
        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output",
            "debug_dir" : test_root_dir + "/debug_dir",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level": 3,
            "op_debug_config": "dump_cce",
            "optional_input_mode": "gen_placeholder"
        }
        try:
            with open(input_param_path, "r") as file_handle:
                json_dict = json.load(file_handle)
        except Exception as e:
            logger.error("load file[%s] failed, reason: %s.", input_param_path, str(e))
            return False, None
        finally:
            pass
        ret = check_op_optional_paramtype(json_dict, opc_compile_args)
        self.assertEqual(ret, True)
        logger.debug("End to execute ============ test_check_op_optional_paramtype ============")

    def test_get_int64_mode(self):
        """test_get_int64_mode"""
        logger.debug("Start to execute ============ test_get_int64_mode ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/ConcatD.json"
        with open(input_param_path, "r") as file_in:
            json_data = json.load(file_in)
        op_list = json_data.get("op_list")
        for op_operator in op_list:
            ret = get_int64_mode(op_operator.get("inputs"))
            self.assertEqual(ret, False)
            ret = get_int64_mode(op_operator.get("outputs"))
            self.assertEqual(ret, False)

        logger.debug("End to execute ============ test_get_int64_mode ============")

    def test_compile_op_api(self):
        """test_compile_op_api"""
        logger.debug("Start to execute ============ test_compile_op_api ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/sqrt.json"
        output_dir = test_root_dir + "/output"
        debug_dir = test_root_dir + "/debug_dir"
        op_path = test_root_dir + "/stub/files/sqrt.py"
        with open(input_param_path, "r") as file_in:
            json_data = json.load(file_in)

        build_options = {
            "op_path" : op_path,
            "soc_version" : "Ascend910A",
            "output" : output_dir,
            "debug_dir" : debug_dir,
            "optional_input_mode": "gen_placeholder",
            "bin_filename": "ConcatD"
        }
        res = opc_api.compile_op(json_data, build_options)
        self.assertEqual(res, True)

        logger.debug("End to execute ============ test_compile_op_api ============")

    def test_op_mode_is_single_op_1(self):
        logger.debug("Start to execute ============ test_op_mode_is_single_op_1 ============")
        self.assertEqual(opc.op_mode_is_single_op("single_op_compile_config_file_mode"), True)
        logger.debug("End to execute ============ test_op_mode_is_single_op_1 ============")

    def test_op_mode_is_single_op_2(self):
        logger.debug("Start to execute ============ test_op_mode_is_single_op_2 ============")
        self.assertEqual(opc.op_mode_is_single_op("fusion_op_compile_graph_mode"), False)
        logger.debug("End to execute ============ test_op_mode_is_single_op_2 ============")


if __name__ == '__main__':
    unittest.main()
