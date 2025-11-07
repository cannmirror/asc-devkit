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
import unittest.mock
import os
import tbe.common.utils.log as logger
from pathlib import Path
import json
import tempfile

from constant import OpcCompileMode, OpcCompileMode
from op_compilation import OpCompilation, OpcOptions
from post_compile_base import PostCompilation
from fusion_op_post_compile import FusionOpPostCompile
from single_op_post_compile import SingleOpPostCompile
from tbe.common.platform import platform_info
from opc_common import normalize_optional_impl_mode, get_new_attrs_for_op_compile
from single_op_compile import SingleOpCompile
from op_info_store import OpKernelInfo, SubOpInfoStore, load_op_info_store, load_set_op_content
from op_manager import (get_compute_op_func, get_single_op_operator, get_core_type_from_op_content)
from op_tensor_utils import (compress_node, trans_matmulcompress_bias_shape, trans_matmul_bias_shape,
                             trans_bninference_shape, trans_fully_connection, trans_batch_matmul_shape,
                             trans_ascend_quant_shape, trans_arequant_s16, trans_conv3d_shape, trans_depthwise_conv2d,
                             trans_biasadd_shape, trans_mul_shape, trans_elemwise_shape, trans_shape_fullycompress,
                             _inner_replace_cube_tvm_shapes, replace_cube_tvm_shapes)
from fusion_op_utils import (get_extra_params, set_l1_fusion_type, get_fusion_pattern, get_classify_info,
                             get_op_inputs_args, get_fusion_build_cfg)
from op_compile_info_check import check_op_compilation_json

class TestOpCompilationSt(unittest.TestCase):
    def set_up(self):
        pass

    def tear_down(self):
        pass

    def del_files(self, file_path):
        if not os.path.isdir(file_path):
            logger.error("file_path[%s] not exist.", file_path)
            return

        for file in os.listdir(file_path):
            to_del_file = "{}/{}".format(file_path, file)
            try:
                os.remove(to_del_file)
                logger.info("Deleta file[%s] success", to_del_file)
            except Exception as e:
                logger.error("Delete file[%s] filed, reason: %s.", to_del_file, str(e))
            finally:
                pass

    def test_00_op_compilation_st(self):
        """test op_compilation"""
        logger.debug("Start to execute ============ test_00_op_compilation_st ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/MatMul.json"
        op_path = test_root_dir + "/stub/files/MatMul.py"
        json_path = test_root_dir + "/stub/files/ascend910.json"
        with open(json_path, "r") as file_in:
            op_builtin_info_dict = json.load(file_in)
        SubOpInfoStore().set_op_content(op_builtin_info_dict)

        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "debug_dir" : test_root_dir + "/debug_dir", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
             "impl_mode" : "",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level" : 1
        }
        op_compile = OpCompilation(opc_compile_args)
        # if opc process success, op stub func(matmul) will copy json and o file from stub file dir to debug_dir dir,
        # opc will copy json and o file from debug_dir dir to output dir
        # opc will finally delete file in output dir
        ret = op_compile.op_compilation()

        # if ret is false, it means opc process failed and throw exceptions. Please check if your codes are correct
        self.assertEqual(ret, True)

        # execute again, output dir has json files, opc process will check file exist
        ret = op_compile.op_compilation()
        self.assertEqual(ret, True)
        ret, json_dict = check_op_compilation_json(OpcOptions.INPUT_PARAM, opc_compile_args)
        self.assertEqual(ret, True)
        ret = op_compile.check_and_update_core_type(OpcCompileMode.SINGLE_OP_CONFIG_FILE_MODE, json_dict)
        self.assertEqual(ret, True)
        # delete file in output
        logger.debug("Testcase delete files in output dir")
        self.del_files(test_root_dir + "/output")
        logger.debug("End to execute ============ test_00_op_compilation_ut ============")

    def test_01_op_compilation(self):
        """test op_compilation"""
        logger.debug("Start to execute ============ test_01_op_compilation ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_1.json"
        op_path = test_root_dir + "/stub/files/MatMul.py"
        json_path = test_root_dir + "/stub/files/ascend910.json"
        with open(json_path, "r") as file_in:
            op_builtin_info_dict = json.load(file_in)
        SubOpInfoStore().set_op_content(op_builtin_info_dict)

        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output",
            "debug_dir" : test_root_dir + "/debug_dir",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level": 3,
            "core_type" : "AiCore"
        }
        op_compile = OpCompilation(opc_compile_args)
        ret = op_compile.op_compilation()
        self.assertEqual(ret, True)
        self.del_files(test_root_dir + "/output")
        logger.debug("End to execute ============ test_01_op_compilation ============")

    def test_02_op_compilation(self):
        """test op_compilation"""
        logger.debug("Start to execute ============ test_02_op_compilation ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_1.json"
        op_path = test_root_dir + "/stub/files/MatMul.py"
        json_path = test_root_dir + "/stub/files/ascend910.json"
        with open(json_path, "r") as file_in:
            op_builtin_info_dict = json.load(file_in)
        SubOpInfoStore().set_op_content(op_builtin_info_dict)

        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output",
            "debug_dir" : test_root_dir + "/debug_dir",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level": 3,
            "simplified_key_mode" : 0,
            "optional_input_mode": "gen_placeholder",
            "dynamic_param_mode": "folded_with_desc"
        }
        op_compile = OpCompilation(opc_compile_args)
        ret = op_compile.op_compilation()
        self.assertEqual(ret, True)
        self.del_files(test_root_dir + "/output")
        logger.debug("End to execute ============ test_02_op_compilation ============")

    def test_03_op_compilation_st(self):
        """test test_03_op_compilation_st"""
        logger.debug("Start to execute ============ test_03_op_compilation_st ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/MatMul.json"
        op_path = test_root_dir + "/stub/files/MatMul.py"
        json_path = test_root_dir + "/stub/files/ascend910.json"
        with open(json_path, "r") as file_in:
            op_builtin_info_dict = json.load(file_in)
        SubOpInfoStore().set_op_content(op_builtin_info_dict)

        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "debug_dir" : test_root_dir + "/debug_dir", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
             "impl_mode" : "optional,high_performance",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level" : 1,
            "simplified_key_mode" : 0
        }
        op_compile = OpCompilation(opc_compile_args)
        # if opc process success, op stub func(matmul) will copy json and o file from stub file dir to debug_dir dir,
        # opc will copy json and o file from debug_dir dir to output dir
        # opc will finally delete file in output dir
        ret = op_compile.op_compilation()

        # if ret is false, it means opc process failed and throw exceptions. Please check if your codes are correct
        self.assertEqual(ret, True)
        logger.debug("Testcase delete files in output dir")
        self.del_files(test_root_dir + "/output")
        logger.debug("End to execute ============ test_03_op_compilation_st ============")

    def test_compile_cann_ub_search_st(self):
        """test test_compile_cann_ub_search_st"""
        logger.debug("Start to execute ============ test_compile_cann_ub_search_st ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/gather_v2.json"
        op_path = test_root_dir + "/stub/files/gather_v2.py"
        json_path = test_root_dir + "/stub/files/ascend910.json"
        load_set_op_content(json_path)

        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output",
            "debug_dir" : test_root_dir + "/debug_dir", 
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "impl_mode" : "optional,high_performance",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level" : 1,
            "simplified_key_mode" : 0
        }
       
        op_compile = OpCompilation(opc_compile_args)
        ret = op_compile.op_compilation()
        self.assertEqual(ret, True)
        logger.debug("Testcase delete files in output dir")
        self.del_files(test_root_dir + "/output")
        logger.debug("End to execute ============ test_compile_cann_ub_search_st ============")

    def test_check_and_update_core_type(self):
        logger.debug("Start to execute ============ test_check_and_update_core_type ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/MatMul_no_op_type.json"
        op_path = test_root_dir + "/stub/files/MatMul.py"
        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "debug_dir" : test_root_dir + "/debug_dir", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : None,
            "impl_mode" : "high_performance, optional",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level" : 1
        }
        op_compile = OpCompilation(opc_compile_args)
        ret = op_compile.op_compilation()
        self.assertEqual(ret, False)

        ret, json_dict = check_op_compilation_json(OpcOptions.INPUT_PARAM, opc_compile_args)
        self.assertEqual(ret, True)
        ret = op_compile.check_and_update_core_type(OpcCompileMode.SINGLE_OP_CONFIG_FILE_MODE, json_dict)
        self.assertEqual(ret, False)
        json_dict["op_type"] = 'MatMul'
        platform_info.set_soc_spec({"cube_vector_combine" : "split"})
        ret = op_compile.check_and_update_core_type(OpcCompileMode.SINGLE_OP_CONFIG_FILE_MODE, json_dict)
        self.assertEqual(ret, True)
        platform_info.set_soc_spec({"cube_vector_combine" : "fuse"})
        ret = op_compile.check_and_update_core_type(OpcCompileMode.SINGLE_OP_CONFIG_FILE_MODE, json_dict)
        self.assertEqual(ret, True)

        op_compile = OpCompilation(opc_compile_args)
        ret = op_compile.op_compilation()
        self.assertEqual(ret, False)
        ret = get_core_type_from_op_content('')
        self.assertEqual(ret, None)
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_path = test_root_dir + "/stub/files/ascend910.json"
        with open(json_path, "r") as file_in:
            op_builtin_info_dict = json.load(file_in)
        SubOpInfoStore().set_op_content(op_builtin_info_dict)
        ret = get_core_type_from_op_content('MatMul')
        self.assertEqual(ret, 'AiCore,VectorCore')
        ret = get_core_type_from_op_content('Add')
        self.assertEqual(ret, None)
        logger.debug("End to execute ============ test_check_and_update_core_type ============")

    def test_01_op_compilation_json_file_error_st(self):
        """test op_compilation json_file_error"""
        logger.debug("Start to execute ============ test_01_op_compilation_json_file_error_st ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_test.json" # json file is none, json load fail
        op_path = test_root_dir + "/stub/files/MatMul.py"
        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output",
            "debug_dir" : test_root_dir + "/debug_dir",
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "op_compile_classify": "single_op_compile_config_file_mode",
        }
        op_compile = OpCompilation(opc_compile_args)

        ret = op_compile.op_compilation()
        self.assertEqual(ret, False)

        input_param_path = test_root_dir + "/stub/files/matmul_test_notdict.json" # json file not dict
        opc_compile_args["input_param"] = input_param_path
        op_compile = OpCompilation(opc_compile_args)

        ret = op_compile.op_compilation()
        self.assertEqual(ret, False)

        input_param_path = test_root_dir + "/stub/files/MatMul_build_res.json" # json file has no op_list
        opc_compile_args["input_param"] = input_param_path
        op_compile = OpCompilation(opc_compile_args)

        ret = op_compile.op_compilation()
        self.assertEqual(ret, False)

        logger.debug("End to execute ============ test_01_op_compilation_json_file_error_st ============")

    def test_02_op_compilation_mode64_st(self):
        """test op_compilation mode 64: large shape"""
        logger.debug("Start to execute ============ test_02_op_compilation_mode64_st ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/MatMul_mode64.json"
        op_path = test_root_dir + "/stub/files/MatMul.py"
        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "debug_dir" : test_root_dir + "/debug_dir", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "op_compile_classify": "single_op_compile_config_file_mode",
        }
        op_compile = OpCompilation(opc_compile_args)
        # if opc process success, op stub func(matmul) will copy json and o file from stub file dir to debug_dir dir,
        # opc will copy json and o file from debug_dir dir to output dir
        # opc will finally delete file in output dir
        ret = op_compile.op_compilation()

        # if ret is false, it means opc process failed and throw exceptions. Please check if your codes are correct
        self.assertEqual(ret, True)

        # delete file in output
        logger.debug("Testcase delete files in output dir")
        self.del_files(test_root_dir + "/output")

        logger.debug("End to execute ============ test_02_op_compilation_mode64_st ============")

    def test_op_compilation_input_options_st(self):
        """test op_compilation mode 64: large shape"""
        logger.debug("Start to execute ============ test_op_compilation_input_options_st ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/apply_adam_v2.json"
        op_path = test_root_dir + "/stub/files/apply_adam_v2.py"
        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "debug_dir" : test_root_dir + "/debug_dir", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "optional_input_mode": "gen_placeholder",
            "op_compile_classify": "single_op_compile_config_file_mode",
        }
        json_path = test_root_dir + "/stub/files/ascend910.json"
        with open(json_path, "r") as file_in:
            op_builtin_info_dict = json.load(file_in)
        SubOpInfoStore().set_op_content(op_builtin_info_dict)

        op_compile = OpCompilation(opc_compile_args)
        ret = op_compile.op_compilation()
        self.assertEqual(ret, True)

        # delete file in output
        logger.debug("Testcase delete files in output dir")
        self.del_files(test_root_dir + "/output")

        logger.debug("End to execute ============ test_op_compilation_input_options_st ============")

    def test_03_record_compile_error_info_st(self):
        """test record_compile_error_info"""
        logger.debug("Start to execute ============ test_03_record_compile_error_info_st ============")
        op_info = {"op_type": "matmul",
                   "kernel_name": "kernel_namexxx"}
        idx = 1
        op = {
            "comment": "ND_float16 with attr = true",
            "inputs": "inputs"
        }
        error_info = "errorinfo"
        OpCompilation.record_compile_error_info(op_info, idx, op, error_info)

        logger.debug("End to execute ============ test_03_record_compile_error_info_st ============")

    def test_04_create_kernel_meta_dir_st(self):
        """test op_compilation"""
        logger.debug("Start to execute ============ test_04_create_kernel_meta_dir_st ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_3.json"
        op_path = test_root_dir + "/stub/files/MatMul.py"
        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "debug_dir" : test_root_dir + "/debug_dir", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level" : 2
        }
        op_compile = OpCompilation(opc_compile_args)
        # if opc process success, op stub func(matmul) will copy json and o file from stub file dir to debug_dir dir,
        # opc will copy json and o file from debug_dir dir to output dir
        # opc will finally delete file in output dir
        ret = op_compile.op_compilation()

        # try to test create kernel_mata_xxx dir by opc, so we can not stub the file under the kernel_meta dir
        # There will be a runtime error while copying file from kernel meta dir to output dir
        self.assertEqual(ret, True)

        # delete file in output
        logger.debug("Testcase delete files in output dir")
        self.del_files(test_root_dir + "/output")

        logger.debug("End to execute ============ test_04_create_kernel_meta_dir_st ============")

    def test_05_verify_kernel_meta_lock_st(self):
        """test op_compilation"""
        logger.debug("Start to execute ============ test_05_verify_kernel_meta_lock_st ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/matmul_4.json"
        op_path = test_root_dir + "/stub/files/MatMul.py"
        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "debug_dir" : test_root_dir + "/debug_dir", # path will be used in llt\atc\opcompiler\opc\stub\files\MatMul.py
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level" : 2
        }
        op_compile = OpCompilation(opc_compile_args)
        # if opc process success, op stub func(matmul) will copy json and o file from stub file dir to debug_dir dir,
        # opc will copy json and o file from debug_dir dir to output dir
        # opc will finally delete file in output dir
        ret = op_compile.op_compilation()

        # try to test create kernel_mata_xxx dir by opc, so we can not stub the file under the kernel_meta dir
        # There will be a runtime error while copying file from kernel meta dir to output dir
        self.assertEqual(ret, True)

        # delete file in output
        logger.debug("Testcase delete files in output dir")
        self.del_files(test_root_dir + "/output")

        logger.debug("End to execute ============ test_05_verify_kernel_meta_lock_st ============")

    def test_generate_supportinfo_01_st(self):
        logger.debug("Start to execute ============ test_generate_supportinfo_01_st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_dict_path = test_root_dir + "/stub/files/json_dict.json"
        test_debug_dir = test_root_dir + "/debug_dir/kernel_meta/testSupportInfo.json"
        opc_compile_args = {
            "debug_dir" : test_root_dir + "/debug_dir",
            "bin_filename" : "add12343",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output",
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "impl_mode" : "high_performance, optional",
        }
        fusion_impl_mode = dict()
        fusion_impl_mode["Conv2d"] = ("high_performance", True)
        with open(json_dict_path, "r") as file_in:
            json_info = json.load(file_in)
        l1_fusion_flag = "false"
        l2_fusion_flag = "false"
        l2_mode_flag = "0"
        fusion_post_compilation = FusionOpPostCompile(opc_compile_args, l1_fusion_flag, l2_fusion_flag, l2_mode_flag, fusion_impl_mode)
        single_post_compilation = SingleOpPostCompile(opc_compile_args, l1_fusion_flag, l2_fusion_flag, l2_mode_flag, fusion_impl_mode)
        try:
            fusion_post_compilation.update_supportinfo_to_json_file(json_info, test_debug_dir)
        except RuntimeError as e:
            logger.debug("Exception occured as expected.%s", str(e))
        op_info = {
            "op_type": "Add",
            "kernel_name": "add12343",
            "kernel_meta_path":test_debug_dir,
            "inputs": [
                {
                    "name": "x1",
                    "index": 0,
                    "dtype": "float16",
                    "format": "FRACTAL_NZ",
                    "ori_format": "ND",
                    "shape": [
                        -2
                    ]
                },
                None
            ],
            "outputs": [
                {
                    "name": "x1",
                    "index": 0,
                    "dtype": "float16",
                    "format": "FRACTAL_NZ",
                    "ori_format": "ALL",
                    "shape": [
                        -2
                    ]
                }
            ],
            "attrs": [
                {
                    'name': 'adj_x1',
                    'dtype': 'bool',
                    'value': False
                },
                None
            ]
        }
        single_post_compilation.update_info_to_json_file(op_info, opc_compile_args, test_debug_dir)
        op = {
            "is_dynamic_impl": True,
            "graph_pattern": "123456",
            "op_list": [{
                "inputs": [
                    {
                        "name": "x1",
                        "index": 0,
                        "dtype": "float16",
                        "format": "FRACTAL_NZ",
                        "ori_format": "ND",
                        "shape": [
                            -2
                        ]
                    },
                    None
                ],
                "outputs": [
                    {
                        "name": "x1",
                        "index": 0,
                        "dtype": "float16",
                        "format": "FRACTAL_NZ",
                        "ori_format": "ND",
                        "shape": [
                            -2
                        ]
                    }
                ],
                "op_attrs": [
                    {
                        'name': 'adj_x1',
                        'dtype': 'bool',
                        'value': False
                    }
                ]
            }]
        }
        try:
            fusion_post_compilation.update_supportinfo_to_json_file(op, test_debug_dir)
        except RuntimeError as e:
            logger.debug("Exception occured as expected.%s", str(e))

        dyn_op = {
            "op_type": "Add",
            "kernel_name": "add12343",
            "kernel_meta_path":test_debug_dir,
                "inputs":[
                    [
                        {
                            "name": "input_values",
                            "index": 0,
                            "dtype": "float16",
                            "format": "ND",
                            "paramType": "dynamic",
                            "shape": [
                                -2
                            ],
                            "format_match_mode": "FormatAgnostic",
                            "dtype_match_mode": "DtypeByte"
                        },
                        {
                            "name": "input_values",
                            "index": 0,
                            "dtype": "float16",
                            "format": "ND",
                            "paramType": "dynamic",
                            "shape": [
                                -2
                            ],
                            "format_match_mode": "FormatAgnostic",
                            "dtype_match_mode": "DtypeByte"
                        }
                    ]
                ],
                "outputs": [
                    {
                        "name": "x1",
                        "index": 0,
                        "dtype": "float16",
                        "format": "FRACTAL_NZ",
                        "ori_format": "ND",
                        "shape": [
                            -2
                        ]
                    }
                ],
                "attrs": [
                    {
                        'name': 'var_attrs',
                        'dtype': 'string',
                        'value': 'aa'
                    }
                ]
        }
        single_post_compilation.update_info_to_json_file(dyn_op, dyn_op, test_debug_dir)
        dyn_op = {
            "op_type": "Add",
            "kernel_name": "add12343",
            "kernel_meta_path":test_debug_dir,
                "inputs":[
                    [
                        {
                            "name": "input_values",
                            "index": 0,
                            "dtype": "float16",
                            "format": "ND",
                            "paramType": "dynamic",
                            "shape": [
                                -2
                            ],
                            "format_match_mode": "FormatAgnostic",
                            "dtype_match_mode": "DtypeByte"
                        },
                        {
                            "name": "input_values",
                            "index": 0,
                            "dtype": "float16",
                            "format": "ND",
                            "paramType": "dynamic",
                            "shape": [
                                -2
                            ],
                            "format_match_mode": "FormatAgnostic",
                            "dtype_match_mode": "DtypeByte"
                        }
                    ]
                ],
                "outputs": [
                    {
                        "name": "x1",
                        "index": 0,
                        "dtype": "float16",
                        "format": "FRACTAL_NZ",
                        "shape": [
                            -2
                        ]
                    }
                ],
                "attrs": [
                    {
                        'name': 'var_attrs',
                        'dtype': 'string',
                        'value': 'aa'
                    }
                ]
        }
        single_post_compilation.update_info_to_json_file(dyn_op, dyn_op, test_debug_dir)
        logger.debug("End to execute ============ test_generate_supportinfo_01_st ============")

    def test_generate_supportinfo_02_ut(self):
        logger.debug("Start to execute ============ test_generate_supportinfo_02_ut ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_dict_path = test_root_dir + "/stub/files/json_dict.json"
        test_debug_dir = test_root_dir + "/debug_dir/kernel_meta/testSupportInfo.json"
        opc_compile_args = {
            "debug_dir" : test_root_dir + "/debug_dir",
            "bin_filename" : "add12343",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output",
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "optional_input_mode" : "no_placeholder",
            "impl_mode" : "high_performance, optional",
            "simplified_key_mode" : 1,
        }
        fusion_impl_mode = dict()
        fusion_impl_mode["Conv2d"] = ("high_performance", True)
        with open(json_dict_path, "r") as file_in:
            json_info = json.load(file_in)
        l1_fusion_flag = "false"
        l2_fusion_flag = "false"
        l2_mode_flag = "0"
        fusion_post_compilation = FusionOpPostCompile(opc_compile_args, l1_fusion_flag, l2_fusion_flag, l2_mode_flag, fusion_impl_mode)
        single_post_compilation = SingleOpPostCompile(opc_compile_args, l1_fusion_flag, l2_fusion_flag, l2_mode_flag, fusion_impl_mode)
        fusion_post_compilation.update_supportinfo_to_json_file(json_info,  test_debug_dir)
        op_info = {
            "op_type": "Add",
            "kernel_name": "add12343",
            "kernel_meta_path":test_debug_dir,
            "inputs": [
                {
                    "name": "x1",
                    "index": 0,
                    "dtype": "float16",
                    "format": "FRACTAL_NZ",
                    "shape": [
                        -2
                    ],
                    "formatMode": "nd",
                },
                None
            ],
            "outputs": [
                {
                    "name": "x1",
                    "index": 0,
                    "dtype": "float16",
                    "format": "FRACTAL_NZ",
                    "shape": [
                        -2
                    ]
                }
            ],
            "attrs": [
                {
                    'name': 'adj_x1',
                    'dtype': 'bool',
                    'value': False
                },
                None
            ]
        }
        single_post_compilation.update_info_to_json_file(op_info, op_info, test_debug_dir)
        logger.debug("End to execute ============ test_generate_supportinfo_02_ut ============")


    def test_construct_op_kernel_info_st(self):
        logger.debug("Start to execute ============ test_construct_op_kernel_info_st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_path = test_root_dir + "/stub/files/ascend910.json"
        with open(json_path, "r") as file_in:
            op_builtin_info_dict = json.load(file_in)
        SubOpInfoStore().set_op_content(op_builtin_info_dict)
        op_type = "GNTrainingReduce"
        ret = SubOpInfoStore().construct_op_kernel_info(op_type)
        self.assertEqual(ret, True)
        op_type = "GIoU"
        ret = SubOpInfoStore().construct_op_kernel_info(op_type)
        self.assertEqual(ret, True)
        op_type = "FusedMulAdd"
        ret = SubOpInfoStore().construct_op_kernel_info(op_type)
        self.assertEqual(ret, True)
        op_type = "FresnelSin"
        ret = SubOpInfoStore().construct_op_kernel_info(op_type)
        self.assertEqual(ret, True)
        op_type = "Fills"
        ret = SubOpInfoStore().construct_op_kernel_info(op_type)
        self.assertEqual(ret, True)
        op_type = "Fills_1"
        ret = SubOpInfoStore().construct_op_kernel_info(op_type)
        self.assertEqual(ret, True)
        op_type = "Fills_2"
        ret = SubOpInfoStore().construct_op_kernel_info(op_type)
        self.assertEqual(ret, False)
        logger.debug("End to execute ============ test_construct_op_kernel_info_st ============")

    def test_get_extra_params_st(self):
        logger.debug("Start to execute ============ test_get_extra_params_st ============")
        extra_params = {"op_type": "matmul",
                   "kernel_name": "kernel_namexxx"}
        get_extra_params(extra_params)
        logger.debug("End to execute ============ test_get_extra_params_st ============")

    def test_set_l1_fusion_type_st(self):
        logger.debug("Start to execute ============ test_set_l1_fusion_type_st ============")
        op_node = {
            "output_desc": [{
                "L1_fusion_type": 123
            }],
            "name": "conv2d"
        }
        set_l1_fusion_type(True, op_node)
        set_l1_fusion_type(False, op_node)
        logger.debug("End to execute ============ test_set_l1_fusion_type_st ============")

    def test_compress_node_st(self):
        logger.debug("Start to execute ============ test_compress_node_st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_path = test_root_dir + "/stub/files/json_test_1.json"
        with open(json_path, "r") as file_in:
            json_data = json.load(file_in)
        op_list = json_data.get("op_list")
        for op_operator in op_list:
            if op_operator["type"] == "Conv2DCompress" or op_operator["type"] == "Const":
                result = compress_node(op_operator, op_list)
                self.assertEqual(result, None)
                op_list_1 = []
                op_node_1 = {}
                result = compress_node(op_node_1, op_list_1)
                self.assertEqual(result, None)

        logger.debug("End to execute ============ test_compress_node_st ============")

    def test_trans_matmulcompress_bias_shape_st(self):
        logger.debug("Start to execute ============ test_trans_matmulcompress_bias_shape_st ============")
        node = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [{
			"name": "Conv2D/weight",
			"format": "NC1HWC0",
			"dtype": "float32",
			"shape": [1, 1, 1, 1, 1, 1],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		}
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        data_node = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "NHWC",
        "shape": [1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        data_node_1 = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "NHWC",
        "shape": [1, 1, 1, 1, 1],
        "total_shape": [1, 1, 1, 1, 1],
        "valid_shape": [1, 1, 1, 1, 1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        data_node_2 = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "NCHW",
        "shape": [1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        data_node_3 = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "NCHW",
        "shape": [1, 1, 1, 1, 1, 1, 1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        data_node_4 = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "FRACTAL_NZ",
        "shape": [1, 1, 1, 1, 1, 1, 1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        data_node_5 = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "FRACTAL_NZ",
        "shape": [1, 1, 1, 1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        data_node_6 = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "NHWC",
        "shape": [1, 1, 1, 1, 1, 1],
        "total_shape": [1, 1, 1, 1, 1, 1],
        "valid_shape": [1, 1, 1, 1, 1, 1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        trans_matmulcompress_bias_shape(data_node, node)
        trans_matmul_bias_shape(data_node, node)
        trans_bninference_shape(data_node, node)
        trans_bninference_shape(data_node_2, node)
        trans_fully_connection(data_node_1, node)
        trans_batch_matmul_shape(data_node_3, node)
        trans_ascend_quant_shape(data_node_4, node)
        trans_arequant_s16(data_node_1, node)
        trans_conv3d_shape(data_node_5, node)
        trans_depthwise_conv2d(data_node_1, node)
        trans_depthwise_conv2d(data_node_6, node)
        logger.debug("End to execute ============ test_trans_matmulcompress_bias_shape_st ============")

    def test_trans_biasadd_shape_st(self):
        logger.debug("Start to execute ============ test_trans_biasadd_shape_st ============")
        node = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "impl_mode": "high_performance",
		"attrs": ["NCHW"],
		"input_desc": [{
			"name": "Conv2D/weight",
			"format": "NC1HWC0",
			"dtype": "float32",
			"shape": [1, 1, 1, 1, 1, 1],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		}
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        data_node = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "NHWC",
        "shape": [1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        trans_biasadd_shape(data_node, node)
        node["input_desc"][0]["format"] = "NDHWC"
        trans_biasadd_shape(data_node, node)
        node["input_desc"][0]["format"] = "NCDHW"
        trans_biasadd_shape(data_node, node)
        node["input_desc"][0]["format"] = "NDC1HWC0"
        trans_biasadd_shape(data_node, node)
        del node["input_desc"][0]["format"]
        trans_biasadd_shape(data_node, node)
        logger.debug("End to execute ============ test_trans_biasadd_shape_st ============")


    def test_trans_mul_shape_st(self):
        logger.debug("Start to execute ============ test_trans_mul_shape_st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_path = test_root_dir + "/stub/files/json_test_2.json"
        with open(json_path, "r") as file_in:
            json_data = json.load(file_in)
        op_list = json_data.get("op_list")
        node = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "impl_mode": "high_performance",
		"attrs": ["NCHW"],
		"input_desc": [{
			"name": "Conv2D/weight",
			"format": "NC1HWC0",
			"dtype": "float32",
			"shape": [1, 1, 1, 1, 1, 1],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		}
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        data_node = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "NHWC",
        "shape": [1, 1, 1, 2, 1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        trans_mul_shape(data_node, node, op_list)
        op_list_1 = []
        data_node_1 = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "NHWC",
        "shape": [1, 1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        trans_mul_shape(data_node_1, node, op_list_1)
        logger.debug("Start to execute ============ test_trans_mul_shape_st ============")

    def test_trans_elemwise_shape_st(self):
        logger.debug("Start to execute ============ test_trans_elemwise_shape_st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_path = test_root_dir + "/stub/files/json_test_2.json"
        with open(json_path, "r") as file_in:
            json_data = json.load(file_in)
        op_list = json_data.get("op_list")
        node = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "impl_mode": "high_performance",
		"attrs": ["NCHW"],
		"input_desc": [{
			"name": "Conv2D/weight",
			"format": "NC1HWC0",
			"dtype": "float32",
			"shape": [1, 1, 1, 1, 1, 1],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		}
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        data_node = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "NHWC",
        "shape": [1, 1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        trans_elemwise_shape(data_node, node, op_list)
        del op_list[0]["pattern"]
        trans_elemwise_shape(data_node, node, op_list)
        logger.debug("End to execute ============ test_trans_elemwise_shape_st ============")

    def test_trans_batch_matmul_shape_st(self):
        logger.debug("Start to execute ============ test_trans_batch_matmul_shape_st ============")
        node = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "impl_mode": "high_performance",
		"attrs": ["NCHW"],
		"input_desc": [{
			"name": "Conv2D/weight",
			"format": "NC1HWC0",
			"dtype": "float32",
			"shape": [1, 1, 1, 1, 1, 1],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		}
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        data_node = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "NCHW",
        "shape": [1, 1, 1, 1, 1, 1, 1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        ret = trans_batch_matmul_shape(data_node, node)
        self.assertEqual(ret, None)
        node["input_desc"][0]["name"] = "adcl"
        ret = trans_batch_matmul_shape(data_node, node)
        self.assertEqual(ret, None)
        node["input_desc"][1]["name"] = "wsshio"
        ret = trans_batch_matmul_shape(data_node, node)
        self.assertEqual(ret, None)
        logger.debug("End to execute ============ test_trans_batch_matmul_shape_st ============")

    def test_trans_shape_fullycompress_st(self):
        logger.debug("Start to execute ============ test_trans_shape_fullycompress_st ============")
        node = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "impl_mode": "high_performance",
		"attrs": ["NCHW"],
		"input_desc": [{
			"name": "Conv2D/weight",
			"format": "NC1HWC0",
			"dtype": "float32",
			"shape": [1, 1, 1, 1, 1, 1],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		},
        {
			"name": "Conv2D/weight",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
			"origin_shape_initialized": False
		}
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        data_node = {
        "name": "Conv2D/weight",
		"type": "Const",
		"id": 3,
        "format": "NCHW",
        "shape": [1, 1, 1, 1, 1],
        "impl_mode": "high_performance",
		"attrs": [{
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}],
		"input_desc": [
        ],
		"output_desc": [{
			"name": "",
			"format": "ND",
			"dtype": "float32",
			"shape": [],
			"origin_format": "ND",
		}]
    }
        trans_shape_fullycompress(data_node, node)
        node["input_desc"][0]["name"] = "sdddddv"
        trans_shape_fullycompress(data_node, node)
        logger.debug("End to execute ============ test_trans_shape_fullycompress_st ============")

    def test_get_fusion_pattern_st(self):
        logger.debug("Start to execute ============ test_get_fusion_pattern_st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_path = test_root_dir + "/stub/files/json_test_3.json"
        with open(json_path, "r") as file_in:
            json_data = json.load(file_in)
        op_list = json_data.get("op_list")
        ret = get_fusion_pattern(op_list)
        self.assertEqual(ret, "Conv2d_backprop_filter")
        op_list[1]["pattern"] = "Convolution"
        ret = get_fusion_pattern(op_list)
        self.assertEqual(ret, "Convolution")
        logger.debug("End to execute ============ test_get_fusion_pattern_st ============")

    def test_get_compute_op_func_st(self):
        logger.debug("Start to excute ============ test_get_compute_op_func_st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_path = test_root_dir + "/stub"
        op_type = "conv2d"
        soc_version = "Ascend910"
        ret = get_compute_op_func(op_type)
        self.assertEqual(ret.__name__, "conv2d_compute")
        op_type = "GNTrainingReduce"
        os.environ["ASCEND_OPP_PATH"] = json_path
        res = get_compute_op_func(op_type)
        self.assertEqual(res.__name__, "gn_training_reduce")
        logger.debug("End to excute ============ test_get_compute_op_func_st ============")

    def test_get_single_op_operator_st(self):
        logger.debug("Start to excute ============ test_get_single_op_operator_st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        ascend_path = test_root_dir + "/stub"
        os.environ["ASCEND_OPP_PATH"] = ascend_path
        op_type = "conv2d"
        ret = get_single_op_operator(op_type, "true", True)
        self.assertEqual(ret.__name__, "conv2d")
        op_type = "fresnel_cos"
        res = get_single_op_operator(op_type, "true", True)
        self.assertEqual(res.__name__, "fresnel_cos")
        op_type = "asdsddd"
        res = get_single_op_operator(op_type, "false", True)
        self.assertEqual(res, None)
        logger.debug("End to excute ============ test_get_single_op_operator_st ============")

    def test_get_custom_op_operator_st(self):
        logger.debug("Start to excute ============ test_get_single_op_operator_st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        custom_path = test_root_dir + "/stub/vendors/custom"
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = custom_path
        load_op_info_store("Ascend910")
        op_type = "conv2d"
        ret = get_single_op_operator(op_type, "true", True)
        self.assertEqual(ret.__name__, "conv2d")

        op_type = "fresnel_cos"
        res = get_single_op_operator(op_type, "true", True)
        self.assertEqual(res.__name__, "fresnel_cos")

        op_type = "FlashAttentionScore"
        res = get_single_op_operator(op_type, "true", True)
        #self.assertEqual(res.__name__, "FlashAttentionScore")
        op_type = "asdsddd"
        res = get_single_op_operator(op_type, "false", True)
        self.assertEqual(res, None)
        logger.debug("End to excute ============ test_get_single_op_operator_st ============")

    def test_get_vendor_and_custom_path_st(self):
        logger.debug("Start to excute ============ test_get_vendor_and_custom_path_st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        ascend_path = test_root_dir + "/stub"
        os.environ["ASCEND_OPP_PATH"] = ascend_path
        config_path = "{}/vendors/config.ini".format(ascend_path)
        with open(config_path, 'r', encoding='utf-8') as file:
            content = file.read()
        new_content = content.replace('custom,mdc', 'vendor')
        with open(config_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        load_op_info_store("Ascend910")
        with open(config_path, 'r', encoding='utf-8') as file:
            content = file.read()
        new_content = content.replace('vendor', 'custom,mdc')
        with open(config_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        logger.debug("End to excute ============ test_get_vendor_and_custom_path_st ============")

    def test_get_classify_info_01__st(self):
        logger.debug("Start to execute ============ test_get_classify_info_01__st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_path = test_root_dir + "/stub/files/cube_classify_fusion.json"
        with open(json_path, "r") as file_in:
            json_data = json.load(file_in)
        op_list = json_data.get("op_list")
        fusion_pattern = ""
        get_classify_info(fusion_pattern, "cube", op_list)
        logger.debug("End to execute ============ test_get_classify_info_01__st ============")

    def test_get_classify_info_02__st(self):
        logger.debug("Start to execute ============ test_get_classify_info_02__st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_path = test_root_dir + "/stub/files/reduce_classify_fusion.json"
        with open(json_path, "r") as file_in:
            json_data = json.load(file_in)
        op_list = json_data.get("op_list")
        fusion_pattern = "CommReduce"
        get_classify_info(fusion_pattern, "reduce", op_list)
        json_path_1 = test_root_dir + "/stub/files/reduce_classify_fusion_1.json"
        with open(json_path_1, "r") as file_in:
            json_data = json.load(file_in)
        op_list = json_data.get("op_list")
        fusion_pattern = "CommReduce"
        get_classify_info(fusion_pattern, "reduce", op_list)
        json_path_2 = test_root_dir + "/stub/files/reduce_classify_fusion_2.json"
        with open(json_path_2, "r") as file_in:
            json_data = json.load(file_in)
        op_list = json_data.get("op_list")
        fusion_pattern = "CommReduce"
        try:
            get_classify_info(fusion_pattern, "reduce", op_list)
        except RuntimeError as e:
            logger.info("RuntimeError")
        op_list[0]["output_desc"][0]["name"] = "placeholder"
        get_classify_info(fusion_pattern, "reduce", op_list)
        logger.debug("End to execute ============ test_get_classify_info_02__st ============")

    def test_replace_cube_tvm_shapes_st(self):
        logger.debug("Start to execute ============ test_replace_cube_tvm_shapes_st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_path = test_root_dir + "/stub/files/cube_classify_fusion.json"
        with open(json_path, "r") as file_in:
            json_data = json.load(file_in)
        op_list = json_data.get("op_list")
        fusion_pattern = ""
        vector_info = get_classify_info(fusion_pattern, "cube", op_list)
        inputs = []
        for op in op_list:
            if op["type"] == "Data":
                inputs.append(op)
            if op["type"] == "Conv2D":
                op["type"] = "TransData"
        replace_cube_tvm_shapes(inputs, vector_info, op_list, None)
        logger.debug("End to execute ============ test_replace_cube_tvm_shapes_st ============")

    def test_get_fusion_build_cfg_st(self):
        logger.debug("Start to execute ============ test_get_fusion_build_cfg_st ============")

        build_config = {
            "read_write_bank_conflict": 1,
            "InjectSync": {"sync_mode": 3},
        }
        import tbe.common.register as register
        register.case_switch = 0

        res = get_fusion_build_cfg()

        self.assertEqual(res, build_config)

        register.case_switch = 1
        res = get_fusion_build_cfg()
        self.assertEqual(res, build_config)

        logger.debug("End to execute ============ test_get_fusion_build_cfg_st ============")

    def test_inner_replace_cube_tvm_shapes_conv2d_transdata_st(self):
        logger.debug("Start to execute ============ test_inner_replace_cube_tvm_shapes_conv2d_transdata_st ============")
        dummy_transdata_input_name_vec = ["Conv2D/weight"]

        dummy_inputs = [
            {
                "name": "Conv2D/weight",
                "input_pattern": "cube",
                "format": "NC1HWC0",
                "dtype": "float16",
                "shape": [],
                "origin_format": "ND",
                "origin_shape_initialized": False,
                "placement": 0,
                "shape_range": []
            }, {
                "name": "Conv2D/weight",
                "format": "NCHW",
                "dtype": "float16",
                "shape": [],
                "origin_format": "ND",
                "origin_shape_initialized": False,
                "placement": 0,
                "shape_range": []
            }, {
                "name": "Conv2D/weight",
                "format": "ND",
                "dtype": "float32",
                "shape": [],
                "origin_format": "ND",
                "origin_shape_initialized": False
            }, {
                "name": "other_nodes",
                "format": "FORMAT_RESERVED",
                "dtype": "undefined",
                "shape": [],
                "origin_format": "ND",
                "origin_shape_initialized": False
            }
        ]
        cube_inputs = []
        vector_inputs = []

        _inner_replace_cube_tvm_shapes(dummy_transdata_input_name_vec, dummy_inputs, cube_inputs, vector_inputs)

        cube_inputs = []
        vector_inputs = []
        dummy_transdata_input_name_vec = ["Conv2D/weight", "Conv2D/weight"]
        _inner_replace_cube_tvm_shapes(dummy_transdata_input_name_vec, dummy_inputs, cube_inputs, vector_inputs)

        logger.debug("End to execute ============ test_inner_replace_cube_tvm_shapes_conv2d_transdata_st ============")

    def test_get_new_attrs_for_op_compile(self):
        logger.debug("Start to execute ============ test_get_new_attrs_for_op_compile ============")
        attrs = [{
			"name": "_input_name_key",
			"dtype": "list_str",
			"value": ["bias", "filter", "offset_w", "x"]
		}, {
			"name": "_input_name_value",
			"dtype": "list_int",
			"value": [2, 1, 3, 0]
		}, {
			"name": "_opt_input",
			"dtype": "list_str",
			"value": ["bias", "offset_w"]
		}, {
			"name": "_output_name_key",
			"dtype": "list_str",
			"value": ["y"]
		}, {
			"name": "_output_name_value",
			"dtype": "list_int",
			"value": [0]
		}, {
			"name": "data_format",
			"dtype": "str",
			"value": "NHWC"
		}, {
			"name": "dilations",
			"dtype": "list_int",
			"value": [1, 1, 1, 1]
		}, {
			"name": "groups",
			"dtype": "int",
			"value": 1
		}, {
			"name": "is_input_const",
			"dtype": "list_bool",
			"value": [False, True, True]
		}, {
			"name": "offset_x",
			"dtype": "int",
			"value": 1
		}, {
			"name": "pads",
			"dtype": "list_int",
			"value": [0, 0, 0, 0]
		}, {
			"name": "strides",
			"dtype": "list_int",
			"value": [1, 1, 1, 1]
		}, {
            "name": "var_attrs",
            "dtype": "list_str",
            "value": ["strides", "dilations", "groups", "data_format", "offset_x"]
        }
        ]
        def op_func(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                    groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d",
                    dsl_flag=True):
            pass
        op_node = {"attrs": attrs}
        res = get_new_attrs_for_op_compile(op_node, op_func, OpcCompileMode.SINGLE_OP_GRAPH_MODE)
        expect_res = [None, [0, 0, 0, 0], None, None, None, None]
        self.assertEqual(res, expect_res)
        logger.debug("End to execute ============ test_get_new_attrs_for_op_compile ============")

    def test_get_op_inputs_args_st(self):
        logger.debug("Start to execute ============ test_get_op_inputs_args ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_path = test_root_dir + "/stub/files/accumulate.json"
        with open(json_path, "r") as file_in:
            json_data = json.load(file_in)
        op_list = json_data.get("op_list")
        op_node = op_list[0]
        get_op_inputs_args(op_list, op_node)
        logger.debug("End to execute ============ test_get_op_inputs_args ============")

    def test_load_op_info(self):
        load_op_info_store("Ascend910B1")
        res = normalize_optional_impl_mode("high_performance, optional")
        self.assertEqual(res, "high_performance")

    def test_get_op_inputs_args(self):
        logger.debug("Start to execute ============ test_get_op_inputs_args ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        json_path = test_root_dir + "/stub/files/accumulate.json"
        with open(json_path, "r") as file_in:
            json_data = json.load(file_in)
        op_list = json_data.get("op_list")
        op_node = op_list[0]
        get_op_inputs_args(op_list, op_node)
        logger.debug("End to execute ============ test_get_op_inputs_args ============")

    def test_implmode_op_compilation_st_01(self):
        """test op_compilation"""
        logger.debug("Start to execute ============ test_implmode_op_compilation_st ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/sqrt.json"
        op_path = test_root_dir + "/stub/files/Sqrt.py"
        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "sqrt",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output",
            "debug_dir" : test_root_dir + "/debug_dir",
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "impl_mode" : "high_performance,optional",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level" : 1
        }
        op_compile = OpCompilation(opc_compile_args)
        ret = op_compile.op_compilation()
        self.assertEqual(ret, True)

        # delete file in output
        logger.debug("Testcase delete files in output dir")
        self.del_files(test_root_dir + "/output")
        logger.debug("End to execute ============ test_implmode_op_compilation_st ============")

    def test_implmode_op_compilation_st_02(self):
        """test op_compilation"""
        logger.debug("Start to execute ============ test_implmode_op_compilation_st_02 ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/MatMul.json"
        op_path = test_root_dir + "/stub/files/mat_mul.py"
        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output",
            "debug_dir" : test_root_dir + "/debug_dir",
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "impl_mode" : "high_performance,optional",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level" : 1
        }
        op_compile = OpCompilation(opc_compile_args)
        ret = op_compile.op_compilation()
        self.assertEqual(ret, True)

        # delete file in output
        logger.debug("Testcase delete files in output dir")
        self.del_files(test_root_dir + "/output")
        logger.debug("End to execute ============ test_implmode_op_compilation_st_02 ============")

    def test_implmode_op_compilation_st_03(self):
        """test op_compilation"""
        logger.debug("Start to execute ============ test_implmode_op_compilation_st_03 ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/sqrt.json"
        op_path = test_root_dir + "/stub/files/aaaaaaaaaaa.py"
        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "sqrt",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output",
            "debug_dir" : test_root_dir + "/debug_dir",
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "impl_mode" : "high_performance,optional",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level" : 1
        }
        op_compile = OpCompilation(opc_compile_args)
        ret = op_compile.op_compilation()
        self.assertEqual(ret, False)

        # delete file in output
        logger.debug("Testcase delete files in output dir")
        self.del_files(test_root_dir + "/output")
        logger.debug("End to execute ============ test_implmode_op_compilation_st_03 ============")

    def test_implmode_op_compilation_st_04(self):
        """test op_compilation"""
        logger.debug("Start to execute ============ test_implmode_op_compilation_st_04 ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        input_param_path = test_root_dir + "/stub/files/MatMul.json"
        op_path = test_root_dir + "/stub/files/impl/dynamic/mat_mul.py"
        opc_compile_args = {
            "op_path" : op_path,
            "input_param" : input_param_path,
            "main_func" : "mat_mul",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output",
            "debug_dir" : test_root_dir + "/debug_dir",
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "impl_mode" : "high_performance,optional",
            "op_compile_classify": "single_op_compile_config_file_mode",
            "op_debug_level" : 1
        }
        op_compile = OpCompilation(opc_compile_args)
        ret = op_compile.op_compilation()
        self.assertEqual(ret, False)
        # delete file in output
        logger.debug("Testcase delete files in output dir")
        self.del_files(test_root_dir + "/output")

        op_path1 = "mat_mul.py"
        opc_compile_args["op_path"] = op_path1
        op_compile1 = OpCompilation(opc_compile_args)
        ret = op_compile1.op_compilation()
        self.assertEqual(ret, False)
        # delete file in output
        logger.debug("Testcase delete files in output dir")
        self.del_files(test_root_dir + "/output")

        logger.debug("End to execute ============ test_implmode_op_compilation_st_04 ============")

    def test_copy_compile_res_files_to_output_st(self):
        """test op_compilation"""
        logger.debug("Start to execute ============ test_copy_compile_res_files_to_output_st ============")

        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))

        output_dir = str(test_root_dir) + "/stub/files/kernel_meta/"

        op_path = str(test_root_dir) + "/stub/files/matmul_add_1.json"

        opc_compile_args = {
            "debug_dir" : str(test_root_dir) + "/debug_dir",
            "bin_filename" : "matmul_add_1",
            "soc_version" : "Ascend910B",
            "output" : output_dir,
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "impl_mode" : "high_performance, optional",
        }
        l1_fusion_flag = "false"
        l2_fusion_flag = "false"
        l2_mode_flag = "0"
        post_compilation = PostCompilation(opc_compile_args, l1_fusion_flag, l2_fusion_flag, l2_mode_flag)

        ret = post_compilation.copy_compile_res_files_to_output(op_path)

        self.assertEqual(os.path.exists(output_dir+"matmul_add_1.json"), True)
        self.assertEqual(os.path.exists(output_dir+"matmul_add_1_mix_aic.json"), True)
        self.assertEqual(os.path.exists(output_dir+"matmul_add_1_mix_aic.txt"), True)
        self.assertEqual(os.path.exists(output_dir+"matmul_add_1_mix_aiv.json"), True)
        self.assertEqual(os.path.exists(output_dir+"matmul_add_1_mix_aiv.txt"), True)

        logger.debug("End to execute ============ test_copy_compile_res_files_to_output_st ============")

    def test_check_simplifiedkey_st(self):
        logger.debug("Start to execute ============ test_check_simplifiedkey_st ============")
        test_file_dir = os.path.abspath(os.path.dirname(__file__))
        test_root_dir =  os.path.abspath(os.path.join(test_file_dir, "../.."))
        test_debug_dir = test_root_dir + "/debug_dir/kernel_meta/testSupportInfo.json"
        opc_compile_args = {
            "debug_dir" : test_root_dir + "/debug_dir",
            "bin_filename" : "add12343",
            "soc_version" : "Ascend910A",
            "output" : test_root_dir + "/output",
            "h" : "not_exist",
            "log" : "not_exist",
            "core_type" : "VectorCore",
            "optional_input_mode" : "no_placeholder",
            "impl_mode" : "high_performance, optional",
            "simplified_key_mode" : 1,
        }
        fusion_impl_mode = dict()
        fusion_impl_mode["Conv2d"] = ("high_performance", True)
        l1_fusion_flag = "false"
        l2_fusion_flag = "false"
        l2_mode_flag = "0"
        single_post_compilation = SingleOpPostCompile(opc_compile_args, l1_fusion_flag, l2_fusion_flag, l2_mode_flag, fusion_impl_mode)
        op = {
            "op_type": "Add",
            "kernel_name": "add12343",
            "kernel_meta_path":test_debug_dir,
            "inputs": [
                {
                    "name": "x1",
                    "index": 0,
                    "dtype": "float16",
                    "dtypeForBinQuery": [
                        "float16",
                        "float32"
                    ],
                    "format": "FRACTAL_NZ",
                    "formatForBinQuery": [
                        "ND",
                        "NCHW"
                    ],
                    "shape": [
                        -2
                    ]
                },
                None
            ],
            "outputs": [
                {
                    "name": "x1",
                    "index": 0,
                    "dtype": "float16",
                    "format": "FRACTAL_NZ",
                    "shape": [
                        -2
                    ]
                }
            ],
            "attrs": [
                {
                    'name': 'adj_x1',
                    'dtype': 'bool',
                    'value': False
                },
                None
            ]
        }
        op_info = {"op_type": "Add", "op_func_attr": "add"}
        single_post_compilation.update_info_to_json_file(op, op_info, test_debug_dir)
        with open(test_debug_dir, 'r') as output_json:
            out_json = json.load(output_json)
            simplifiedKey = out_json["supportInfo"]["simplifiedKey"]
            self.assertEqual(simplifiedKey, ['Add/d=0,p=0/1,2/,/1,29/0', 'Add/d=0,p=0/0,0/,/1,29/0'])

        del op["inputs"][0]["dtypeForBinQuery"]
        single_post_compilation.update_info_to_json_file(op, op_info, test_debug_dir)
        with open(test_debug_dir, 'r') as output_json:
            out_json = json.load(output_json)
            simplifiedKey = out_json["supportInfo"]["simplifiedKey"]
            self.assertEqual(simplifiedKey, ['Add/d=0,p=0/1,2/,/1,29/0', 'Add/d=0,p=0/1,0/,/1,29/0'])

        del op["inputs"][0]["formatForBinQuery"]
        single_post_compilation.update_info_to_json_file(op, op_info, test_debug_dir)
        with open(test_debug_dir, 'r') as output_json:
            out_json = json.load(output_json)
            simplifiedKey = out_json["supportInfo"]["simplifiedKey"]
            self.assertEqual(simplifiedKey, ['Add/d=0,p=0/1,29/,/1,29/0'])

        logger.debug("End to execute ============ test_check_simplifiedkey_st ============")

class TestOpCompilation_WithDeterministic:
    class TempOutputDir:
        def __init__(self, bin_name='test'):
            self._output_temp = tempfile.TemporaryDirectory()
            self.output_dir = self._output_temp.name

            self.meta_dir = self.output_dir + '/kernel_meta_test/kernel_meta/'
            self.bin_name = bin_name
            self.output_bin = self.meta_dir + self.bin_name

            self.compile_args = {
                "output": self.output_dir,
                "debug_dir": self.output_dir
            }

    class DefaultArg:
        def __init__(self):
            self.temp_output = TestOpCompilation_WithDeterministic.TempOutputDir()
            self.compile_args = {
                "soc_version" : "Ascend910A",
                "op_compile_classify": "single_op_compile_config_file_mode",
                "is_dynamic": 'false',
                "core_type" : "AiCore",
                **self.temp_output.compile_args
            }

            self.op_json = {
                "type": 'mat_mul',
                "op_list": [{
                    "bin_filename": self.temp_output.bin_name,
                    "inputs": [],
                    "outputs": [],
                }]
            }

    class MockSingleOpCompile:
        def __init__(self, will_output):
            self._patch_SingleOpCompile = unittest.mock.patch("op_compilation.SingleOpCompile")

            self.patch_SingleOpCompile = self._patch_SingleOpCompile.start()
            self.patch_op_compile = unittest.mock.MagicMock(side_effect=self.create_op_bin)

            self.patch_SingleOpCompile.return_value = unittest.mock.MagicMock()
            self.patch_SingleOpCompile.return_value.op_compile = self.patch_op_compile

            self.will_output = will_output

        def stop(self):
            self._patch_SingleOpCompile.stop()

        def create_op_bin(self):
            # 模拟算子生成输出文件，从而使编译成功
            will_output = self.will_output.pop(0)
            assert(will_output is not None)

            will_output_meta_dir, will_output_bin_path, will_output_json_content = will_output
            will_output_json_content["binFileName"] = will_output_bin_path
            will_output_json_content["binFileSuffix"] = ".o"

            if not os.path.isdir(will_output_meta_dir):
                os.mkdir(will_output_meta_dir)
            with open(will_output_bin_path + '.json', 'w') as output_json:
                with open(will_output_bin_path + '.o', 'w') as output_obj:
                    json.dump(will_output_json_content, output_json)
                    output_obj.write('')
            return will_output_bin_path + '.json'

    def test_deterministic_all__when_op_support__will_rename_binfile__and_compile_nondeterministic(self):
        test_arg = self.DefaultArg()

        opc_compile_args = test_arg.compile_args
        opc_compile_args["deterministic"] = 'all'

        testMock = self.MockSingleOpCompile([
            (test_arg.temp_output.meta_dir, test_arg.temp_output.output_bin,
             {"deterministic": 'true'}),
            (test_arg.temp_output.meta_dir, test_arg.temp_output.output_bin,
             {"deterministic": 'false'})
        ])

        with unittest.mock.patch('op_compilation.get_single_op_operator'):
            test_op_compilation = OpCompilation(opc_compile_args)
            test_compile_result = test_op_compilation.single_op_compilation(test_arg.op_json)

        assert(test_compile_result is True)

        output_bin = test_arg.temp_output.output_dir + '/' + test_arg.temp_output.bin_name

        assert(Path(output_bin + '_deterministic.json').is_file())
        assert(Path(output_bin + '_deterministic.o').is_file())
        with open(output_bin + '_deterministic.json', 'r') as output_deterministic_json:
            output_json = json.load(output_deterministic_json)
            assert(output_json.get('deterministic') == 'true')
            assert(output_json.get('binFileName') == test_arg.temp_output.bin_name + '_deterministic')

        assert(Path(output_bin + '.json').is_file())
        assert(Path(output_bin + '.o').is_file())
        with open(output_bin + '.json', 'r') as output_nondeterministic_json:
            output_json = json.load(output_nondeterministic_json)
            assert(output_json.get('deterministic') == 'false')

        testMock.stop()


class Tensor:
    def __init__(self, dtype, op):
        self.dtype = dtype
        self.op = op


class Op:
    def __init__(self, name):
        self.name =  name

if __name__ == '__main__':
    unittest.main()
