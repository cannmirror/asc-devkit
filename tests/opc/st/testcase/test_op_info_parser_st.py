#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
Opc test case
"""
import unittest
import tbe.common.utils.log as logger

from op_info_parser import (OpInfoParser, TensorData)


class TestOpcOpInfoPaserSt(unittest.TestCase):
    def set_up(self):
        pass

    def tear_down(self):
        pass

    def test_01_get_op_info_and_generate_kernel_name_st(self):
        """test get_op_info and generate_kernel_name"""
        logger.debug("Start to execute ============ test_01_get_op_info_and_generate_kernel_name_st ============")
        opc_compile_args_dict = dict()
        opc_compile_args_dict["core_type"] = "Acend910"
        opc_compile_args_dict["aicore_num"] = 1
        op_info_dict = dict()
        op_info_dict["op_type"] = "add"
        op = {
            "comment": "ND_float16 with attr = true",
            "inputs": [
                {
                    "index": 0,
					"dtype": "float16",
                    "format": "NC1HWC0",
                    "sub_format": 2,
                    "ori_format": "NCHW",
                    "shape": [
                    ],  # specify process: will be set [1]
                    "range": [
                        [
                            1,
                            -1  # specify process: will be set None
                        ],
                        [
                            1,
                            -1
                        ],
                        [
                            1,
                            -1
                        ],
                        [
                            1,
                            -1
                        ]
                    ],
                    "ori_shape": [
                        -1,
                        -1,
                        -1,
                        -1
                    ],
                    "const_value": [
                        0.008575439453125,
                        0.0122833251953125,
                        0.0099334716796875,
                        0.0164642333984375
                    ]
                },
                {
                    "index": 0,
                    "dtype": "float16",
                    "format": "NC1HWC0",
                    #"ori_format": "NCHW", specify process: no ori_format, will be set same as format
                    "shape": [
                        -1,
                        -1,
                        -1,
                        -1
                    ] # specify process:: no range, resolve json input wil has no range
                    # specify process: no ori_shape, will be set same as shape
                },
                {
                    "type": "null", #specify process: no valid tensor, will be set to None
                    "index": 0
                },
                None
            ],
            "outputs": [
                {
                    "index": 0,
                    "dtype": "float16",
                    "format": "NC1HWC0",
                    "ori_format": "NCHW",
                    "shape": [
                        -1,
                        -1,
                        -1,
                        -1
                    ],
                    "range": [
                        [
                            1,
                            -1
                        ],
                        [
                            1,
                            -1
                        ],
                        [
                            1,
                            -1
                        ],
                        [
                            1,
                            -1
                        ]
                    ],
                    "ori_shape": [
                        -1,
                        -1,
                        -1,
                        -1
                    ]
                }
            ],
            "attrs": [
                {
                    "name": "strides",
                    "dtype": "bool",
                    "value": "true"
                },
                {
                    "name": "pads",
                    "dtype": "list_int",
                    "value": None
                },
                {
                    "name": "groups",
                    "dtype": "float",
                    "value": 1.0
                },
                {
                    "name": "data_format",
                    "dtype": "string",
                    "value": ["NCHW","NCHW","NCHW"]
                },
                {
                    "name": "offset_x",
                    "dtype": "int",
                    "value": [1,2,3,4]
                },
                None
            ]
        }
        op_info_parser = OpInfoParser(op, op_info_dict, opc_compile_args_dict)

        op_info_parser.get_op_info("add")
        #check point：size
        self.assertEqual(len(op_info_dict.get("inputs")), 4)
        self.assertEqual(len(op_info_dict.get("outputs")), 1)
        self.assertEqual(len(op_info_dict.get("attrs")), 6)

        #check point：first inputs shape is [1]
        check = op_info_dict.get("inputs")[0].get("shape") == [1]
        self.assertEqual(check, True)
        #check point：first inputs range -1 be set None
        check = op_info_dict.get("inputs")[0].get("range") == [[1, 1]]
        self.assertEqual(check, True)
        #check point：second inputs ori_format be set same as format
        check = op_info_dict.get("inputs")[1].get("ori_format") == op_info_dict.get("inputs")[1].get("format")
        self.assertEqual(check, True)
        #check point：second inputs ori_shape be set same as shape
        check = op_info_dict.get("inputs")[1].get("ori_shape") == op_info_dict.get("inputs")[1].get("shape")
        self.assertEqual(check, True)
        #check point：second inputs has no range
        check = op_info_dict.get("inputs")[1].get("range") == [[1, None], [1, None], [1, None], [1, None]]
        self.assertEqual(check, True)
        #check point：third inputs is none
        check = op_info_dict.get("inputs")[2] == None
        self.assertEqual(check, True)

        kernel_name = op_info_parser.generate_kernel_name()
        self.assertEqual(kernel_name, "add_cad913795db5b7909271604914284ea6ab28f95aa5a7a9db2451127c1cc3d855")
        logger.debug("End to execute ============ test_01_get_op_info_and_generate_kernel_name_st ============")

    def test_02_generate_kernel_name_binfilename_st(self):
        """test generate_kernel_name using bin_filename"""
        logger.debug("Start to execute ============ test_02_generate_kernel_name_binfilename_st ============")
        op_info_dict = dict()
        op_info_dict["bin_filename"] = "add_033fc89a3368bcab301917125079cc0fa68362669c81eb3d14bdea5f1ae2e044"

        op_info_parser = OpInfoParser(None, op_info_dict, None)
        kernel_name = op_info_parser.generate_kernel_name()
        self.assertEqual(kernel_name, "add_033fc89a3368bcab301917125079cc0fa68362669c81eb3d14bdea5f1ae2e044")
        logger.debug("End to execute ============ test_02_generate_kernel_name_binfilename_st ============")

    def test_04_check_get_op_info_st(self):
        """test get_op_info"""
        logger.debug("Start to execute ============ test_04_check_get_op_info_st ============")
        opc_compile_args_dict = dict()
        opc_compile_args_dict["core_type"] = "Acend910"
        opc_compile_args_dict["aicore_num"] = 1
        op_info_dict = dict()
        op_info_dict["op_type"] = "add"
        op = {
            "comment": "ND_float16 with attr = true",
            "attrs": [
                {
                    "name": "strides",
                    "dtype": "bool",
                    "value": "true"
                },
                {
                    "name": "pads",
                    "dtype": "listInt",
                    "value": None
                },
                {
                    "name": "groups",
                    "dtype": "float",
                    "value": 1.0
                },
                {
                    "name": "data_format",
                    "dtype": "string",
                    "value": ["NCHW","NCHW","NCHW"]
                },
                {
                    "name": "offset_x",
                    "dtype": "int",
                    "value": [1,2,3,4]
                },
                None
            ]
        }
        op_info_parser = OpInfoParser(op, op_info_dict, opc_compile_args_dict)
        try:
            op_info_parser.get_op_info("add")
        except Exception:
            pass
        logger.debug("End to execute ============ test_04_check_get_op_info_st ============")

    def test_05_check_get_op_info_st(self):
        """test get_op_info"""
        logger.debug("Start to execute ============ test_05_check_get_op_info_st ============")
        opc_compile_args_dict = dict()
        opc_compile_args_dict["core_type"] = "Acend910"
        opc_compile_args_dict["aicore_num"] = 1
        op_info_dict = dict()
        op_info_dict["op_type"] = "add"
        op = {
            "comment": "ND_float16 with attr = true",
            "attrs": [
                {
                    "fake_name": "strides",
                    "dtype": "bool",
                    "value": "true"
                },
                {
                    "name": "pads",
                    "dtype": "listInt",
                    "value": None
                },
                None
            ]
        }
        op_info_parser = OpInfoParser(op, op_info_dict, opc_compile_args_dict)
        try:
            op_info_parser.get_op_info("add")
        except Exception:
            pass
        logger.debug("End to execute ============ test_05_check_get_op_info_st ============")

    def test_06_check_get_op_info_st(self):
        """test get_op_info"""
        logger.debug("Start to execute ============ test_06_check_get_op_info_st ============")
        opc_compile_args_dict = dict()
        opc_compile_args_dict["core_type"] = "Acend910"
        opc_compile_args_dict["aicore_num"] = 1
        op_info_dict = dict()
        op_info_dict["op_type"] = "add"
        op = {
            "comment": "ND_float16 with attr = true",
            "attrs": [
                {
                    "name": "strides",
                    "fake_dtype": "bool",
                    "value": "true"
                },
                {
                    "name": "pads",
                    "dtype": "listInt",
                    "value": None
                },
                None
            ]
        }
        op_info_parser = OpInfoParser(op, op_info_dict, opc_compile_args_dict)
        try:
            op_info_parser.get_op_info("add")
        except Exception:
            pass
        logger.debug("End to execute ============ test_06_check_get_op_info_st ============")

    def test_07_check_get_op_info_ut(self):
        """test get_op_info"""
        logger.debug("Start to execute ============ test_07_check_get_op_info_st ============")
        opc_compile_args_dict = dict()
        opc_compile_args_dict["core_type"] = "Acend910"
        opc_compile_args_dict["aicore_num"] = 1
        op_info_dict = dict()
        op_info_dict["op_type"] = "add"
        op = {
            "comment": "ND_float16 with attr = true",
            "attrs": [
                {
                    'name': 'var_attrs',
                    'dtype': 'list_string',
                    'value': ['p', 'a', 'c']
                }, {
                    'name': 'a',
                    'value_range': [
                        [1, 2]
                    ],
                    'range_mode': ['left_excluded'],
                    'dtype': 'int'
                }, {
                    'name': 'p',
                    'value_list': [1, 2, 3],
                    'dtype': 'int'
                }
            ]
        }
        op_info_parser = OpInfoParser(op, op_info_dict, opc_compile_args_dict)
        try:
            op_info_parser.get_op_info("add")
        except Exception:
            pass
        logger.debug("End to execute ============ test_07_check_get_op_info_st ============")


if __name__ == '__main__':
    unittest.main()
