#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import os
import sys
import shutil
import unittest
from unittest import mock


THIS_FILE_NAME = __file__
FILE_PATH = os.path.dirname(os.path.realpath(THIS_FILE_NAME))
TOP_PATH = os.path.join(FILE_PATH, "../../../")
FRAMEWORK_PATH = os.path.join(
    TOP_PATH, "tools/build/")
sys.path.insert(0, FRAMEWORK_PATH)

import asc_op_compile_base
from asc_op_compile_base.common.buildcfg import buildcfg


class TestBuildcfg(unittest.TestCase):
    def setUp(self):
        # operator before each testcase
        print(f"-------------------SetUp----------------")


    def tearDown(self):
        # operator after each testcase
        print(f"-------------------TearDown-------------")


    def test_default_build_cfg(self):
        self.assertEqual(buildcfg.get_default_build_config("dynamic_shape"), False)
        self.assertEqual(buildcfg.get_default_build_config("bool_storage_as_1bit"), True)
        self.assertEqual(buildcfg.get_default_build_config("enable_mask_counter_mode"), "default_normal")
        self.assertEqual(buildcfg.get_default_build_config()["dump_ir"], False)
        self.assertEqual(buildcfg.get_default_build_config("all")["tik_debug_context_id"], -1)
        self.assertEqual(buildcfg.get_default_build_config("tir.test"), None)


    def test_current_build_cfg(self):
        buildcfg.set_current_build_config("dynamic_shape", True)
        buildcfg.set_current_build_config("bool_storage_as_1bit", False)
        buildcfg.set_current_build_config("early_start_mode", "enable")
        buildcfg.set_current_build_config("tir.test", "test")
        self.assertEqual(buildcfg.get_current_build_config("dynamic_shape"), True)
        self.assertEqual(buildcfg.get_current_build_config("bool_storage_as_1bit"), False)
        self.assertEqual(buildcfg.get_current_build_config("early_start_mode"), "enable")
        self.assertEqual(buildcfg.get_current_build_config("tir.test"), None)
        buildcfg.set_current_build_config("tir.is_dynamic_shape", False)
        self.assertEqual(buildcfg.get_current_build_config("tir.is_dynamic_shape"), False)
        self.assertEqual(buildcfg.get_current_build_config("dynamic_shape"), False)
        self.assertEqual(buildcfg.get_current_build_config()["is_dynamic_shape"], False)


if __name__ == "__main__":
    unittest.main()