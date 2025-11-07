#!/usr/bin/python
# -*- coding: utf-8 -*-
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
import ctypes

def _initial_lib():
    lib_path = os.path.join(os.path.dirname(__file__), "libasc_platform.so")
    g_lib = ctypes.cdll.LoadLibrary(lib_path)
    g_lib.ASCInitSocSpec.argtypes = [
        ctypes.c_char_p,  # str1
        ctypes.c_char_p,  # str2
        ctypes.c_char_p,  # str3
        ctypes.c_char_p   # str4
    ]
    g_lib.ASCInitSocSpec.restype = ctypes.c_int

    g_lib.ASCTeUpdateVersion.argtypes = [
        ctypes.c_char_p,  # str1
        ctypes.c_char_p,  # str2
        ctypes.c_char_p,  # str3
        ctypes.c_char_p   # str4
    ]
    g_lib.ASCTeUpdateVersion.restype = ctypes.c_int

    g_lib.ASCSetSocSpec.argtypes = [
        ctypes.c_char_p,  # str1
    ]
    g_lib.ASCSetSocSpec.restype = ctypes.c_int

    g_lib.ASCGetSocSpec.argtypes = [
        ctypes.c_char_p,  # str1
        ctypes.c_char_p,  # buffer
        ctypes.c_int32,  # buffer size
    ]
    g_lib.ASCGetSocSpec.restype = None

    g_lib.CreateStrStrMap.argtypes = []
    g_lib.CreateStrStrMap.restype = ctypes.c_void_p

    g_lib.MapInsert.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,  # str1
        ctypes.c_char_p,  # str2
    ]
    g_lib.MapInsert.restype = None

    g_lib.MapDelete.argtypes = [
        ctypes.c_void_p
    ]
    g_lib.MapDelete.restype = None

    g_lib.ASCSetPlatformInfoRes.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    g_lib.ASCSetPlatformInfoRes.restype = ctypes.c_bool

    g_lib.ASCSetCoreNumByCoreType.argtypes = [
        ctypes.c_char_p
    ]
    g_lib.ASCSetCoreNumByCoreType.restype = ctypes.c_bool


def _init_soc_spec(soc_version, core_type, aicore_num=None, l1_fusion_flag=None):
    if g_lib is None:
        _initial_lib()

    if not aicore_num:
        aicore_num = ""

    if not l1_fusion_flag:
        l1_fusion_flag = ""

    b_soc_version = soc_version.encode('utf-8') if isinstance(soc_version, str) else soc_version
    b_core_type = core_type.encode('utf-8') if isinstance(core_type, str) else core_type
    b_aicore_num = aicore_num.encode('utf-8') if isinstance(aicore_num, str) else aicore_num
    b_l1_fusion_flag = l1_fusion_flag.encode('utf-8') if isinstance(l1_fusion_flag, str) else l1_fusion_flag

    res = g_lib.ASCInitSocSpec(b_soc_version, b_core_type, b_aicore_num, b_l1_fusion_flag)
    return "success" if res == 0 else "error"


def _te_update_version(soc_version, core_type, aicore_num, l1_fusion):
    if g_lib is None:
        _initial_lib()
    b_soc_version = soc_version.encode('utf-8') if isinstance(soc_version, str) else soc_version
    b_core_type = core_type.encode('utf-8') if isinstance(core_type, str) else core_type
    b_aicore_num = aicore_num.encode('utf-8') if isinstance(aicore_num, str) else aicore_num
    b_l1_fusion = l1_fusion.encode('utf-8') if isinstance(l1_fusion, str) else l1_fusion

    res = g_lib.ASCTeUpdateVersion(b_soc_version, b_core_type, b_aicore_num, b_l1_fusion)
    return "success" if res == 0 else "error"


def _get_soc_spec(spec):
    if g_lib is None:
        _initial_lib()
    buffer_size = 100
    buffer = ctypes.create_string_buffer(b'', buffer_size)

    b_str1 = spec.encode('utf-8')
    g_lib.ASCGetSocSpec(ctypes.c_char_p(b_str1), buffer, buffer_size)
    if buffer.value == b'':
        return ""

    res_str = ctypes.string_at(buffer).decode('utf-8')

    return res_str


def _set_soc_spec(spec):
    if g_lib is None:
        _initial_lib()
    b_str1 = spec.encode('utf-8')
    res = g_lib.ASCSetSocSpec(ctypes.c_char_p(b_str1))
    return "success" if res == 0 else "error"


def _set_platform_info_res(device_id, res):
    if g_lib is None:
        _initial_lib()
    m = g_lib.CreateStrStrMap()
    for key, value in res.items():
        g_lib.MapInsert(m, key.encode('utf-8'), value.encode('utf-8'))
    ret = g_lib.ASCSetPlatformInfoRes(device_id, m)
    g_lib.MapDelete(m)
    return ret


def _set_core_num_by_core_type(core_type):
    if g_lib is None:
        _initial_lib()
    return g_lib.ASCSetCoreNumByCoreType(core_type.encode('utf-8'))

g_lib = None