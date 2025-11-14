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
"""
ascendc compile gen code
"""
import re
from functools import reduce
from .get_op_tiling import TilingInfo
from .ascendc_common_utility import CommonUtility, CompileInfo
from .ascendc_constants import TILING_KEY_MACRO, CORE_TYPE_CUBE, INPUT_OUTPUT_DTYPE_LEN
from .get_op_tiling import OpInfo
from .global_storage import global_var_storage


def add_time_stamp_codes(desc_id, space_len: int = 1):
    source = "#ifdef ASCENDC_TIME_STAMP_ON\n"
    source += "    " * space_len + \
        f"AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::{desc_id}));\n"
    source += "#endif\n"
    return source


def gen_init_dump_code(is_mix: bool, dump_size: int):
    source = ""
    source += "    #if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON\n"
    source += "    workspace += " + str(global_var_storage.get_variable("ascendc_required_dump_workspace_size")) + ";\n"
    source += "    #endif\n"
    if global_var_storage.get_variable("ascendc_enable_dump_workspace") is True or \
        (not CommonUtility.is_support_workspace_offset()):
        source += "    AscendC::SetSysWorkspaceForce(workspace);\n"
    else:
        source += "    // actually SetSysWorkspaceForce is empty function here\n"
        source += "    AscendC::SetSysWorkspaceForce(workspace);\n"
    source += "    #if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON\n"
    source += "    constexpr uint32_t ASCENDC_DUMP_SIZE = 123;\n"
    if global_var_storage.get_variable("ascendc_dump_assert_only") is True:
        if is_mix:
            source += "    AscendC::StoreArgsOfInitDump(true);\n"
        else:
            source += "    AscendC::StoreArgsOfInitDump(false);\n"
    else:
        if is_mix:
            source += "    AscendC::InitDump(true, ASCENDC_DUMP_SIZE);\n"
        else:
            source += "    AscendC::InitDump(false, ASCENDC_DUMP_SIZE);\n"
    if global_var_storage.get_variable("ascendc_recognize_simtvf") is True:
        source += "    AscendC::Simt::SetSimtDumpWorkspace(workspace);\n"
    source += add_time_stamp_codes('TIME_STAMP_WRAP_INIT_DUMP')
    source += "    #endif\n"
    return source


def gen_usr_origin_kernel_function_call(func_name: str, opinfo: OpInfo, tiling_info: TilingInfo,
                                         has_template: bool = False):
    # call usr kernel function
    if has_template:
        source = f"    {func_name}<TEMPLATE_PARAMS>("
    else:
        source = f"    {func_name}("
    for origin_input in opinfo.origin_inputs:
        if origin_input is not None:
            if isinstance(origin_input, (list, tuple)):
                if len(origin_input) == 0:
                    source += " nullptr, "
                else:
                    source += "{}, ".format(origin_input[0]["param_name"])
            else:
                source += "{}, ".format(origin_input["param_name"])
        else:
            source += " nullptr, "
    for output in opinfo.outputs:
        source += "{}, ".format(output["param_name"])

    if opinfo.output_shape_depend_on_compute is not None and len(opinfo.output_shape_depend_on_compute) > 0:
        source += "__ascendc_output_shape, "

    # static shape need pass nullptr
    if tiling_info.static_shape_flag:
        source += "usrWorkspace, nullptr);\n"
    else:
        source += "usrWorkspace, tiling);\n"
    return source


def gen_template_tiling_params(compile_info):
    source = "#define TEMPLATE_PARAMS -1\n"
    source += "#define TEMPLATE_PARAMS_LEN 0\n\n"
    if not compile_info.template_tiling_info:
        return source
    for template_tiling_key, template_tiling_info in compile_info.template_tiling_info.items():
        if not template_tiling_info or not template_tiling_info.get("paramArgs", []):
            continue
        template_tiling_info_str = ', '.join([str(i) for i in template_tiling_info.get("paramArgs", [])])
        source += f"#if {TILING_KEY_MACRO} == {template_tiling_key}UL\n"
        source += f"#undef TEMPLATE_PARAMS\n"
        source += f"#define TEMPLATE_PARAMS {template_tiling_info_str}\n"
        source += f"#undef TEMPLATE_PARAMS_LEN\n"
        source += f"#define TEMPLATE_PARAMS_LEN {len(template_tiling_info.get('paramArgs', []))}\n"
        source += "#endif\n\n"
    return source


def gen_global_isolation_macro(compile_info: CompileInfo, tiling_info: TilingInfo):
    tiling_key = compile_info.tiling_key_list[0]
    if tiling_info.static_shape_flag:
        tiling_key = tiling_info.tiling_key

    if CommonUtility.is_v220():
        macro_branch_statment = f"#if {TILING_KEY_MACRO} == {tiling_key}UL && defined(__DAV_C220_VEC__)\n"
        # judge operator is aic only
        if compile_info.no_set_kernel_type is False:
            kernel_type = compile_info.tiling_key_kernel_type[str(tiling_key)]
            if kernel_type.value in [1, 3, 5, 6, 7]:
                macro_branch_statment = f"#if {TILING_KEY_MACRO} == {tiling_key}UL && defined(__DAV_C220_CUBE__)\n"
        elif compile_info.code_channel == CORE_TYPE_CUBE:
            macro_branch_statment = f"#if {TILING_KEY_MACRO} == {tiling_key}UL && defined(__DAV_C220_CUBE__)\n"
    elif CommonUtility.is_v200():
        macro_branch_statment = f"#if {TILING_KEY_MACRO} == {tiling_key}UL && defined(__DAV_M200__)\n"
        if compile_info.no_set_kernel_type is False:
            kernel_type = compile_info.tiling_key_kernel_type[str(tiling_key)]
            if kernel_type.value in [9]:
                macro_branch_statment = f"#if {TILING_KEY_MACRO} == {tiling_key}UL && defined(__DAV_M200_VEC__)\n"
    else:
        macro_branch_statment = f"#if {TILING_KEY_MACRO} == {tiling_key}UL\n"
    return macro_branch_statment


def get_code_for_l2_cache(compile_info: CompileInfo, source, tiling_info: TilingInfo):
    source += gen_global_isolation_macro(compile_info, tiling_info)
    source += f"    __gm__ struct OpSystemRunCfg g_opSystemRunCfg = {{{0}}};\n"
    source += f"#else\n"
    source += f"    extern __gm__ struct OpSystemRunCfg g_opSystemRunCfg;\n"
    source += f"#endif\n\n"
    return source


def gen_dci_codes():
    source = ''
    source += '#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)\n'
    source += '    pipe_barrier(PIPE_ALL);\n'
    source += '    dsb(mem_dsb_t::DSB_ALL);\n'
    source += '    dci();\n'
    source += '#endif\n'
    return source


def skip_mc2_context_size(opinfo: OpInfo):
    content = ""
    if opinfo.mc2_ctx:
        for _ in opinfo.mc2_ctx:
            content += "    tmpTilingSizeForOOM += 8;\n"
    return content


def match_options(options, compile_options):
    result = []
    for option in options:
        match = re.search(f'{option}=(\w+)', ' '.join(compile_options))
        if match:
            result.append(match.group(1))
        else:
            result.append(None)
    return result


def add_dtype_fmt_option_single(x_n, is_ref: bool = False):
    options = []
    x_n_in_kernel = x_n + '_REF' if is_ref else x_n
    options.append("DORIG_DTYPE_{n}".format(n=x_n_in_kernel))
    return options


def get_dtype_fmt_options(opinfo: OpInfo):
    options = []
    unique_param_name_set = set()

    inputs_length = len(opinfo.inputs)
    for idx, x in enumerate(opinfo.inputs):
        if x is None:
            options += [None]
            continue
        unique_param_name_set.add(x['param_name'])
        if opinfo.param_type_list[idx] == "dynamic":
            tmp = x['param_name']
            res = tmp[:tmp.index('_in')].upper()
            options += add_dtype_fmt_option_single(res)
        else:
            options += [None]

    for idx, x in enumerate(opinfo.outputs):
        if x is None:
            options += [None]
            continue
        if opinfo.param_type_list[idx + inputs_length] == "dynamic":
            tmp = x['param_name']
            res = tmp[:tmp.index('_out')].upper()
            if x['param_name'] in unique_param_name_set:
                options += add_dtype_fmt_option_single(res, True)
            else:
                options += add_dtype_fmt_option_single(res)
        else:
            options += [None]
    return options


data_type_map = {
    'DT_INT4': (2 << 16) + 1,
    'DT_INT8': 1,
    'DT_UINT8': 1,
    'DT_FLOAT16': 2,
    'DT_BF16': 2,
    'DT_INT16': 2,
    'DT_UINT16': 2,
    'DT_FLOAT': 4,
    'DT_INT32': 4,
    'DT_UINT32': 4,
    'DT_INT64': 8,
    'DT_UINT64': 8
}


def get_value(key):
    if key in data_type_map:
        return data_type_map[key]
    else:
        return None


def update_tiling_size_for_oom(compile_info: CompileInfo, tiling_info: TilingInfo, dyn_input_shape_offset):
    content = ""
    # use user-defined tiling struct
    if len(compile_info.tiling_key_struct_map) > 0:
        for tiling_key in tiling_info.tiling_key_list:
            content += f"#if {TILING_KEY_MACRO} == {tiling_key}UL\n"
            content += f"    uint64_t tmpTilingSizeForOOM = sizeof({compile_info.tiling_key_struct_map[tiling_key]});\n"
            content += f"    tmpTilingSizeForOOM = (tmpTilingSizeForOOM + 7) / 8 * 8;\n"
            content += "#endif\n"
    elif global_var_storage.get_variable("ascendc_tiling_no_register"):
        for tiling_key in tiling_info.tiling_key_list:
            content += f"#if {TILING_KEY_MACRO} == {tiling_key}UL\n"
            content += f"    uint64_t tmpTilingSizeForOOM = g_custom_tiling_size_meta_{tiling_key};\n"
            content += f"    tmpTilingSizeForOOM = (tmpTilingSizeForOOM + 7) / 8 * 8;\n"
            content += "#endif\n"
    else:
        if len(tiling_info.tiling_key_data_size) == 0:
            content += "    uint64_t tmpTilingSizeForOOM = {};\n".format(int(dyn_input_shape_offset))
        else:
            for tiling_key in tiling_info.tiling_key_list:
                content += f"#if {TILING_KEY_MACRO} == {tiling_key}UL\n"
                if tiling_key not in tiling_info.tiling_key_data_size:
                    content += "    uint64_t tmpTilingSizeForOOM = {};\n".format(int(tiling_info.default_tiling_size))
                else:
                    content += "    uint64_t tmpTilingSizeForOOM = {};\n".\
                        format(int(tiling_info.tiling_key_data_size[tiling_key]))
                content += "#endif\n"
    return content


def set_workspace_param(opinfo: OpInfo, tiling_info: TilingInfo):
    # set workspace addr && workspace len
    source = ""
    if tiling_info.static_shape_flag:
        if opinfo.output_shape_depend_on_compute is not None and len(opinfo.output_shape_depend_on_compute) > 0:
            # each output needs 9 uint64 elements
            output_shape_len = (9 * 8 * len(opinfo.output_shape_depend_on_compute) + 32 - 1) // 32 * 32
            source += "    AscendC::OOMCheckAddrRange(__ascendc_output_shape, {});\n".format(output_shape_len)
        source += "#ifdef ASCENDC_DUMP\n"
        source += "    AscendC::OOMCheckAddrRange(workspace - " + str(
            global_var_storage.get_variable("ascendc_required_dump_workspace_size")) + \
                ", {});\n".format(tiling_info.static_workspace_size)
        source += "#else\n"
        source += "    AscendC::OOMCheckAddrRange(workspace, {});\n".format(tiling_info.static_workspace_size)
        source += "#endif\n"
    else:
        if opinfo.output_shape_depend_on_compute is not None and len(opinfo.output_shape_depend_on_compute) > 0:
            output_shape_len = "*((__gm__ uint64_t *)((__gm__ uint8_t *)tiling + tmpTilingSizeForOOM))"
            source += "    AscendC::OOMCheckAddrRange(__ascendc_output_shape, {});\n".format(output_shape_len)
            source += "    tmpTilingSizeForOOM += 8;\n"
        workspace_len = "*((__gm__ uint64_t *)((__gm__ uint8_t *)tiling + tmpTilingSizeForOOM))"
        source += "#ifdef ASCENDC_DUMP\n"
        source += "    AscendC::OOMCheckAddrRange(workspace - " + str(global_var_storage.get_variable(
            "ascendc_required_dump_workspace_size")) + ", {});\n".format(workspace_len)
        source += "#else\n"
        source += "    AscendC::OOMCheckAddrRange(workspace, {});\n".format(workspace_len)
        source += "#endif\n"
    source += "#endif\n"
    return source


def add_op_param_to_workspace(opinfo: OpInfo, tiling_info: TilingInfo, source: str, \
                              dump_size: int, compile_options: list, compile_info: CompileInfo):
    input_output_info = []
    for io_info in [opinfo.inputs, opinfo.outputs]:
        if list(io_info):
            input_output_info += io_info
    dyn_input_shape_offset = tiling_info.tiling_data_size
    dyn_input_shape_offset = (dyn_input_shape_offset + 8 - 1) // 8 * 8
    count = 0
    source += "#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1\n"
    source += "    AscendC::OOMInit();\n"

    options = get_dtype_fmt_options(opinfo)
    dtype_char = match_options(options, compile_options)
    dtype_int = list(map(get_value, dtype_char))

    source += update_tiling_size_for_oom(compile_info, tiling_info, dyn_input_shape_offset)
    source += skip_mc2_context_size(opinfo)

    for io_index, op_param in enumerate(input_output_info):
        if op_param is None:
            continue
        if opinfo.param_type_list[io_index] == "dynamic":
            if dtype_int[io_index]:
                source += "    AscendC::OOMCheckTensorListRange({}, {});\n".format(
                    op_param.get("param_name"), dtype_int[io_index])
        else:
            if tiling_info.static_shape_flag:
                input_shape_len = reduce(lambda x, y: x * y, op_param.get("shape")) * \
                    INPUT_OUTPUT_DTYPE_LEN.get(op_param.get("dtype"))
                input_shape_len = (input_shape_len + 32 - 1) // 32 * 32
            else:
                input_shape_len = "*((__gm__ uint64_t *)((__gm__ uint8_t *)tiling + tmpTilingSizeForOOM))"
            source += "    AscendC::OOMCheckAddrRange({}, {});\n".format(
                op_param.get("param_name"), input_shape_len)
        source += "    tmpTilingSizeForOOM += 8;\n"
        count = count + 1
    source += set_workspace_param(opinfo, tiling_info)
    count = count + 1
    if count > 128:
        raise Exception(f"input and output num exceed 128")
    return source