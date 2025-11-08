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
compile operator
"""

import os
import stat
import subprocess
import copy
import json
import hashlib
import re
import shutil
import sys
import struct
from tbe.tvm.contrib.ccec import CCECInfo
from tbe.tvm.runtime.cce_runtime import tvm_callback_cce_postproc
from tbe.common.buildcfg import get_current_build_config
from tbe.common.buildcfg.buildcfg_mapping import enable_vector_core
from tbe.common.platform.platform_info import get_soc_spec, set_soc_spec
from tbe.tvm.error_mgr import raise_tbe_python_err, TBE_DEFAULT_PYTHON_ERROR_CODE
from tbe.tvm import var
from tbe.common.context import get_context
from .get_op_tiling import TilingInfo, is_static_shape, OpInfo
from .template_tiling import extract_template_tiling_info, decode_tiling
from .log_utils import LogUtil, AscendCLogLevel, CompileStage, COMPILE_STAGE_MSG_INFO
from .global_storage import global_var_storage
from .ascendc_constants import InferChannelParamsFromIFile, InferChannelParams, KernelMetaType, \
    STR_TO_KERNEL_TYPE_V220, STR_TO_KERNEL_TYPE_V200, CompileOptionTuple, \
    CORE_TYPE_MIX, CORE_TYPE_CUBE, CORE_TYPE_VEC, TILING_KEY_MACRO, \
    ASCENDC_OOM, MIX_CORE_MACRO
from .ascendc_common_utility import CommonUtility, CompileInfo, \
    process_ascendc_api_version, gen_func_align_attribute
from .ascendc_compile_dfx import DFXParamType, DFXPointType, DFXArgInfo, DFXSectionGenerator
from .ascendc_compile_v220 import gen_compile_cmd_v220, get_v220_kernel_type_mix_flag, call_bisheng_v220, \
    get_ktype_section_variable, get_code_channel_v220_by_first_tiling_key, \
    set_dynamic_sub_func_names_of_super_kernel_with_kernel_type, gen_compile_cmd_for_meta_info
from .ascendc_compile_v200 import call_bisheng_v200_static, call_bisheng_v200_dynamic
from .ascendc_compile_gen_code import get_code_for_l2_cache, \
    gen_usr_origin_kernel_function_call, gen_template_tiling_params, add_time_stamp_codes, \
    gen_init_dump_code, add_op_param_to_workspace, gen_dci_codes
from .ascendc_compile_gen_json import _gen_mix_json_from_seperate_json, \
    _gen_mix_json_from_seperate_json_for_kernel_type, _dynamic_kernel_list_to_json, \
    _dynamic_regbase_kernel_list_to_json, _static_regbase_kernel_list_to_json, _gen_mix_sub_json, \
    _gen_static_json_for_no_mix_v200, _gen_non_mix_sub_json, _gen_static_json_for_mix_v200, \
    _gen_dynamic_json_for_v200, _generate_final_json
from .ascendc_compile_base import compile_multi_tilingkey, link_relocatable, fatbin_objs, \
    SingleTilingKeyCompileParams, get_actual_kernel_type, compile_pre_process, link_relocatable_meta_file
from .super_kernel_sub_op_compile import gen_sub_kernel_name, split_sub_kernel_objs, \
    gen_sub_super_kernel_compile_options, add_sub_super_kernel_info
from .super_kernel_utility import check_exist_instrinsic_when_super_kernel
from .super_kernel_constants import SuperKernelStreamFusionMode
from .super_kernel_option_parse import parse_super_kernel_options

DEFAULT_TILING_KEY = '0'
COMPILE_INFO_KEY = 'compileInfo'
GEN_PLACE_HOLDER_STR = 'gen_placeholder'
TILING_KEY_SEARCH_KEYWORD = 'Contents of section'            # used in new tiling to search tiling section lines

class KernelInfoInfer:
    """
    This class is used for get tiling key list and some kernel info
    i.g. code channel, kernel type for v200 and v220
    """
    @staticmethod
    def get_hard_sync_instr_from_i_file(content):
        """
        find whether used SyncAll instr in kernel func
        if so, return true
        otherwise, return false
        """
        pattern = re.compile(r"SyncAll\s*<\w+>\s*\(\s*\)\s*;")
        result = pattern.findall(content)
        if len(result) != 0:
            return True
        pattern = r'SyncAll\s*\(\s*\)\s*\;'
        match = re.findall(pattern, content)
        if len(match) > 2:
            return True
        return False


    @staticmethod
    def get_sync_task_start_end_instr_from_i_file(content):
        """
        find whether used SetNextTaskStart WaitPreTaskEnd instr in kernel func
        if so, return true
        otherwise, return false
        """
        set_pattern = r'SetNextTaskStart\s*\(\s*\)\s*\;'
        wait_pattern = r'WaitPreTaskEnd\s*\(\s*\)\s*\;'
        set_match = re.findall(set_pattern, content)
        wait_match = re.findall(wait_pattern, content)
        return len(set_match) > 2, len(wait_match) > 2


    @staticmethod
    def get_enable_deterministic_var_from_i_file(content):
        """
        find whether enable_deterministic in kernel func
        if so, return true
        otherwise, return false
        """
        pattern = r"__enable_feature_for_compile_deterministic\s*=\s*(-?\d+)\s*;"
        match = re.search(pattern, content)
        if match and match.group(1) == "1":
            return True
        return False


    @staticmethod
    def find_kernel_type(s):
        match = re.search(r"__enable_feature_for_compile_default\s*=\s*([0-9a-zA-Z_]{1,})\s*;", s)
        if match:
            return None, match.group(1)
        else:
            match = re.search(r"__enable_feature_for_compile_(-?\d+)([a-zA-Z]*)\s*=\s*([0-9a-zA-Z_]{1,})\s*;", s)
            if match:
                return match.group(1), match.group(3)
            return None, None

    @staticmethod
    def find_tilingkey(s):
        if re.search(r"g_tilingKey == \(", s):
            matches = re.findall(r"g_tilingKey == \((-?\d+)", s)
            if matches:
                return matches
            else:
                matches = re.findall(r"g_tilingKey == \((.*?)\)", s)
                if matches:
                    CommonUtility.print_compile_log("", f"Var: {matches[0]} in TILING_KEY_IS({matches[0]}) can not be \
processed as numeric variables in the precompilation phase. please use numeric constants, macros or const variables.", \
AscendCLogLevel.LOG_ERROR)
                return None
        return None

    @staticmethod
    def find_tiling_struct_and_expression(s):
        # find tiling keywords used in REGISTER_TILING_DEFAULT and REGISTER_TILING_FOR_TILINGKEY from input string
        # If find __enable_custom_tiling, return its tiling struct and expression
        if "__enable_custom_tiling" not in s:
            return None, None
        match = re.search(r"__enable_custom_tiling\s*([0-9a-zA-Z_:<>]{1,})\s*=\s*default;", s)
        if match:
            return match.group(1), None
        else:
            match = re.search(r"__enable_custom_tiling\s*([0-9a-zA-Z_:<>]{1,})\s*=\s*\"(.*)\";", s)
            if match:
                return match.group(1), match.group(2)
            else:
                raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE,
                    ("tiling struct match expression is wrong. lines: " + s))
                return None, None


    @staticmethod
    def find_tiling_struct_no_register_flag(s: str) -> bool:
        return "__enable_no_register_custom_tiling" in s
    

    @staticmethod
    def get_dump_info_from_i_file(content):
        dump_info = {"dump_type" : "", "dump_size" : 1048576}
        actual_dump_size = 1048576
        if CommonUtility.is_c310() or CommonUtility.is_310r6():
            actual_dump_size *= 108
        else:
            actual_dump_size *= 75

        match_printf = re.search(r"__enable_feature_for_compile_printf = 1", content)
        match_assert = re.search(r"__enable_feature_for_compile_assert = 1;", content)
        # recognizes whether this is simt_vf, and enable once exists simt_vf + printf
        match_simtvf = bool(re.search(r"cce_simt_entry", content) and match_printf)
        global_var_storage.set_variable("ascendc_recognize_simtvf", match_simtvf)
        if match_printf and match_assert:
            dump_info["dump_type"] = "printf,assert"
        elif match_printf:
            dump_info["dump_type"] = "printf"
        elif match_assert:
            dump_info["dump_type"] = "assert"

        if global_var_storage.get_variable("ascendc_time_stamp_compile_options"):
            if dump_info["dump_type"] != "":
                dump_info["dump_type"] = dump_info["dump_type"] + ",timestamp"
            else:
                dump_info["dump_type"] = "timestamp"
        else:
            match = re.search(r"__enable_feature_for_compile_printfBufSize = \s*([0-9]{1,})", content)
            if match:
                dump_info["dump_size"] = int(match.group(1))
            else:
                match = re.search(r"__enable_feature_for_compile_assertBufSize = \s*([0-9]{1,})", content)
                if match:
                    dump_info["dump_size"] = int(match.group(1))

        if CommonUtility.is_c310() or CommonUtility.is_310r6():
            actual_dump_size = 108 * dump_info["dump_size"]
        else:
            actual_dump_size = 75 * dump_info["dump_size"]
        simt_in_c310 = match_simtvf and (CommonUtility.is_c310() or CommonUtility.is_310r6())
        if dump_info["dump_type"] != "" and simt_in_c310:
            actual_dump_size = 1048576 * 108 + 72 * 2048 * 2048 # david 72 vec + 36 cube + simt
            dump_info["dump_size"] = 1048576 # reserved for ONE_CORE_DUMP_SIZE

        global_var_storage.set_variable("ascendc_required_dump_workspace_size", actual_dump_size)

        return dump_info

    @staticmethod
    def get_kernel_type_enum(kernel_type, compile_log_path):
        if CommonUtility.is_v220() or CommonUtility.is_c310() or CommonUtility.is_310r6():
            if kernel_type in STR_TO_KERNEL_TYPE_V220.keys():
                return STR_TO_KERNEL_TYPE_V220[kernel_type]
            else:
                if kernel_type not in STR_TO_KERNEL_TYPE_V200.keys():
                    CommonUtility.print_compile_log("", \
                        "current kernel type: {} is not support in current core version".\
                        format(kernel_type), AscendCLogLevel.LOG_WARNING)
                return None
        elif CommonUtility.is_v200():
            if kernel_type in STR_TO_KERNEL_TYPE_V200.keys():
                return STR_TO_KERNEL_TYPE_V200[kernel_type]
            else:
                if kernel_type not in STR_TO_KERNEL_TYPE_V220.keys():
                    CommonUtility.print_compile_log("", \
                        "current kernel type: {} is not support in current core version".\
                        format(kernel_type), AscendCLogLevel.LOG_WARNING)
                return None
        else:
            raise Exception("current kernel type: {} is not support in current core version".format(kernel_type))
        return None

    @staticmethod
    def gen_tiling_struct_macro_src_file(tiling_key_list, tiling_struct_expr_map, src_file):
        file_contents = ""
        for key, value in tiling_struct_expr_map.items():
            for expression in value:
                for tiling_key in tiling_key_list:
                    new_expression = expression.replace(TILING_KEY_MACRO, TILING_KEY_MACRO + "_" + tiling_key)
                    file_contents += f"#if defined({TILING_KEY_MACRO}_{tiling_key}) && {new_expression}\n"
                    file_contents += f"    auto __ascendc_custom_tiling_struct = ({tiling_key}, {key});\n"
                    file_contents += "#endif\n"
        try:
            with open(src_file, 'w') as f:
                f.writelines(file_contents)
        except Exception as err:
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE,\
                                ("write tiling struct tmp file failed, reason is :", err))

    @staticmethod
    def get_tiling_key_corresponding_struct(tiling_key_list, default_tiling_struct,\
        src_tiling_file, dst_tiling_file, compile_log_path):
        tiling_key_struct_map = {}
        tiling_compile_cmd = [global_var_storage.get_variable("ascendc_compiler_path"), \
                              "-c", '-O3', '-std=c++17', '-E', src_tiling_file, '-o', dst_tiling_file]
        for tiling_key in tiling_key_list:
            tiling_compile_cmd.append(f"-D{TILING_KEY_MACRO}_{tiling_key}={tiling_key}UL")
        CommonUtility.run_cmd_inner(tiling_compile_cmd, CompileStage.PRECOMPILE, compile_log_path)
        match_tiling_struct = ""
        try:
            with open(dst_tiling_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("#"):
                        continue
                    match = \
re.search(r"auto __ascendc_custom_tiling_struct\s*=\s*\((-?\d+),\s([0-9a-zA-Z_:]{1,})\);", line)
                    if match:
                        if match.group(1) in tiling_key_struct_map and \
                            tiling_key_struct_map[match.group(1)] != match.group(2):
                            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, \
                                                 (f"tiling key {match.group(1)} should have 1 unique corresponding tiling struct, but \
found following structs::{tiling_key_struct_map[match.group(1)]}, {match.group(2)}"))
                        else:
                            tiling_key_struct_map[match.group(1)] = match.group(2)
        except Exception as err:
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, \
                ("read tiling struct dump file failed, reason is :", err))
        for tiling_key in tiling_key_list:
            if tiling_key not in tiling_key_struct_map:
                tiling_key_struct_map[tiling_key] = default_tiling_struct
        return tiling_key_struct_map

    @staticmethod
    def check_func_name_exist(pattern: str, text: str):
        match = re.search(r"\b{}\b\s*\(".format(pattern), text)
        if match:
            return True
        else:
            return False


    @staticmethod
    def dfx_for_func_name(cce_file: str, origin_func_name: str, func_name_exist: bool):
        if not func_name_exist:
            cce_file = cce_file.split("/")[-1]
            CommonUtility.print_compile_log("", f"kernel entry `{origin_func_name}' not implement in `{cce_file}', \
please check whether the function name is correct in the kernel file.",
                AscendCLogLevel.LOG_ERROR)
            raise Exception("An error occurred during stage of infer compile info")


    @staticmethod
    def search_any_in_line(line, keywords):
        pattern = re.compile(r'\b(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b')
        matches = pattern.findall(line)
        return matches


    @staticmethod
    def get_kernel_meta_type(value):
        for member in KernelMetaType:
            if member.value == value:
                return member
        return None

    @staticmethod
    def _gen_tiling_key_struct_map(default_tiling_struct, declare_param_str, select_param_str, decode_tiling_result, \
                                   dst_i_file, tiling_key_list, tiling_struct_expr_map, compile_log_path):
        tiling_key_struct_map = {}
        if default_tiling_struct != "":
            if declare_param_str and select_param_str:
                for tiling_key, information in decode_tiling_result.items():
                    tiling_key_struct_map[str(tiling_key)] = information.get('tilingStruct', default_tiling_struct)
            else:
                src_tiling_file = dst_i_file[:-2] + "_tiling_key_tiling_struct.cpp"
                dis_tiling_i_file = dst_i_file[:-2] + "_tiling_key_tiling_struct" + dst_i_file[-2:]
                KernelInfoInfer.gen_tiling_struct_macro_src_file(tiling_key_list,\
                                                                tiling_struct_expr_map, src_tiling_file)
                tiling_key_struct_map = KernelInfoInfer.get_tiling_key_corresponding_struct(tiling_key_list, \
                    default_tiling_struct, src_tiling_file, dis_tiling_i_file, compile_log_path)
        else:
            if len(tiling_struct_expr_map) != 0:
                raise Exception(f'if use user-defined tiling structure, must provide default tiling struct, use macro \
REGISTER_TILING_DEFAULT')
            for _, information in decode_tiling_result.items():
                if 'tilingStruct' in information:
                    raise Exception(f'if use user-defined tiling structure, must provide default tiling struct,\
 use macro REGISTER_TILING_DEFAULT')
        return tiling_key_struct_map

    @staticmethod
    def infer_info_from_ifile(dst_i_file: str, compile_log_path, cce_file: str, origin_func_name: str):
        tiling_key_list = []
        declare_param_str = ""
        select_param_str = ""
        decode_tiling_result = {}
        code_channel:int = -1
        no_kfc_server_flag = False
        find_kfc_server = False
        default_tiling_struct = ""
        tiling_struct_expr_map = {}
        if not (CommonUtility.is_v220() or CommonUtility.is_c310() or CommonUtility.is_310r6()):
            code_channel = CORE_TYPE_MIX
        if global_var_storage.get_variable("ascendc_enable_super_kernel") is True:
            check_exist_instrinsic_when_super_kernel(dst_i_file)
        try:
            with open(dst_i_file, 'r') as fd:
                content = fd.read()
                fd.close()
        except Exception as err:
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, ("read dst_i_file failed, reason is:", err))
        hard_sync = KernelInfoInfer.get_hard_sync_instr_from_i_file(content)
        global_var_storage.set_variable("ascendc_op_with_syncall", hard_sync)
        if global_var_storage.get_variable("ascendc_enable_super_kernel") is True:
            set_task_bar, wait_task_bar = KernelInfoInfer.get_sync_task_start_end_instr_from_i_file(content)
        else:
            set_task_bar = False
            wait_task_bar = False
        enable_deterministic = KernelInfoInfer.get_enable_deterministic_var_from_i_file(content)
        tiling_key_kernel_type = {}
        tiling_key_deterministic = {}
        default_kernel_type = KernelMetaType.KERNEL_TYPE_MAX
        dump_info = KernelInfoInfer.get_dump_info_from_i_file(content)
        func_name_exist = False
        expect_tilingkey_set = set()
        cur_deterministic_flag = get_current_build_config("enable_deterministic_mode") == 1
        need_find_kernel_type = (not CommonUtility.is_m510())
        tiling_no_register_flag = False
        try:
            with open(dst_i_file, 'r') as fd:
                lines = fd.readlines()
                fd.close()
        except Exception as err:
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, ("read dst_i_file failed, reason is:", err))
        keywords = ['ccec_compiler', 'tikcpp/tikcfw', 'gnu/bits', 'include/c++', \
            'include/kernel_tiling/kernel_tiling.h']
        is_op_block: bool = True
        for line in lines:
            if line.startswith("#"):
                found_built_in = KernelInfoInfer.search_any_in_line(line, keywords)
                is_op_block = not found_built_in
                continue
            if not is_op_block:
                continue
            func_name_exist = (func_name_exist or \
                KernelInfoInfer.check_func_name_exist(origin_func_name, line))
            if declare_param_str == "" and '@@ASCENDC_TPL_ARGS_DECL' in line:
                declare_param_str = line
            if select_param_str == "" and '@@ASCENDC_TPL_LISTS' in line:
                select_param_str = line
            if (not find_kfc_server) and 'AscendC::KfcServer' in line:
                code_channel = CORE_TYPE_MIX
                find_kfc_server = True
            # process register tiling strcut and expression
            tiling_struct, tiling_expression = KernelInfoInfer.find_tiling_struct_and_expression(line)
            if tiling_struct is not None:
                if tiling_expression is None:
                    if default_tiling_struct != "" and default_tiling_struct != tiling_struct:
                        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE,
                            ("Only one default tiling structure can be configured."))
                    else:
                        default_tiling_struct = tiling_struct
                else:
                    if tiling_struct in tiling_struct_expr_map:
                        tiling_struct_expr_map[tiling_struct].add("(" + tiling_expression + ")")
                    else:
                        tiling_struct_expr_map[tiling_struct] = set([str("(" + tiling_expression + ")")])
            tiling_key, kernel_type = KernelInfoInfer.find_kernel_type(line)
            if need_find_kernel_type:
                if tiling_key is None and kernel_type is not None:
                    cur_kernel_type = KernelInfoInfer.get_kernel_type_enum(kernel_type, compile_log_path)
                    if cur_kernel_type is not None:
                        default_kernel_type = cur_kernel_type
                if tiling_key is not None and kernel_type is not None:
                    cur_kernel_type = KernelInfoInfer.get_kernel_type_enum(kernel_type, compile_log_path)
                    if cur_kernel_type is not None:
                        tiling_key_kernel_type[str(int(tiling_key))] = cur_kernel_type
            tiling_no_register_flag |= KernelInfoInfer.find_tiling_struct_no_register_flag(line)
            numbers = KernelInfoInfer.find_tilingkey(line)
            if numbers is None:
                continue
            for number in numbers:
                if str(int(number)) not in tiling_key_list:
                    tiling_key_list.append(str(int(number)))
        if declare_param_str and select_param_str:
            # TPL
            extract_template_tiling_info(declare_param_str, select_param_str)
            decode_tiling_result = decode_tiling()
            tiling_key_list = [str(k) for k in decode_tiling_result.keys()]
            tpl_set_kernel_type_cnt = 0
            for k in decode_tiling_result.keys():
                internal_dict = decode_tiling_result[k]
                if "kernelType" in internal_dict:
                    tpl_set_kernel_type_cnt += 1
                    tpl_kernel_type = KernelInfoInfer.get_kernel_meta_type(internal_dict['kernelType'])
                    if tpl_kernel_type is not None:
                        tiling_key_kernel_type[str(k)] = tpl_kernel_type
                    else:
                        CommonUtility.print_compile_log("", \
                            "get_kernel_meta_type return tpl_kernel_type is None, kernel_type value is {}".\
                            format(internal_dict['kernelType']), AscendCLogLevel.LOG_ERROR)
            if tpl_set_kernel_type_cnt != 0 and tpl_set_kernel_type_cnt != len(tiling_key_list):
                CommonUtility.print_compile_log("", 
                    "All ASCENDC_TPL_ARGS_SEL must set ASCENDC_TPL_KERNEL_TYPE_SEL simultaneously!", \
                    AscendCLogLevel.LOG_ERROR)

            for k, v in decode_tiling_result.items():
                if "deterministic" in v:
                    tiling_key_deterministic[str(k)] = v["deterministic"]
                    deterministic_flag = True if v["deterministic"].lower() == "true" else False
                    if deterministic_flag == cur_deterministic_flag:
                        expect_tilingkey_set.add(str(k))

        KernelInfoInfer.dfx_for_func_name(cce_file, origin_func_name, func_name_exist)

        if tiling_no_register_flag:
            CommonUtility.dump_log("Found Using REGISTER_NONE_TILING", compile_log_path)
            global_var_storage.set_variable("ascendc_tiling_no_register", True)

        if len(tiling_key_list) == 0:
            tiling_key_list = [DEFAULT_TILING_KEY]
        if not find_kfc_server:
            no_kfc_server_flag = True

        no_set_kernel_type = False
        if default_kernel_type == KernelMetaType.KERNEL_TYPE_MAX and not tiling_key_kernel_type:
            # TPL ORIGIN
            if get_current_build_config(enable_vector_core):
                CommonUtility.dump_log("Information Library Configuration Takes Effect", compile_log_path)
                for tiling_key in tiling_key_list:
                    tiling_key_kernel_type[tiling_key] = KernelMetaType.KERNEL_TYPE_MIX_VECTOR_CORE
            else:
                no_set_kernel_type = True
        else:
            if len(tiling_key_kernel_type) > 0 and len(tiling_key_kernel_type) != len(tiling_key_list) \
                and default_kernel_type == KernelMetaType.KERNEL_TYPE_MAX:
                raise Exception(f'must provide default kernel type')
            for tiling_key in tiling_key_list:
                if tiling_key not in tiling_key_kernel_type:
                    tiling_key_kernel_type[tiling_key] = default_kernel_type
            if get_current_build_config(enable_vector_core):
                CommonUtility.dump_log(\
                    "Information Library Configuration Does Not Take Effect After the Macro Is Enabled",\
                    compile_log_path, "[WARNING] : ")
        if not global_var_storage.get_variable("ascendc_compile_debug_config"):
            CommonUtility.remove_temp_file(dst_i_file)

        tiling_key_struct_map = KernelInfoInfer._gen_tiling_key_struct_map(default_tiling_struct, declare_param_str, \
                                                                           select_param_str, decode_tiling_result, \
                                                                           dst_i_file, tiling_key_list, \
                                                                           tiling_struct_expr_map, compile_log_path)

        if len(expect_tilingkey_set) > 0 and len(decode_tiling_result) > 0:
            tiling_key_list = [x for x in tiling_key_list if x in expect_tilingkey_set]
            decode_tiling_result = {k: v for k, v in decode_tiling_result.items() if str(k) in expect_tilingkey_set}

        return InferChannelParamsFromIFile(tiling_key_list, code_channel, hard_sync, no_kfc_server_flag, \
                                           enable_deterministic, tiling_key_kernel_type, no_set_kernel_type,\
                                           default_kernel_type, dump_info, decode_tiling_result,
                                           default_tiling_struct, tiling_struct_expr_map, tiling_key_struct_map,\
                                           set_task_bar, wait_task_bar, tiling_key_deterministic)

    @staticmethod
    def get_tiling_key_list_and_simple_infer_code_channel(cce_file: str, dst_i_file: str, \
        compile_option_tuple: CompileOptionTuple, compile_log_path, origin_func_name):
        """
        get tiling key list and simple infer code channel
        :param cce_file:
        :return:InferedInfo
        """
        compile_option_tuple_pre = copy.deepcopy(compile_option_tuple)
        compile_option_tuple_pre.compile_options = compile_option_tuple_pre.compile_options\
            + ['-E'] + ['-D__CHECK_FEATURE_AT_PRECOMPILE'] + ['-includestdio.h']
        compile_option_tuple_pre.compile_options = compile_option_tuple_pre.compile_options + ['-DASCENDC_TPL_PRE']
        chip_version = CommonUtility.get_chip_version()
        # generate .i file
        if CommonUtility.is_v220() or CommonUtility.is_c310() or CommonUtility.is_310r6():
            arch = f"dav-{chip_version}-cube"
            dis_i_file_cube = dst_i_file[:-2] + "_cube" + dst_i_file[-2:]
            pre_compile_cmd = gen_compile_cmd_v220(cce_file, dis_i_file_cube, \
                                    compile_option_tuple_pre, arch, '', False)
            CommonUtility.run_cmd_inner(pre_compile_cmd, CompileStage.PRECOMPILE, compile_log_path)
            arch = f"dav-{chip_version}-vec"
            dis_i_file_vec = dst_i_file[:-2] + "_vec" + dst_i_file[-2:]
            pre_compile_cmd = gen_compile_cmd_v220(cce_file, dis_i_file_vec, \
                                    compile_option_tuple_pre, arch, '', False)
            CommonUtility.run_cmd_inner(pre_compile_cmd, CompileStage.PRECOMPILE, compile_log_path)
            with open(dis_i_file_cube, 'r') as f_cube, open(dis_i_file_vec, 'r') as f_vec:
                cube_content = f_cube.read()
                vec_content = f_vec.read()
            # merge sub core .i file in dst_i_file
            with open(dst_i_file, 'w') as f:
                f.write(cube_content + vec_content)
            # chmod sub core .i file permission
            os.chmod(dis_i_file_cube, stat.S_IRUSR + stat.S_IWUSR)
            os.chmod(dis_i_file_vec, stat.S_IRUSR + stat.S_IWUSR)
        elif CommonUtility.is_m510():
            pre_compile_cmd = gen_compile_cmd_v220(cce_file, dst_i_file, compile_option_tuple_pre, None, '', False)
            CommonUtility.run_cmd_inner(pre_compile_cmd, CompileStage.PRECOMPILE, compile_log_path)
        else:
            pre_compile_cmd = _gen_compile_cmd(cce_file, dst_i_file, compile_option_tuple_pre, '', False)
            CommonUtility.run_cmd_inner(pre_compile_cmd, CompileStage.PRECOMPILE, compile_log_path)
        if not os.path.exists(dst_i_file):
            raise Exception(f"Geneate file {dst_i_file} failed, probably due to error in compile")
        os.chmod(dst_i_file, stat.S_IRUSR + stat.S_IWUSR)
        # get tiling key list and simpel infer code channel
        return KernelInfoInfer.infer_info_from_ifile(dst_i_file, compile_log_path, cce_file, origin_func_name)


def _check_if_gen_placehoder(op_info: OpInfo, is_input: bool) -> bool:
    context = get_context()
    input_output_info = op_info.inputs if is_input is True else op_info.outputs
    if is_input:
        option_mode = context.get_addition("optional_input_mode")
    else:
        option_mode = context.get_addition("optional_output_mode")
    if option_mode != GEN_PLACE_HOLDER_STR:
        return False
    if len(input_output_info) == 0:
        return False
    for param in input_output_info:
        if param is None:
            err_msg = f"[ERROR] : context is {GEN_PLACE_HOLDER_STR}, but have null input, " \
                            f"params are not full, inputs is: {input_output_info}"
            CommonUtility.print_compile_log(op_info.kernel_name, err_msg, AscendCLogLevel.LOG_ERROR)
            raise Exception(err_msg)
    return True


def _set_compile_info(op_info: OpInfo, value_depends: dict = None):
    """set compile info in order to let AOE tools set tune params into compile info
        only support static shape ops
    """
    context = get_context()
    if is_static_shape(op_info.inputs, op_info.outputs, value_depends, op_info.param_type_list):
        from tbe.common.tiling import BANK_CACHE
        if BANK_CACHE is not None and len(BANK_CACHE) != 0:
            tiling = context.get_addition('tune_param')
            if tiling is None:
                from tbe.common.utils.create_kb_query_key import get_op_compile_unique_key
                from tbe.common.repository_manager.interface import cann_kb_search
                info_dict = get_op_compile_unique_key(op_info.op_type, op_info.inputs, op_info.outputs, op_info.attrs,\
                    op_info.impl_mode, False)
                tiling = cann_kb_search(info_dict, search_config={"op_type": op_info.op_type, "full_info": True}, \
                    option={})
            if tiling is not None:
                context.add_compile_info('tune_param', tiling)


def _infer_name(key, sub_operater_infos, chip_version):
    if key == 'stream':
        if sub_operater_infos["sub_operator_kernel_type"] == "KERNEL_TYPE_AIV_ONLY" \
            or sub_operater_infos["sub_operator_kernel_type"] == "KERNEL_TYPE_MIX_AIV_1_0":
            name = f'dav-{chip_version}-vec'
        elif sub_operater_infos["sub_operator_kernel_type"] == "KERNEL_TYPE_MIX_AIC_1_1" \
            or sub_operater_infos["sub_operator_kernel_type"] == "KERNEL_TYPE_MIX_AIC_1_2":
            name = f'dav-{chip_version}-mix'
        else:
            name = f'dav-{chip_version}-cube'
    else:
        name = 'aicore'
    return name


def _update_super_dfx_info(name, chip_version, sub_dfx_info, super_dfx_info):
    if name == f'dav-{chip_version}-mix':
        name_list = [f'dav-{chip_version}-vec', f'dav-{chip_version}-cube']
        for sub_name in name_list:
            if sub_name in super_dfx_info and isinstance(super_dfx_info[sub_name], list):
                super_dfx_info[sub_name].append(sub_dfx_info)
            else:
                super_dfx_info[sub_name] = [sub_dfx_info]
    else:
        if name in super_dfx_info and isinstance(super_dfx_info[name], list):
            super_dfx_info[name].append(sub_dfx_info)
        else:
            super_dfx_info[name] = [sub_dfx_info]


def _json_except_info(compile_info: CompileInfo):
    super_dfx_info = {}
    super_dfx_list = {}
    key = 'aicore'
    chip_version = CommonUtility.get_chip_version()
    if 'stream-fusion' in compile_info.super_kernel_info["sp_options"]:
        stream_fusion = compile_info.super_kernel_info["sp_options"]['stream-fusion']
        if stream_fusion == SuperKernelStreamFusionMode.StreamFusionEnable:
            key = 'stream'
    i = 0
    for sub_op in compile_info.super_kernel_info["op_list"]:
        sub_json_path = sub_op.get("json_path")
        sub_dfx_info = {}
        arg_list = {}
        with open(sub_json_path, 'r') as fd:
            sub_operater_infos = json.load(fd)
            sub_dfx_info["func_name"] = sub_operater_infos["kernelName"]
            sub_dfx_info["split_mode"] = sub_operater_infos.get("split_mode")
            sub_dfx_info["blockDim"] = sub_operater_infos["blockDim"]
            sub_dfx_info["sub_operator_kernel_type"] = sub_operater_infos["sub_operator_kernel_type"]
            sub_dfx_info["sub_operator_early_start_set_flag"] = \
            sub_operater_infos['sub_operator_early_start_set_flag']
            sub_dfx_info["sub_operator_early_start_wait_flag"] = \
                sub_operater_infos['sub_operator_early_start_wait_flag']
            sub_dfx_info["streamid"] = sub_op.get('stream_id')
            sub_dfx_info["send_event_list"] = compile_info.super_kernel_info["send_event_list"][i]
            sub_dfx_info["recv_event_list"] = compile_info.super_kernel_info["recv_event_list"][i]
            arg_list["param_offset"] = compile_info.super_kernel_info["param_offset"][i]
            if compile_info.super_kernel_info["send_event_list"][i]:
                arg_list["notify_param_offset"] = compile_info.super_kernel_info["notify_param_offset"][i]
            else:
                arg_list["notify_param_offset"] = None
            if compile_info.super_kernel_info["recv_event_list"][i]:
                arg_list["wait_param_offset"] = compile_info.super_kernel_info["wait_param_offset"][i]
            else:
                arg_list["wait_param_offset"] = None
            sub_dfx_info["arg_list"] = arg_list
            if "debugOptions" in sub_operater_infos:
                sub_dfx_info["debug_option"] = sub_operater_infos["debugOptions"]
                sub_dfx_info["debug_size"] = sub_operater_infos["debugBufSize"]
            name = _infer_name(key, sub_operater_infos, chip_version)
        _update_super_dfx_info(name, chip_version, sub_dfx_info, super_dfx_info)
        i += 1
    super_dfx_list["kernelList"] = super_dfx_info
    return super_dfx_list


def _init_param_value(op_info: OpInfo, tiling_info: TilingInfo, js):
    if op_info.init_value_list is not None:
        #generate clear output for atomic instrs
        param_of_init_values = [None for _ in op_info.inputs]
        if tiling_info.clear_atomic:
            for output, init_value in zip(op_info.outputs, op_info.init_value_list):
                if init_value is not None:
                    if init_value.isdigit() :
                        # generate init value for InitValue(uint64_t)
                        param_init_value = {'dtype': output['dtype'], 'init_value': int(init_value)}
                    else :
                        try:
                            init_value_json = json.loads(init_value)
                            # generate init value for InitValue(std::vector<ScalarVar>)
                            if init_value_json["is_list"]:
                                param_init_value = {'dtype': init_value_json[output['dtype']]['type'],
                                                    'init_value': init_value_json[output['dtype']]['value']}
                            else :
                            #  generate init value for InitValue(ScalarVar)
                                param_init_value = {'dtype': init_value_json['type'],
                                                    'init_value': init_value_json['value']}
                        except Exception as err:
                            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE,
                                ("read initValue error, reason is:", err))
                    param_of_init_values.append(param_init_value)
                else:
                    param_of_init_values.append(None)
        else:
            param_of_init_values += [None for _ in op_info.outputs]
        # generate null for workspace
        param_of_init_values.append(None)
        js["parameters"] = param_of_init_values


def _json_post_process(compile_info: CompileInfo, op_info: OpInfo, tiling_info: TilingInfo,\
                        input_gen_placehoder: bool, output_gen_placehoder: bool, compile_log_path):
    kernel_meta_path = CommonUtility.get_kernel_meta_dir()
    json_path = os.path.join(kernel_meta_path, compile_info.kernel_name + '.json')
    obj_path = os.path.join(kernel_meta_path, compile_info.kernel_name + '.o')

    try:
        with open(json_path, 'r') as fd:
            js = json.load(fd)
    except Exception as err:
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, ("read json file failed, reason is:", err))
    if input_gen_placehoder:
        js['optionalInputMode'] = GEN_PLACE_HOLDER_STR
    if output_gen_placehoder:
        js['optionalOutputMode'] = GEN_PLACE_HOLDER_STR
    if compile_info.enable_deterministic:
        if get_current_build_config("enable_deterministic_mode") == 1:
            js["deterministic"] = "true"
        else:
            js["deterministic"] = "false"
    js["supportSuperKernel"] = 1
    if not global_var_storage.get_variable("ascendc_dump_disable_compile_options") \
        and compile_info.dump_info.get("dump_type", "") != "":
        js["debugOptions"] = compile_info.dump_info["dump_type"]
        js["debugBufSize"] = global_var_storage.get_variable("ascendc_required_dump_workspace_size")

    # set tilingdata of mc2 operator when online static compile
    if tiling_info.static_shape_flag is True and op_info.mc2_ctx is not None and len(op_info.mc2_ctx) != 0:
        js["runInfo"] = tiling_info.raw_run_info

    # gen sub operator infos for super kernel feature
    js = add_sub_super_kernel_info(js, tiling_info.static_shape_flag, compile_info)

    if compile_info.super_kernel_info.get("timestamp_option") is not None and \
        compile_info.super_kernel_info.get("timestamp_option"):
        del js['workspace']
        js["debugOptions"] = compile_info.super_kernel_info["debug_option"]
        js["debugBufSize"] = compile_info.super_kernel_info["debug_size"]

    if compile_info.super_kernel_info.get("workspace_size") is not None and \
        compile_info.super_kernel_info.get("workspace_size") > 0:
        js['workspace'] = {
            "num": 1,
            "size": [compile_info.super_kernel_info.get("workspace_size")],
            "type": [0]
        }

    # get max tiling size when use tiling new
    if len(compile_info.tiling_key_struct_map) > 0:
        max_tiling_size = _get_tiling_struct_size(compile_info)
    # get max tiling size without register tiling
    elif global_var_storage.get_variable("ascendc_tiling_no_register"):
        max_tiling_size = compile_info.max_tiling_size
        delete_tiling_section(compile_info)
    # get max tiling size when use tiling old
    else:
        max_tiling_size = tiling_info.tiling_data_size

    # updata op_param size by flag of oom
    if "oom" in get_current_build_config("tir.op_debug_config"):
        # tiling need align to 8 bytes, dfx need 8 bytes for dfx point,
        # oom need allocate 8 * (input + output + shape_tensor+ workspace)
        op_param_size = ((max_tiling_size + 7) // 8) * 8 \
                                    + 8 + 8 * DFXSectionGenerator().param_placeholder_num
    else:
        op_param_size = max_tiling_size + 8

    js["opParaSize"] = int(op_param_size)

    if COMPILE_INFO_KEY not in js:
        js[COMPILE_INFO_KEY] = {}
    if tiling_info.static_shape_flag:
        del js[COMPILE_INFO_KEY]
        # generate schedule_mode for static shape
        if tiling_info.static_shape_flag and tiling_info.schedule_mode != 0:
            js["schedule_mode"] = tiling_info.schedule_mode
    if tiling_info.local_memory_size != -1:
        js["localMemorySize"] = tiling_info.local_memory_size
    if "param_type_dynamic" in op_info._fields and op_info.param_type_dynamic:
        js["dynamicParamMode"] = "folded_with_desc"

    _init_param_value(op_info, tiling_info, js)

    if compile_info.super_kernel_info.get("kernel_name") is not None:
        js["SuperkernelInfo"] = _json_except_info(compile_info)

    try:
        with open(obj_path, 'rb') as obj_file:
            js['sha256'] = hashlib.sha256(obj_file.read()).hexdigest()
    except Exception as err:
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, ("read obj_file failed, reason is:", err))
    try:
        with open(json_path, 'w') as fd_write:
            os.chmod(json_path, stat.S_IRUSR + stat.S_IWUSR)
            json.dump(js, fd_write, indent=2)
    except Exception as err:
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, ("write json file failed, reason is:", err))


def _gen_kernel_func_declare_head_with_workspace(tiling_info: TilingInfo, super_kernel_params, func_params):
    dfx_generator = DFXSectionGenerator()
    # static shape do not have tiling
    if CommonUtility.is_v100() or CommonUtility.is_v200():
        if tiling_info.static_shape_flag:
            func_params.append("GM_ADDR workspace")
            func_params.append("GM_ADDR overflowStatus")
        else:
            func_params.append("GM_ADDR workspace")
            func_params.append("GM_ADDR tiling")
            func_params.append("GM_ADDR overflowStatus")
            dfx_generator.insert_param(DFXArgInfo("tiling", DFXParamType.TILING))
    else:
        if tiling_info.static_shape_flag:
            func_params.append("GM_ADDR workspace")
            super_kernel_params.append("workspace")
        else:
            func_params.append("GM_ADDR workspace")
            func_params.append("GM_ADDR tiling")
            super_kernel_params.append("workspace")
            super_kernel_params.append("tiling")
            dfx_generator.insert_param(DFXArgInfo("tiling", DFXParamType.TILING))
    return super_kernel_params, func_params


def _gen_kernel_func_declare_head(kernel_func_dec: str, is_mix: bool, is_single_and_using_hard_sync: bool, \
                                    opinfo: OpInfo, tiling_info: TilingInfo):
    # generate kernel function
    source = kernel_func_dec
    dfx_generator = DFXSectionGenerator()
    func_params = []
    super_kernel_params = []
    needs_ffts = (is_mix or is_single_and_using_hard_sync) and not (CommonUtility.is_c310() or
        CommonUtility.is_310r6() or CommonUtility.is_m510())
    workspace_idx = 0
    if needs_ffts:
        func_params.append("GM_ADDR ffts_addr")
        workspace_idx += 1
        super_kernel_params.append("ffts_addr")
        dfx_generator.insert_param(DFXArgInfo("ffts", DFXParamType.FFTS))

    if opinfo.mc2_ctx:
        for ctx_name in opinfo.mc2_ctx:
            func_params.append("GM_ADDR {}".format(ctx_name))
            workspace_idx += 1
            super_kernel_params.append(str(ctx_name))
            dfx_generator.insert_param(DFXArgInfo(ctx_name, DFXParamType.MC2CTX))

    for input in opinfo.inputs:
        if input is None:
            continue
        func_params.append("GM_ADDR {}".format(input["param_name"]))
        workspace_idx += 1
        super_kernel_params.append(input["param_name"])
        dfx_generator.insert_param(DFXArgInfo(input["param_name"], DFXParamType.INPUT))

    for output in opinfo.outputs:
        if output is None:
            continue
        func_params.append("GM_ADDR {}".format(output["param_name"]))
        workspace_idx += 1
        super_kernel_params.append(output["param_name"])
        dfx_generator.insert_param(DFXArgInfo(output["param_name"], DFXParamType.OUTPUT))

    if opinfo.output_shape_depend_on_compute is not None and len(opinfo.output_shape_depend_on_compute) > 0:
        func_params.append("GM_ADDR __ascendc_output_shape")
        workspace_idx += 1
        super_kernel_params.append("__ascendc_output_shape")
        dfx_generator.insert_param(DFXArgInfo("shape_tensor", DFXParamType.SHAPE_TENSOR))
        # modify point type for OutputShapeDependOnCompute output
        for index in opinfo.output_shape_depend_on_compute:
            parameter: DFXArgInfo = dfx_generator.get_param(opinfo.outputs[index]["param_name"])
            parameter.point_type = DFXPointType.LEVEL_1_FOR_SHAPE_TENSOR
        # for static shape, set size to max value, len(OutputShapeDependOnCompute output) * mix dim(8) * uint64_t(8)
        if tiling_info.static_shape_flag:
            dfx_generator.set_size_of_dfx_info("shape_tensor", len(opinfo.output_shape_depend_on_compute) * 8 * 8)

    # dynamic: must add workspace, static: if workspace_size >= 0 add workspace
    if not tiling_info.static_shape_flag or tiling_info.static_workspace_size >= 0:
        dfx_generator.insert_param(DFXArgInfo("workspace", DFXParamType.WORKSPACE))

    super_kernel_params, func_params = \
        _gen_kernel_func_declare_head_with_workspace(tiling_info, super_kernel_params, func_params)
    if global_var_storage.get_variable("ascendc_enable_super_kernel") is True:
        global_var_storage.set_variable("ascendc_sub_super_kernel_params", super_kernel_params)
        source += "uint64_t args_offset) {\n"
        source += "    GM_ADDR *param_base = (GM_ADDR *)get_para_base();\n"
        for param in func_params:
            source += f"    {param} = param_base[args_offset++];\n"
    else:
        source += ", ".join(func_params) + ") {\n"
    return source, workspace_idx


def _gen_set_workspace_codes(is_mix: bool, is_single_and_using_hard_sync: bool, \
    opinfo: OpInfo, tiling_info: TilingInfo, dump_size: int, \
    compile_options: list, compile_info: CompileInfo):
    # set workspace
    source = ""
    if global_var_storage.get_variable("ascendc_enable_dump_workspace") is True or \
        (not CommonUtility.is_support_workspace_offset()):
        source += "    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);\n"
    else:
        source += "    GM_ADDR usrWorkspace = workspace + AscendC::RESERVED_WORKSPACE;\n"
    if "oom" in get_current_build_config("tir.op_debug_config"):
        source = add_op_param_to_workspace(opinfo, tiling_info, source, dump_size, compile_options, compile_info)

    needs_ffts = (is_mix or is_single_and_using_hard_sync) and not (CommonUtility.is_c310() or
        CommonUtility.is_310r6())
    # set ffts_addr for ascend910b mix op or is_single_and_using_hard_sync scene
    if needs_ffts:
        source += "    icache_preload(1);\n"
        source += "    if (ffts_addr != nullptr) {\n"
        source += "        set_ffts_base_addr((uint64_t)ffts_addr);\n"
        source += "    }\n"
        source += add_time_stamp_codes('TIME_STAMP_WRAP_FFTS_ADDR')

    # restart enable begin position
    if global_var_storage.get_variable("ascendc_enable_aicore_exception_restart"):
        source += "do {\n"

    # is_single_and_using_hard_sync scene not need clear workspace
    if is_mix and (not CommonUtility.is_c310()):  # c310 doesn't need clearWorkspace
        source += f"#ifdef {MIX_CORE_MACRO} \n"
        source += "    if constexpr (g_coreType == AscendC::AIC) {\n"
        source += "        matmul::clearWorkspace(workspace);\n"
        source += add_time_stamp_codes('TIME_STAMP_WRAP_CLEAR_WK_SPAC', 2)
        source += "    }\n"
        source += "#endif\n"
    if "printf" in compile_info.dump_info["dump_type"]:
        source += "#ifdef ASCENDC_DUMP\n"
        source += "    uint64_t __ascendc_tStamp = 0;\n"
        source += "    uint64_t __ascendc_version = 0;\n"
        source += "     __gm__ char* __ascendc_versionStr = nullptr;\n"
        source += "    GetCannVersion(__ascendc_versionStr, __ascendc_version, __ascendc_tStamp);\n"
        source += "    if (__ascendc_tStamp == 0) {\n"
        source += "        AscendC::printf(\"[WARNING]: CANN TimeStamp is invalid, \
CANN TimeStamp is %u\\n\", __ascendc_tStamp);\n"
        source += "    } else {\n"
        source += "        AscendC::printf(\"CANN Version: %s, TimeStamp: %u\\n\", \
(__gm__ const char*)(__ascendc_versionStr), __ascendc_tStamp);\n"
        source += "    }\n"
        source += "#endif\n"
    return source


def _gen_set_mc2_ctx_param(opinfo: OpInfo):
    if opinfo.mc2_ctx is None:
        return ""
    source = ""
    index = 0
    for ctx_name in opinfo.mc2_ctx:
        source += f"    AscendC::SetHcclContext<{index}>({ctx_name});\n"
        index += 1
    return source


def _check_custom_dcci_end_false(compile_option_tuple) -> bool:
    for option_list in [compile_option_tuple.mllvm_options, compile_option_tuple.compile_options]:
        for option in reversed(option_list):
            if not option.startswith('-cce-aicore-dcci-before-kernel-end='):
                continue
            value = option.split('=')[1]
            if value == 'true':
                return False
            elif value == 'false':
                return True
    return False


def gen_meta_info_section(compile_info, op_info):

    meta_info = {}
    section_var = f""
    out_file = compile_info.gen_kernel_func_file
    kernel_name = op_info.kernel_name

    #version
    section_var += \
        f"static const struct BinaryMetaVersion {kernel_name}_kernel_metainfo_version_section __attribute__ "
    section_var += f"((used, section (\".ascend.meta\"))) = "
    section_var += f" {{{{B_TYPE_BIN_VERSION_INFO, sizeof(unsigned int)}}, 0x01}};\n"

    #debug
    debug_options = 0
    debug_buf_size = 0
    debug_options_table = {"printf" :0x001, "dumptensor" : 0x001, "assert" : 0x002, "timestamp" : 0x004, "oom" : 0x008}

    if "oom" in get_current_build_config("tir.op_debug_config"):
        debug_options |= debug_options_table["oom"]
    if not global_var_storage.get_variable("ascendc_dump_disable_compile_options") \
        and compile_info.dump_info.get("dump_type", "") != "":
        for dump_type in compile_info.dump_info["dump_type"].split(','):
            debug_options |= debug_options_table[dump_type]
        debug_buf_size = global_var_storage.get_variable("ascendc_required_dump_workspace_size")
    section_var += \
        f"static const struct BinaryMetaDebug {kernel_name}_kernel_metainfo_debug_section __attribute__ "
    section_var += f"((used, section (\".ascend.meta\"))) = "
    section_var += f" {{{{B_TYPE_DEBUG_INFO, 8}}, {debug_buf_size}, {debug_options}}};\n"

    #dynamicparam
    dynamic_param = 1 if "param_type_dynamic" in op_info._fields and op_info.param_type_dynamic else 0
    section_var += \
        f"static const struct BinaryMetaDynamicParam "
    section_var += f"{kernel_name}_kernel_metainfo_dynamicparam_section __attribute__ "
    section_var += f"((used, section (\".ascend.meta\"))) = "
    section_var += f" {{{{B_TYPE_DYNAMIC_PARAM, 2}}, {dynamic_param}}};\n"

    #optinalparam
    optional_input_mode = 1 if _check_if_gen_placehoder(op_info, True) else 0
    optional_output_mode = 1 if _check_if_gen_placehoder(op_info, False) else 0
    section_var += \
        f"static const struct BinaryMetaOptionalParam "
    section_var += f"{kernel_name}_kernel_metainfo_optionalparam_section __attribute__ "
    section_var += f"((used, section (\".ascend.meta\"))) = "
    section_var += f" {{{{B_TYPE_OPTIONAL_PARAM, 4}}, {optional_input_mode}, {optional_output_mode}}};\n"

    global_var_storage.set_variable("ascendc_meta_info", section_var)


def gen_kernel_fun(compile_info: CompileInfo, func_name: str, opinfo: OpInfo, \
                    tiling_info: TilingInfo, compile_option_tuple):
    compile_options = compile_option_tuple.compile_options
    src_file = compile_info.src_file
    out_file = compile_info.gen_kernel_func_file
    dump_size = compile_info.dump_info["dump_size"]

    file_name = os.path.basename(src_file)
    file_name_without_ext = os.path.splitext(file_name)[0]
    # begin generate code
    # File Isolation Macro
    source = f"#ifndef __{file_name_without_ext.upper()}__KERNEL_FUN_H__\n"
    source += f"#define __{file_name_without_ext.upper()}__KERNEL_FUN_H__\n\n"
    # replace __global micro for usr kernel function, and recover after usr kernel function
    source += "#undef __global__\n"
    source += "#define __global__ inline\n"
    source += f"#include \"{src_file}\"\n"
    source += "#include \"kernel_utils.h\"\n"

    ascendc_dump_on = "-DASCENDC_DUMP=0" not in compile_options
    dump_info = compile_info.dump_info["dump_type"] != "" and ascendc_dump_on

    if (CommonUtility.is_c310() or CommonUtility.is_310r6()) and dump_info:
        source += "#if defined(RAW_AIC_ONLY_DUMP_TENSOR)\n"  # dump L1
        source += "#include \"include/adv_api/matmul/matmul_intf.h\"\n"  # maybe cube only no matmul::clearWorkspace
        source += "#endif\n"

    source += "#undef __global__\n"
    source += "#if ASCENDC_CPU_DEBUG\n"
    source += "#define __global__\n"
    source += "#else\n"
    source += "#define __global__ __attribute__((cce_kernel))\n"
    source += "#endif\n\n"

    if global_var_storage.get_variable("ascendc_tiling_no_register"):
        for tiling_key in compile_info.tiling_key_list:
            source += f"extern __gm__ uint64_t g_custom_tiling_size_meta_{tiling_key};\n"

    # add template_param
    source += gen_template_tiling_params(compile_info)

    is_mix, is_single_and_using_hard_sync = get_v220_kernel_type_mix_flag(compile_info, tiling_info)

    # generate code for l2 cache
    if global_var_storage.get_variable("ascendc_enable_sanitizer") is False and \
        global_var_storage.get_variable("ascendc_debug_compile_options") is False and \
        global_var_storage.get_variable("ascendc_enable_super_kernel") is False:
        if CommonUtility.is_v220() or CommonUtility.is_v200():
            source = get_code_for_l2_cache(compile_info, source, tiling_info)

    # generate kernel function
    auto_gen_kernel_func = f'auto_gen_{func_name}_kernel'

    gen_func_attributes = "__global__"
    if global_var_storage.get_variable("ascendc_enable_super_kernel") is True:
        align_size = compile_info.super_kernel_info["sp_options"].get('func-align', 512)
        gen_func_attributes = gen_func_align_attribute(align_size)
        if '--cce-auto-sync=off' not in compile_options and '--cce-auto-sync' in compile_options:
            gen_func_attributes += " __attribute__((need_auto_sync))"

    kernel_func_dec = f"extern \"C\" {gen_func_attributes} [aicore] void {auto_gen_kernel_func}("
    source_declare, workspace_idx = _gen_kernel_func_declare_head(kernel_func_dec, is_mix,\
        is_single_and_using_hard_sync, opinfo, tiling_info)
    source += source_declare

    if (CommonUtility.is_c310() or CommonUtility.is_310r6()) and dump_info:
        source += "#if defined(ASCENDC_DUMP) && defined(RAW_AIC_ONLY_DUMP_TENSOR)\n"
        source += "    if ASCEND_IS_AIV {\n"
        source += "        AscendC::EnableL1Dump();\n"
        source += "        workspace += " + \
            str(global_var_storage.get_variable(
                "ascendc_required_dump_workspace_size")) + ";\n"
        source += "        AscendC::SetSysWorkspaceForce(workspace);\n"
        source += "        constexpr uint32_t ASCENDC_DUMP_SIZE = 123;\n"
        if is_mix:
            source += "        AscendC::InitDump(true, ASCENDC_DUMP_SIZE);\n"
        else:
            source += "        AscendC::InitDump(false, ASCENDC_DUMP_SIZE);\n"
        source += "        AscendC::DumpL1TensorTransferByUB();\n"
        source += "        return;\n"
        source += "    }\n"
        source += "#endif\n"

    # init dump and system workspace
    if global_var_storage.get_variable("ascendc_enable_super_kernel") is False:
        source += gen_init_dump_code(is_mix, dump_size)
        # set mc2 context
        source += _gen_set_mc2_ctx_param(opinfo)
        source += add_time_stamp_codes('TIME_STAMP_WRAP_MC2_CTX')
        # implicit add aicore exception restart begin position
        # set workspace
        source += _gen_set_workspace_codes(is_mix, is_single_and_using_hard_sync, opinfo, tiling_info, \
                                       dump_size, compile_options, compile_info)
    else:
        source += _gen_set_mc2_ctx_param(opinfo)
        source += "    AscendC::SetSysWorkspaceForce(workspace);\n"
        source += "    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);\n"

        # restart enable begin position
        if global_var_storage.get_variable("ascendc_enable_aicore_exception_restart"):
            source += "do {\n"

    need_ffts = is_mix or is_single_and_using_hard_sync

    # call usr kernel function call
    source += "#if defined(TEMPLATE_PARAMS_LEN) && TEMPLATE_PARAMS_LEN != 0\n"
    source += gen_usr_origin_kernel_function_call(
        func_name, opinfo, tiling_info, has_template=True)
    source += "#else\n"
    source += gen_usr_origin_kernel_function_call(
        func_name, opinfo, tiling_info, has_template=False)
    source += "#endif\n"

    # aicore exception restart main block
    if global_var_storage.get_variable("ascendc_enable_aicore_exception_restart"):
        for key in tiling_info.tiling_key_list:
            source += f"#if {TILING_KEY_MACRO} == {key}UL"
            source += "\n"

            actual_kernel_type = get_actual_kernel_type(key, compile_info, need_ffts, opinfo.kernel_name)

            if actual_kernel_type == CORE_TYPE_CUBE:
                source += "    if ASCEND_IS_AIC {\n"
                source += "        AscendC::PipeBarrier<PIPE_ALL>();\n"
                source += "        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AscendC::SYNC_AIC_FLAG);\n"
                source += "        AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIC_FLAG);\n"
                source += "    }\n"
            elif actual_kernel_type == CORE_TYPE_VEC:
                source += "    AscendC::SyncAll();\n"
            elif actual_kernel_type == CORE_TYPE_MIX:
                source += "    AscendC::SyncAll<false>();\n"
            source += "#endif\n"

        ctx_num = 0
        if opinfo.mc2_ctx is not None:
            ctx_num = len(opinfo.mc2_ctx)
        source += f"    auto __ascendc_is_restart = AscendC::GetRestart({ctx_num});\n"
        source += "    if (__ascendc_is_restart > 0) {\n"
        source += "        AscendC::PipeBarrier<PIPE_ALL>();\n"
        source += "        dcci((__gm__ int64_t*)0, cache_line_t::ENTIRE_DATA_CACHE);\n"
        # add corresponding sync all by kernel type
        for key in tiling_info.tiling_key_list:
            source += f"#if {TILING_KEY_MACRO} == {key}UL"
            source += "\n"

            actual_kernel_type = get_actual_kernel_type(key, compile_info, need_ffts, opinfo.kernel_name)

            if actual_kernel_type == CORE_TYPE_CUBE:
                source += "        if ASCEND_IS_AIC {\n"
                source += "            AscendC::PipeBarrier<PIPE_ALL>();\n"
                source += "            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AscendC::SYNC_AIC_FLAG);\n"
                source += "            AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIC_FLAG);\n"
                source += "        }\n"
            elif actual_kernel_type == CORE_TYPE_VEC:
                source += "        AscendC::SyncAll();\n"
            elif actual_kernel_type == CORE_TYPE_MIX:
                source += "        AscendC::SyncAll<false>();\n"
            source += "#endif\n"
        source += f"        AscendC::SetRestart({ctx_num});\n"
        source += "    } else {\n"
        source += "        break;\n"
        source += "    }\n"
        source += "} while(1);\n"

    from tbe.common.buildcfg.buildcfg_mapping import status_check
    if get_current_build_config(status_check) and (CommonUtility.is_v200() or CommonUtility.is_v100()):
        source += "    AscendC::WriteBackOverflow(overflowStatus);\n"

    if (CommonUtility.is_c310() or CommonUtility.is_310r6()) and dump_info:
        source += "#if defined(ASCENDC_DUMP) && defined(RAW_AIC_ONLY_DUMP_TENSOR)\n"
        source += "    if ASCEND_IS_AIC {\n"
        source += "        pipe_barrier(PIPE_ALL);\n"
        source += "        AscendC::FinalizeL1TensorDump();\n"
        source += "    }\n"
        source += "#endif\n"

    if not global_var_storage.get_variable("ascendc_enable_super_kernel") and \
                    (CommonUtility.is_c310() or CommonUtility.is_310r6() or CommonUtility.is_m510()):
        if _check_custom_dcci_end_false(compile_option_tuple):
            source += gen_dci_codes()

    source += "}\n\n"
    source += "#endif\n"
    # write code into file
    try:
        with os.fdopen(\
            os.open(out_file, os.O_TRUNC | os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as ofd:
            ofd.write(source)
    except Exception as err:
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, ("gen kernel func file failed, reason is:", err))
    return workspace_idx


def gen_tiling_struct_size_and_dfx_section_file(compile_info: CompileInfo, tiling_info: TilingInfo, \
    tiling_key_struct_size_map: dict):
    out_file = compile_info.tiling_and_dfx_utils_file
    source = f"#undef __global__\n"
    source += f"#define __global__ inline\n"
    source += "#include \"kernel_utils.h\"\n"
    source += "#undef __global__\n"
    source += "#if ASCENDC_CPU_DEBUG\n"
    source += "#define __global__\n"
    source += "#else\n"
    source += "#define __global__ __attribute__((cce_kernel))\n"
    source += "#endif\n\n"
    for tiling_key in compile_info.tiling_key_list:
        tiling_struct_info = tiling_key_struct_size_map.get(str(tiling_key), None)
        if tiling_struct_info is not None:
            _, tiling_struct_size = tiling_struct_info
            source += f"__gm__ uint64_t g_custom_tiling_size_meta_{tiling_key} = {tiling_struct_size};\n"
    if compile_info.no_set_kernel_type is False:
        if tiling_info.static_shape_flag:
            tiling_key = tiling_info.tiling_key
            kernel_type = compile_info.tiling_key_kernel_type[str(tiling_key)]
            if kernel_type in [KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1, KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2]:
                cube_marker = "_mix_aic"
                kernel_name = compile_info.kernel_name + cube_marker
                source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                    tiling_info, tiling_key_struct_size_map, kernel_name)
                vec_marker = "_mix_aiv"
                kernel_name = compile_info.kernel_name + vec_marker
                source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                    tiling_info, tiling_key_struct_size_map, kernel_name)
            else:
                current_kernel_name = compile_info.get_kernel_func_name()
                kernel_name = current_kernel_name
                source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                    tiling_info, tiling_key_struct_size_map, kernel_name)
        else:
            for tiling_key in compile_info.tiling_key_list:
                kernel_type = compile_info.tiling_key_kernel_type[str(tiling_key)]  
                if kernel_type.value >= 6 and kernel_type.value <= 7:
                    cube_marker = "_mix_aic"
                    kernel_name = compile_info.kernel_name + '_%s' % tiling_key + cube_marker
                    source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                        tiling_info, tiling_key_struct_size_map, kernel_name)
                    vec_marker = "_mix_aiv"
                    kernel_name = compile_info.kernel_name + '_%s' % tiling_key + vec_marker
                    source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                        tiling_info, tiling_key_struct_size_map, kernel_name)
                elif kernel_type.value >= 2 and kernel_type.value <= 5:
                    if kernel_type in [KernelMetaType.KERNEL_TYPE_MIX_AIC_HARD_SYNC, \
                        KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0]:
                        sub_marker = "_mix_aic"
                    else:
                        sub_marker = "_mix_aiv"
                    kernel_name = compile_info.kernel_name + '_%s' % tiling_key + sub_marker
                elif kernel_type.value >= 0 and kernel_type.value <= 1:
                    kernel_name = compile_info.kernel_name + '_%s' % tiling_key
                    source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                        tiling_info, tiling_key_struct_size_map, kernel_name)
    else:
        if tiling_info.static_shape_flag:
            tiling_key = tiling_info.tiling_key
            if compile_info.code_channel == CORE_TYPE_MIX:
                cube_marker = "_mix_aic"
                kernel_name = compile_info.kernel_name + cube_marker
                source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                    tiling_info, tiling_key_struct_size_map, kernel_name)
                vec_marker = "_mix_aiv"
                kernel_name = compile_info.kernel_name + vec_marker
                source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                    tiling_info, tiling_key_struct_size_map, kernel_name)
            elif compile_info.hard_sync and compile_info.code_channel in [CORE_TYPE_VEC, CORE_TYPE_CUBE]:
                core_type_marker = "_mix_aic" if compile_info.code_channel == CORE_TYPE_CUBE else "_mix_aiv"
                kernel_name = compile_info.kernel_name + core_type_marker
                source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                    tiling_info, tiling_key_struct_size_map, kernel_name)
            else:
                kernel_name = compile_info.get_kernel_func_name()
                source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                    tiling_info, tiling_key_struct_size_map, kernel_name)
        else:
            for tiling_key in compile_info.tiling_key_list:
                if compile_info.code_channel == CORE_TYPE_MIX:
                    cube_marker = "_mix_aic"
                    kernel_name = compile_info.kernel_name + '_%s' % tiling_key + cube_marker
                    source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                        tiling_info, tiling_key_struct_size_map, kernel_name)
                    vec_marker = "_mix_aiv"
                    kernel_name = compile_info.kernel_name + '_%s' % tiling_key + vec_marker
                    source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                        tiling_info, tiling_key_struct_size_map, kernel_name)
                elif compile_info.hard_sync and compile_info.code_channel in [CORE_TYPE_VEC, CORE_TYPE_CUBE]:
                    core_type_marker = "_mix_aic" if compile_info.code_channel == CORE_TYPE_CUBE else "_mix_aiv"
                    kernel_name = compile_info.kernel_name + '_%s' % tiling_key + core_type_marker
                    source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                        tiling_info, tiling_key_struct_size_map, kernel_name)
                else:
                    kernel_name = compile_info.kernel_name + '_%s' % tiling_key
                    source += DFXSectionGenerator().generate_dfx_section_without_tiling_register(tiling_key, \
                        tiling_info, tiling_key_struct_size_map, kernel_name)
    try:
        with os.fdopen(\
            os.open(out_file, os.O_TRUNC | os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as ofd:
            ofd.write(source)
    except Exception as err:
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, ("gen kernel func file failed, reason is:", err))


def _add_op_compile_options_by_customized_json(op_compile_option: str, compile_option_tuple: CompileOptionTuple):
    js = json.loads(op_compile_option)
    if '--cce-auto-sync=off' not in compile_option_tuple.compile_options and js.get('auto_sync') is not False:
        compile_option_tuple.compile_options.append('--cce-auto-sync')
        compile_option_tuple.compile_options.append('-mllvm')
        compile_option_tuple.compile_options.append('-api-deps-filter')
    short_soc_version = global_var_storage.get_variable("ascendc_short_soc_version").lower()
    compile_options_custom = js.get('compile_options')
    if compile_options_custom is not None:
        if '__ALL__' in compile_options_custom:
            for opt in compile_options_custom.get('__ALL__'):
                if opt.startswith('-mllvm'):
                    compile_option_tuple.mllvm_options.append('-mllvm')
                    compile_option_tuple.mllvm_options.append(opt[7:])
                else:
                    compile_option_tuple.compile_options.append(opt)
        if short_soc_version in compile_options_custom:
            for opt in compile_options_custom.get(short_soc_version):
                if opt.startswith('-mllvm'):
                    compile_option_tuple.mllvm_options.append('-mllvm')
                    compile_option_tuple.mllvm_options.append(opt[7:])
                else:
                    compile_option_tuple.compile_options.append(opt)


def _get_tiling_struct_size(compile_info):
    tiling_struct_set = set()
    tiling_struct_size_map = {}
    max_tiling_size = 0
    for _, tiling_struct in compile_info.tiling_key_struct_map.items():
        tiling_struct_set.add(tiling_struct)

    for tiling_struct in tiling_struct_set:
        objdump_cmd = ['llvm-objdump', '-s', '-j',\
            '.ascendc_tiling.{}'.format(tiling_struct), '{}'.format(compile_info.dst_file)]
        proc = subprocess.Popen(objdump_cmd, stdout=subprocess.PIPE, stderr=None)
        (out, _) = proc.communicate()
        '''
        e.g.
        Contents of section .ascend.meta.TilingData:         # main_tiling_info[0]
        0000 50000000 00000000                    P.......   # main_tiling_info[1] that needs to be parsed
        '''
        tiling_str_info = out.decode('utf-8')
        if TILING_KEY_SEARCH_KEYWORD in tiling_str_info:    # key words from llvm-objdump .ascendc_tiling.
            main_line_start_index = tiling_str_info.index(TILING_KEY_SEARCH_KEYWORD)
            main_tiling_info = tiling_str_info[main_line_start_index:].split("\n")
            hex_num = main_tiling_info[1].split(' ')[2:4]
            hex_num_str = CommonUtility.parser_uint64_hex_num(hex_num)
            bytes_data = bytes.fromhex(hex_num_str)
            dec_data = struct.unpack('>Q', bytes_data)[0]
            tiling_struct_size_map[tiling_struct] = dec_data
            max_tiling_size = max(max_tiling_size, dec_data)

    # The sk sub operator failed to rm tiling section because llvm-objcopy could not correctly process the ar obj.
    if global_var_storage.get_variable("ascendc_enable_super_kernel") is True and \
                                        compile_info.is_super_kernel_compile is False:
        CommonUtility.print_compile_log(compile_info.kernel_name, \
            "[Superkernel]In sk sub kernel compile, do not need rm tiling seciton!", AscendCLogLevel.LOG_INFO)
        return max_tiling_size
    #remove ascendc_tiling section
    objdump_cmd = ['llvm-objcopy', '--remove-section=.ascendc_tiling.*', '{}'.format(compile_info.dst_file)]
    proc = subprocess.Popen(objdump_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    CommonUtility.print_compile_log(compile_info.kernel_name, "need rm tiling seciton!", AscendCLogLevel.LOG_INFO)
    (out, _) = proc.communicate()
    return max_tiling_size

def _get_tiling_struct_without_register_size(compile_info: CompileInfo):
    section_name_set = set()
    tiling_key_struct_size_map = {}  # tiling_key -> (tiling_struct, struct_size)
    max_tiling_size = 0
    objdump_cmd = ['llvm-objdump', '-s', f'{compile_info.dst_file}']
    proc = subprocess.Popen(objdump_cmd, stdout=subprocess.PIPE, stderr=None)
    (out, _) = proc.communicate()
    tiling_str_info = out.decode('utf-8')
    tiling_lines = [line for line in tiling_str_info.splitlines() if '.ascendc_tiling' in line]
    pattern = re.compile(r'\.ascendc_tiling\.[^\s]+')

    for line in tiling_lines:
        for match in pattern.findall(line):
            # match eg. ".ascendc_tiling.optiling::TilingData1_2UL.0"
            match = match.rstrip(':;,')
            section_name_set.add(match)
            name_part = match.split('.ascendc_tiling.', 1)[1]
            name_part = name_part.rsplit('.', 1)[0]

            if '_' in name_part:
                tiling_struct, tiling_key_value = name_part.rsplit('_', 1)
                if tiling_key_value.endswith('UL'):
                    tiling_key_value = tiling_key_value[:-2]
                tiling_key_struct_size_map[tiling_key_value] = (tiling_struct, 0)

    for section_name in section_name_set:
        objdump_cmd = ['llvm-objdump', '-s', '-j', '{}'.format(section_name), '{}'.format(compile_info.dst_file)]
        proc = subprocess.Popen(objdump_cmd, stdout=subprocess.PIPE, stderr=None)
        (out, _) = proc.communicate()
        tiling_str_info = out.decode('utf-8')
        if TILING_KEY_SEARCH_KEYWORD in tiling_str_info:    # key words from llvm-objdump .ascendc_tiling.
            main_line_start_index = tiling_str_info.index(TILING_KEY_SEARCH_KEYWORD)
            main_tiling_info = tiling_str_info[main_line_start_index:].split("\n")
            hex_num = main_tiling_info[1].split(' ')[2:4]
            hex_num_str = CommonUtility.parser_uint64_hex_num(hex_num)
            bytes_data = bytes.fromhex(hex_num_str)
            dec_data = struct.unpack('>Q', bytes_data)[0]
            name_part = section_name.split('.ascendc_tiling.', 1)[1].rsplit('.', 1)[0]
            if '_' in name_part:
                tiling_struct, tiling_key_value = name_part.rsplit('_', 1)
                if tiling_key_value.endswith('UL'):
                    tiling_key_value = tiling_key_value[:-2]
                    tiling_key_struct_size_map[tiling_key_value] = (tiling_struct, dec_data)
            max_tiling_size = max(max_tiling_size, dec_data)
    compile_info.max_tiling_size = max_tiling_size
    return tiling_key_struct_size_map


def delete_tiling_section(compile_info: CompileInfo):
    # The sk sub operator failed to rm tiling section because llvm-objcopy could not correctly process the ar obj.
    if global_var_storage.get_variable("ascendc_enable_super_kernel") is True and \
                                        compile_info.is_super_kernel_compile is False:
        CommonUtility.print_compile_log(compile_info.kernel_name, \
            "[Superkernel]In sk sub kernel compile, do not need rm tiling seciton!", AscendCLogLevel.LOG_INFO)
        return
    #remove ascendc_tiling section
    objdump_cmd = ['llvm-objcopy', '--remove-section=.ascendc_tiling.*', '{}'.format(compile_info.dst_file)]
    proc = subprocess.Popen(objdump_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    CommonUtility.print_compile_log(compile_info.kernel_name, "need rm tiling seciton!", AscendCLogLevel.LOG_INFO)
    (out, _) = proc.communicate()
    return


def _update_compile_option(kernel_name: str, compile_options: list, extend_options: dict):
    bisheng = os.environ.get('BISHENG_REAL_PATH')
    if bisheng is None:
        bisheng = shutil.which("bisheng")
    if bisheng is not None:
        bisheng_path = os.path.dirname(bisheng)
        tikcpp_path = os.path.realpath(
            os.path.join(bisheng_path, "..", "..", "tikcpp"))
    else:
        tikcpp_path = os.path.realpath(
            "/usr/local/Ascend/latest/compiler/tikcpp")
    cann_version_file_path = os.path.join(tikcpp_path, "..", "..",
                                          "include", "version", "cann_version.h")
    if os.path.exists(cann_version_file_path):
        compile_options.append("-include" + cann_version_file_path)
    else:
        CommonUtility.print_compile_log(
            kernel_name, "not found cann_version.h", AscendCLogLevel.LOG_WARNING)

    if extend_options.get('opp_kernel_hidden_dat_path', None) is not None:
        compile_options.append("-cce-vfs")
        compile_options.append(extend_options.get('opp_kernel_hidden_dat_path'))


def compile_op_common_part(cce_file: str, origin_func_name: str, op_info: OpInfo, compile_option_tuple,
                           infered_info_from_ifile: InferChannelParamsFromIFile, extend_options: dict):
    value_depend_dict = extend_options.get("valueDepend")
    kernel_meta_dir = CommonUtility.get_kernel_meta_dir()
    distinct_tag =  CommonUtility.get_distinct_filename_tag()
    compile_log_path = None
    if global_var_storage.get_variable("ascendc_compile_debug_config"):
        compile_log_path = os.path.join(kernel_meta_dir, op_info.kernel_name + distinct_tag + '.log')

    input_gen_placehoder = _check_if_gen_placehoder(op_info, True)
    output_gen_placehoder = _check_if_gen_placehoder(op_info, False)

    LogUtil.detail_log_print(
        op_info.kernel_name,
        COMPILE_STAGE_MSG_INFO["generate_tiling_start"],
        AscendCLogLevel.LOG_INFO
    )
    tiling_info: TilingInfo = CommonUtility.get_tiling_info_by_tiling(
                                op_info, infered_info_from_ifile, value_depend_dict)

    CommonUtility.print_compile_log(op_info.kernel_name, "get tiling info success", AscendCLogLevel.LOG_INFO)

    file_name_tag = distinct_tag + "_tiling_data.h"
    tiling_data_file_path = os.path.join(kernel_meta_dir, op_info.kernel_name + file_name_tag)
    tiling_info.save_file(tiling_data_file_path)
    global_var_storage.set_variable("ascendc_is_static_op", tiling_info.static_shape_flag)
    # replace tiling key when tiling_key is set in compile params
    tiling_key_list = infered_info_from_ifile.tiling_key_list
    context_tiling_key = get_context().get_addition("tiling_key")
    # override customized tiling key list if the input is passed from
    customize_tiling_key = "customized_tiling_key_list" 
    if customize_tiling_key in extend_options and isinstance(extend_options[customize_tiling_key], list):
        context_tiling_key = extend_options[customize_tiling_key]
    if context_tiling_key:
        new_tiling_keys = []
        for tiling_key in context_tiling_key:
            if tiling_key in tiling_key_list:
                new_tiling_keys.append(tiling_key)
            else:
                CommonUtility.print_compile_log(op_info.kernel_name,\
                f"given tiling key {tiling_key} is not in supported tiling_key list", AscendCLogLevel.LOG_WARNING)
        tiling_key_list = new_tiling_keys
        if len(tiling_key_list) == 0:
            msg_info = "None of the given tiling keys are in the supported list."
            LogUtil.log_print(op_info.kernel_name, msg_info, AscendCLogLevel.LOG_WARNING)
            sys.exit(1)
    code_channel = infered_info_from_ifile.code_channel
    hardware_sync_in_asm = False
    # code channel can not infer by .i neet infer by .o
    if code_channel == -1 and infered_info_from_ifile.no_set_kernel_type is True:
        dst_file_header = os.path.join(kernel_meta_dir, op_info.kernel_name + "_infer_channel")
        CommonUtility.print_compile_log(op_info.kernel_name, \
            "get kernel type by infer channel...", AscendCLogLevel.LOG_INFO)
        code_channel, hardware_sync_in_asm = get_code_channel_v220_by_first_tiling_key(
            InferChannelParams(cce_file, dst_file_header, compile_option_tuple, \
                               tiling_key_list[0], tiling_info, \
                               compile_log_path, infered_info_from_ifile.no_kfc_server_flag))
        CommonUtility.print_compile_log(op_info.kernel_name, \
            "get kernel type by infer channel success", AscendCLogLevel.LOG_INFO)

    default_dump_info = {'dump_type': '', 'dump_size': 1024}

    compile_info = CompileInfo()
    compile_info.src_file = cce_file
    compile_info.dst_file = os.path.join(kernel_meta_dir, op_info.kernel_name + ".o")
    compile_info.kernel_name = op_info.kernel_name
    compile_info.origin_func_name = origin_func_name
    compile_info.op_type = op_info.op_type
    compile_info.code_channel = code_channel
    compile_info.tiling_key_list = tiling_key_list
    compile_info.compile_log_path = compile_log_path
    compile_info.hard_sync = infered_info_from_ifile.hard_sync or hardware_sync_in_asm
    compile_info.enable_deterministic = infered_info_from_ifile.enable_deterministic
    compile_info.tiling_key_deterministic = infered_info_from_ifile.tiling_key_deterministic
    compile_info.tiling_key_kernel_type = infered_info_from_ifile.tiling_key_kernel_type
    compile_info.no_set_kernel_type = infered_info_from_ifile.no_set_kernel_type
    compile_info.default_kernel_type = infered_info_from_ifile.default_kernel_type
    compile_info.dump_info = infered_info_from_ifile.dump_info \
        if (infered_info_from_ifile.dump_info.get('dump_type') is not None
            and infered_info_from_ifile.dump_info.get('dump_size') is not None) \
        else default_dump_info
    compile_info.template_tiling_info = infered_info_from_ifile.template_tiling_info
    compile_info.tiling_key_struct_map = infered_info_from_ifile.tiling_key_struct_map

    set_dump_assert_flag(compile_info)

    # if enable super kernel, get super kernel option to compile info
    if global_var_storage.get_variable("ascendc_enable_super_kernel") is True:
        compile_info.super_kernel_early_start_set_flag = infered_info_from_ifile.super_kernel_early_start_set_flag
        compile_info.super_kernel_early_start_wait_flag = infered_info_from_ifile.super_kernel_early_start_wait_flag
        sp_info = get_context().get_addition("super_kernel_sub_info")
        if sp_info is not None:
            compile_info.super_kernel_info["sp_options"] = \
                parse_super_kernel_options(sp_info.get("super_kernel_options", ""))
        else:
            compile_info.super_kernel_info["sp_options"] = {}
    LogUtil.detail_log_print(
        op_info.kernel_name,
        COMPILE_STAGE_MSG_INFO["generate_tiling_end"],
        AscendCLogLevel.LOG_INFO
    )
    # generate kernel fun for ffts_addr, overflow, workspace
    msg_info = "<{}> <{}> generate kernel stub start".format(compile_info.op_type, compile_info.tiling_key_list)
    LogUtil.detail_log_print(op_info.kernel_name, msg_info, AscendCLogLevel.LOG_INFO)
    file_name_tag = distinct_tag + "_kernel.cpp"
    compile_info.gen_kernel_func_file = os.path.join(kernel_meta_dir, op_info.kernel_name + file_name_tag)

    # generate tiling struct size, dfx section
    if global_var_storage.get_variable("ascendc_tiling_no_register"):
        file_name_tag = distinct_tag + "_meta_info.cpp"
        compile_info.tiling_and_dfx_utils_file = os.path.join(kernel_meta_dir, op_info.kernel_name + \
            file_name_tag)
        file_name_tag = distinct_tag + "_meta_info.o"
        compile_info.tiling_and_dfx_utils_bin_path = os.path.join(kernel_meta_dir, op_info.kernel_name + \
            file_name_tag)

    ascendc_dump_on = "-DASCENDC_DUMP=0" not in compile_option_tuple.compile_options
    dump_info = compile_info.dump_info["dump_type"] != "" and ascendc_dump_on
    compile_info.raw_tiling_key_kernel_type = copy.deepcopy(compile_info.tiling_key_kernel_type)
    if (CommonUtility.is_c310()) and dump_info:
        tiling_key_kernel_type = compile_info.tiling_key_kernel_type
        for tiling_key in tiling_key_kernel_type:
            if tiling_key_kernel_type[tiling_key] == KernelMetaType.KERNEL_TYPE_AIC_ONLY:
                tiling_key_kernel_type[tiling_key] = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2

    # dump function determins how kernel wrapper will be, must be handle early
    handle_dump_options(compile_info, compile_option_tuple)

    # dump or acc or timestamp or recognize_simtvf will need extra workspace
    ascendc_enable_dump_workspace = ("-DASCENDC_DUMP=0" not in compile_option_tuple.compile_options) or \
        ("assert" == compile_info.dump_info["dump_type"]) or \
        (global_var_storage.get_variable("ascendc_time_stamp_compile_options") is True) or \
        "-DASCENDC_ACC_DUMP" in compile_option_tuple.compile_options or \
        (global_var_storage.get_variable("ascendc_recognize_simtvf") is True)
    global_var_storage.set_variable("ascendc_enable_dump_workspace", ascendc_enable_dump_workspace)

    if CommonUtility.is_c310() or CommonUtility.is_310r6():
        gen_meta_info_section(compile_info, op_info)
    workspace_idx = gen_kernel_fun(compile_info, origin_func_name, op_info, tiling_info, compile_option_tuple)
    # no dump and no superkernel
    if CommonUtility.is_support_workspace_offset() and (not ascendc_enable_dump_workspace) and \
        (global_var_storage.get_variable("ascendc_enable_super_kernel") is False):
        compile_option_tuple.compile_options.append(f'-DWORKSPACE_PARAM_OFFSET={workspace_idx}')

    # generate compile option for sub operator, when enable super kernel
    if global_var_storage.get_variable("ascendc_enable_super_kernel") is True:
        gen_sub_super_kernel_compile_options(compile_option_tuple, tiling_info, compile_info)

    # check whether ccec_O0 or ccec_g opend in compile context
    compile_info.is_debug = CommonUtility.check_debug_options(compile_option_tuple.compile_options)
    compile_option_tuple.compile_options.append('-DONE_CORE_DUMP_SIZE=' + str(compile_info.dump_info["dump_size"]))
    if global_var_storage.get_variable("ascendc_recognize_simtvf") is True:
        compile_option_tuple.compile_options.append('-DASCENDC_RECOGNIZE_SIMT_VF')


    if global_var_storage.get_variable("ascendc_enable_sanitizer") is False and \
        global_var_storage.get_variable("ascendc_debug_compile_options") is False:
        compile_option_tuple.compile_options.append('-DL2_CACHE_HINT')

    if tiling_info.static_shape_flag:
        compile_option_tuple.compile_options.append('-DCONST_TILING')

    msg_info = "<{}> <{}> generate kernel stub end".format(compile_info.op_type, compile_info.tiling_key_list)
    LogUtil.detail_log_print(op_info.kernel_name, msg_info, AscendCLogLevel.LOG_INFO)
    CommonUtility.print_compile_log(op_info.kernel_name, "start to compile cce file...", AscendCLogLevel.LOG_INFO)
    msg_info = "<{}> <{}> compile kernel start".format(compile_info.op_type, compile_info.tiling_key_list)
    LogUtil.detail_log_print(op_info.kernel_name, msg_info, AscendCLogLevel.LOG_INFO)

    DFXSectionGenerator().generate_dfx_binary(compile_info, op_info, tiling_info)

    if CommonUtility.is_v220() or CommonUtility.is_c310() or CommonUtility.is_310r6():
        if compile_info.no_set_kernel_type is True:
            _compile_ascendc_cce_v220(compile_info, compile_option_tuple, tiling_info)
        else:
            _compile_ascendc_cce_v220_with_kernel_type(compile_info, compile_option_tuple, tiling_info)
    elif CommonUtility.is_m510():
        compile_info.code_channel = CORE_TYPE_CUBE
        compile_info.hard_sync = False
        _compile_ascendc_cce_m510(compile_info, compile_option_tuple, tiling_info)
    elif CommonUtility.is_regbase():
        _compile_ascendc_cce_regbase(compile_info, compile_option_tuple, tiling_info)
    elif CommonUtility.is_v200() and compile_info.no_set_kernel_type is False:
        _compile_ascendc_cce_v200_with_kernel_type(compile_info, compile_option_tuple, tiling_info)
    else:
        _compile_ascendc_cce(compile_info, compile_option_tuple, tiling_info)

    # get tiling struct and size in .asendc.tiling section and generate meta_info.o when using REGISTER_NONE_TILING
    if global_var_storage.get_variable("ascendc_tiling_no_register"):
        tiling_key_struct_size_map = _get_tiling_struct_without_register_size(compile_info)
        gen_tiling_struct_size_and_dfx_section_file(compile_info, tiling_info, tiling_key_struct_size_map)
        chip_version = CommonUtility.get_chip_version()
        arch = f"dav-{chip_version}-vec"
        compile_cmd = gen_compile_cmd_for_meta_info(compile_info.tiling_and_dfx_utils_file, \
            compile_info.tiling_and_dfx_utils_bin_path, compile_option_tuple, arch)
        CommonUtility.run_cmd_inner(compile_cmd, CompileStage.COMPILE, compile_info.compile_log_path)
    msg_info = "<{}> <{}> compile kernel end".format(compile_info.op_type, compile_info.tiling_key_list)
    LogUtil.detail_log_print(op_info.kernel_name, msg_info, AscendCLogLevel.LOG_INFO)
    CommonUtility.print_compile_log(op_info.kernel_name, "compile cce file success", AscendCLogLevel.LOG_INFO)
    msg_info = "<{}> <{}> link kernel start".format(compile_info.op_type, compile_info.tiling_key_list)
    LogUtil.detail_log_print(op_info.kernel_name, msg_info, AscendCLogLevel.LOG_INFO)
    if global_var_storage.get_variable("ascendc_enable_sanitizer"):
        _mssanitizer_link(compile_info.dst_file, compile_info.dst_file, compile_info.compile_log_path)
    # split .o 4
    split_sub_kernel_objs(compile_info.dst_file, tiling_info, compile_info)
    CommonUtility.print_compile_log(op_info.kernel_name, \
        "start to link relocatable for dst obj...", AscendCLogLevel.LOG_INFO)
    if global_var_storage.get_variable("ascendc_enable_super_kernel") is False:
        if not global_var_storage.get_variable("ascendc_tiling_no_register"):
            link_relocatable(compile_info.dst_file, compile_info.compile_log_path)
        else:
            link_relocatable_meta_file(compile_info.dst_file, compile_info.tiling_and_dfx_utils_bin_path, \
                compile_info.compile_log_path)
            if not global_var_storage.get_variable("ascendc_compile_debug_config"):
                CommonUtility.remove_temp_file(compile_info.tiling_and_dfx_utils_bin_path)
    msg_info = "<{}> <{}> link kernel end".format(compile_info.op_type, compile_info.tiling_key_list)
    LogUtil.detail_log_print(op_info.kernel_name, msg_info, AscendCLogLevel.LOG_INFO)
    CommonUtility.print_compile_log(op_info.kernel_name, "link relocatable success", AscendCLogLevel.LOG_INFO)
    _json_post_process(compile_info, op_info, tiling_info, input_gen_placehoder, \
                       output_gen_placehoder, compile_log_path)
    if not global_var_storage.get_variable("ascendc_compile_debug_config"):
        tiling_info.remove_file()
        CommonUtility.remove_temp_file(compile_info.gen_kernel_func_file)
        CommonUtility.remove_temp_file(compile_info.tiling_and_dfx_utils_file)
    CommonUtility.print_compile_log("", \
        "compile Ascend C operator {} success".format(op_info.op_type), AscendCLogLevel.LOG_INFO)
    msg_info = "<{}> <{}> compile op end".format(compile_info.op_type, compile_info.tiling_key_list)
    LogUtil.detail_log_print(op_info.kernel_name, msg_info, AscendCLogLevel.LOG_INFO)


def compile_op(cce_file: str, origin_func_name: str, op_info: OpInfo, compile_options: list = None,
        code_channel: int = -1, op_compile_option: str = "{}", extend_options: dict = {}):
    """get tiling_data/ generate tiling_data file/ compile cce to .o / generate .json file
    Args:
        cce_file (str): cce file to be compiled
        origin_func_name (str): func_name written by user, without md5
        op_info (OpInfo): operator info
        compile_options (list): compile options for bisheng
        code_channel (int): one of CORE_TYPE_MIX/CORE_TYPE_CUBE/CORE_TYPE_VEC
    """
    LogUtil.detail_log_print(op_info.kernel_name, COMPILE_STAGE_MSG_INFO["compile_op_start"], AscendCLogLevel.LOG_INFO)
    LogUtil.detail_log_print(op_info.kernel_name, COMPILE_STAGE_MSG_INFO["preprocess_start"], AscendCLogLevel.LOG_INFO)
    process_ascendc_api_version(cce_file, compile_options, extend_options)
    # online compile reuses thread, dfx infos need to be reset.
    global_var_storage.global_storage_reset()
    if extend_options.get('opp_kernel_hidden_dat_path', None) is None and not os.path.exists(cce_file):
        raise Exception(f"input cce file is not exists, file name: " + cce_file)

    compile_option_tuple = CompileOptionTuple([] if compile_options is None else compile_options, [])
    need_impl_mode_macro = (CommonUtility.is_c310() or CommonUtility.is_m510()) and \
        isinstance(op_info.impl_mode, str) and op_info.impl_mode != ""
    if need_impl_mode_macro:
        impl_mode_def = f"-D{op_info.impl_mode.upper()}_"  # IMPL_MODE_IS
        if impl_mode_def not in compile_option_tuple.compile_options:
            compile_option_tuple.compile_options.append(impl_mode_def)

    _add_op_compile_options_by_customized_json(op_compile_option, compile_option_tuple)

    compile_option_tuple.compile_options = compile_pre_process(op_info, compile_option_tuple.compile_options)

    DFXSectionGenerator().dfx_info_reset(op_info)

    _update_compile_option(op_info.kernel_name, compile_option_tuple.compile_options, extend_options)

    value_depend_dict = extend_options.get("valueDepend")
    _set_compile_info(op_info, value_depend_dict)
    kernel_meta_dir = CommonUtility.get_kernel_meta_dir()

    compile_option_tuple.compile_options.append('-DASCENDC_TPL_KERNEL')
    distinct_tag = CommonUtility.get_distinct_filename_tag()
    compile_log_path = None
    if global_var_storage.get_variable("ascendc_compile_debug_config"):
        compile_log_path = os.path.join(kernel_meta_dir, op_info.kernel_name + distinct_tag + '.log')

    # get tilingkeylist and simple infer code_channel
    CommonUtility.print_compile_log(op_info.kernel_name, \
        "precompile to get some simple kernel info...", AscendCLogLevel.LOG_INFO)
    infered_info_from_ifile = KernelInfoInfer.get_tiling_key_list_and_simple_infer_code_channel(cce_file, \
        os.path.join(kernel_meta_dir, op_info.kernel_name + ".i"), \
        compile_option_tuple, compile_log_path, origin_func_name)
    CommonUtility.print_compile_log(op_info.kernel_name, \
        "precompile to get some simple kernel info success", AscendCLogLevel.LOG_INFO)
    LogUtil.detail_log_print(op_info.kernel_name, COMPILE_STAGE_MSG_INFO["preprocess_end"], AscendCLogLevel.LOG_INFO)

    compile_op_common_part(cce_file, origin_func_name, op_info, compile_option_tuple, infered_info_from_ifile,
                            extend_options)


def compile_op_with_inferinfo(cce_file: str, origin_func_name: str, op_info: OpInfo,
        compile_options: list = None, code_channel: int = -1, op_compile_option: str = "{}",
        extend_options: dict = {}, infered_info_from_ifile: InferChannelParamsFromIFile = None):
    """get tiling_data/ generate tiling_data file/ compile cce to .o / generate .json file
    Args:
        cce_file (str): cce file to be compiled
        origin_func_name (str): func_name written by user, without md5
        op_info (OpInfo): operator info
        compile_options (list): compile options for bisheng
        code_channel (int): one of CORE_TYPE_MIX/CORE_TYPE_CUBE/CORE_TYPE_VEC
    """
    LogUtil.detail_log_print(op_info.kernel_name, COMPILE_STAGE_MSG_INFO["compile_op_start"], AscendCLogLevel.LOG_INFO)
    LogUtil.detail_log_print(op_info.kernel_name, COMPILE_STAGE_MSG_INFO["preprocess_start"], AscendCLogLevel.LOG_INFO)
    process_ascendc_api_version(cce_file, compile_options, extend_options)
    # online compile reuses thread, dfx infos need to be reset.
    global_var_storage.global_storage_reset()
    if extend_options.get('opp_kernel_hidden_dat_path', None) is None and not os.path.exists(cce_file):
        raise Exception(f"input cce file is not exists, file name: " + cce_file)

    compile_option_tuple = CompileOptionTuple([] if compile_options is None else compile_options, [])
    need_impl_mode_macro = (CommonUtility.is_c310() or CommonUtility.is_m510()) and \
        isinstance(op_info.impl_mode, str) and op_info.impl_mode != ""
    if need_impl_mode_macro:
        impl_mode_def = f"-D{op_info.impl_mode.upper()}_"  # IMPL_MODE_IS
        if impl_mode_def not in compile_option_tuple.compile_options:
            compile_option_tuple.compile_options.append(impl_mode_def)

    _add_op_compile_options_by_customized_json(op_compile_option, compile_option_tuple)

    compile_option_tuple.compile_options = compile_pre_process(op_info, compile_option_tuple.compile_options)

    DFXSectionGenerator().dfx_info_reset(op_info)

    _update_compile_option(op_info.kernel_name, compile_option_tuple.compile_options, extend_options)

    compile_option_tuple.compile_options.append('-DASCENDC_TPL_KERNEL')
    value_depend_dict = extend_options.get("valueDepend")
    _set_compile_info(op_info, value_depend_dict)

    compile_op_common_part(cce_file, origin_func_name, op_info, compile_option_tuple, infered_info_from_ifile,
                            extend_options)


def handle_dump_options(compile_info, compile_option_tuple):
    if "assert" not in compile_info.dump_info["dump_type"] and "printf" not in compile_info.dump_info["dump_type"]:
        compile_option_tuple.compile_options.append('-DASCENDC_DUMP=0')
        length_before = len(compile_option_tuple.compile_options)
        CommonUtility.remove_options(compile_option_tuple.compile_options, ['-DASCENDC_DUMP', '-DASCENDC_DUMP=1'])
        length_after = len(compile_option_tuple.compile_options)
        if length_before != length_after:
            CommonUtility.print_compile_log(compile_info.kernel_name, \
                "-DASCENDC_DUMP=1 is deleted because the feature is not used internally in src file", \
                AscendCLogLevel.LOG_WARNING)

    if "assert" == compile_info.dump_info["dump_type"]:
        compile_option_tuple.compile_options.append('-DASCENDC_DUMP_ASSERT_ONLY')


def set_dump_assert_flag(compile_info):
    if compile_info.dump_info["dump_type"] == "assert":
        global_var_storage.set_variable("ascendc_dump_assert_only", True)


def _gen_compile_cmd(src_file: str, dst_file: str, compile_option_tuple, tiling_file: str, \
                            with_tiling_file: bool = True):
    """
    Generate the compile command for the v100/v200 compiler.
    :param src_file: the source file
    :param dst_file: the destination file
    :param extra_options: the extra options
    :param with_tiling_file: whether with the tiling file
    :return: the compile command
    """
    jump_expand_flag = '-cce-aicore-jump-expand=true' in compile_option_tuple.compile_options
    compile_cmd = CommonUtility.ascendc_build_aicore_compile_cmd(src_file, dst_file, "")
    if global_var_storage.get_variable("ascendc_enable_ccache") == True:
        compile_cmd = [os.environ.get("ASCENDC_CCACHE_EXECUTABLE")] + compile_cmd
    to_del_idx = []
    for cmd_idx, cmd in enumerate(compile_cmd):
        if '-fcce-vf-vl=256' in cmd:
            to_del_idx.append(cmd_idx - 1)
            to_del_idx.append(cmd_idx)
        if '-cce-aicore-fp-ceiling' in cmd:
            to_del_idx.append(cmd_idx - 1)
            to_del_idx.append(cmd_idx)
        # whether auto sync or not, it should be ascendc`s charge
        elif '--cce-auto-sync' in cmd:
            to_del_idx.append(cmd_idx)
        # if customize set op jump open, then change jump expand setting which was auto generated
        elif (jump_expand_flag or global_var_storage.get_variable("ascendc_enable_sanitizer")) and \
            '-cce-aicore-jump-expand=false' == cmd:
            compile_cmd[cmd_idx] = '-cce-aicore-jump-expand=true'
        elif cmd == 'ccec':
            compile_cmd[cmd_idx] = global_var_storage.get_variable("ascendc_compiler_path")
    for idx in reversed(to_del_idx):
        del compile_cmd[idx]

    # v100 / v200 add stack size compile_cmd = [cmd.replace('16000', '32000') for cmd in compile_cmd]
    compile_cmd_front = compile_cmd[:3]
    compile_cmd_backend = compile_cmd[3:]
    for option in compile_option_tuple.compile_options:
        compile_cmd_front += [option]
    compile_cmd = compile_cmd_front + compile_cmd_backend
    for opt in compile_option_tuple.mllvm_options:
        compile_cmd += [opt]
    if global_var_storage.get_variable("ascendc_enable_sanitizer"):
        compile_cmd += ["--cce-enable-sanitizer", "-g"]
        compile_cmd += ["-mllvm", "-cce-aicore-long-call", "-mllvm", "-cce-aicore-jump-expand=true"]
    if with_tiling_file:
        compile_cmd += ["-include", tiling_file]
    compile_cmd += ["-std=c++17"]
    compile_cmd += ["--cce-mask-opt"]
    if "oom" in get_current_build_config("tir.op_debug_config"):
        compile_cmd += [f"-D{ASCENDC_OOM}={1}"]
    return compile_cmd


def _compile_single_tiling(tiling_key, compile_info, tiling_info, compile_option_tuple):
    dst_file = compile_info.dst_file[:-2] + '_%s.o' % tiling_key
    compile_cmd = _gen_compile_cmd(compile_info.gen_kernel_func_file, dst_file, compile_option_tuple, \
                            tiling_info.tiling_data_file_path)
    compile_cmd += [f"-D{TILING_KEY_MACRO}={tiling_key}UL"]
    compile_cmd += [f"-D{compile_info.origin_func_name}={compile_info.origin_func_name}_{tiling_key}_tilingkey"]
    kernel_func_name = compile_info.kernel_name + '_%s' % tiling_key
    compile_cmd += [f"-Dauto_gen_{compile_info.origin_func_name}_kernel={kernel_func_name}"]
    section_content = DFXSectionGenerator().generate_dfx_section(tiling_key,\
                                                            tiling_info, kernel_func_name, compile_info, True)
    return compile_cmd, section_content


def _compile_ascendc_cce(compile_info: CompileInfo, compile_option_tuple, tiling_info: TilingInfo):
    """call cce-c to compile a AscendC.cce file, generate a binary file and a json file

    Args:
        compile_info (CompileInfo): compile info for generate .o and .json
        compile_options (list): compile options for bisheng
        tiling_info (TilingInfo): tiling info
    """
    sources = CommonUtility().ascendc_read_file(compile_info.gen_kernel_func_file)

    new_sources = sources[:-1]
    if tiling_info.static_shape_flag:
        compile_cmd = _gen_compile_cmd(compile_info.gen_kernel_func_file, compile_info.dst_file, compile_option_tuple, \
                                            tiling_info.tiling_data_file_path)
        # tbe-pass add "__kernel0" in tbe-codegen and json, we use -D to change function name
        compile_cmd += [f"-Dauto_gen_{compile_info.origin_func_name}_kernel={compile_info.get_kernel_func_name()}"]
        compile_cmd += [f"-D{TILING_KEY_MACRO}={tiling_info.tiling_key}UL"]
        new_sources += DFXSectionGenerator().generate_dfx_section(str(tiling_info.tiling_key),\
                                            tiling_info, compile_info.get_kernel_func_name(), compile_info, True)
        new_sources += "#endif\n"
        # add dfx info section to sourse file
        CommonUtility().ascendc_write_file(compile_info.gen_kernel_func_file, new_sources)

        CommonUtility.run_cmd_inner(compile_cmd, CompileStage.COMPILE, compile_info.compile_log_path)
        target = "cce_core"
        tvm_callback_cce_postproc(target, compile_info.kernel_name, tiling_info.block_dim)
    else:
        obj_files = []
        for tiling_key in compile_info.tiling_key_list:
            dst_file = compile_info.dst_file[:-2] + '_%s.o' % tiling_key
            obj_files.append(dst_file)
        cmds_list = []
        for tiling_key in compile_info.tiling_key_list:
            compile_cmd, section_content = \
                    _compile_single_tiling(tiling_key, compile_info, tiling_info, compile_option_tuple)
            cmds_list.append(compile_cmd)
            new_sources += section_content
        new_sources += "#endif\n"
        # add dfx info section to sourse file
        CommonUtility().ascendc_write_file(compile_info.gen_kernel_func_file, new_sources)
        # compile binary
        compile_multi_tilingkey(compile_info.tiling_key_list, cmds_list,\
            os.path.basename(compile_info.dst_file)[:-2], compile_info.compile_log_path)
        fatbin_objs(obj_files, compile_info.dst_file, compile_info.is_debug, compile_info.compile_log_path)
        target = "cce_core"
        tvm_callback_cce_postproc(target, compile_info.kernel_name, tiling_info.block_dim)
        _dynamic_kernel_list_to_json(compile_info.kernel_name, compile_info.tiling_key_list, \
            compile_info.enable_deterministic, compile_info.tiling_key_deterministic)


def _get_sub_kernel_name(compile_info: CompileInfo, core_type: int):
    core_type_marker = "_mix_aic" if core_type == CORE_TYPE_CUBE else "_mix_aiv"
    # i.e. change demo_kernel.o to demo_kernel_mix_aic.o
    sub_kernel_name = compile_info.kernel_name + core_type_marker
    return sub_kernel_name


def _generate_section_content(kernel_name: str, tiling_key: str, \
                            kernel_type: KernelMetaType, tiling_info: TilingInfo, compile_info: CompileInfo):
    if global_var_storage.get_variable("ascendc_enable_super_kernel") is True:
        return ""
    section_content = ""
    section_content += f"\n#if {TILING_KEY_MACRO} == {tiling_key}UL\n"
    section_content += get_ktype_section_variable(f"{kernel_name}_section",
                                                  f"{kernel_name}", kernel_type)
    section_content += f"#endif\n"
    section_content += DFXSectionGenerator().generate_dfx_section(tiling_key, tiling_info, kernel_name, compile_info)
    return section_content


def _get_compile_cmd_and_section_content(compile_info: CompileInfo, arch: str, \
    compile_option_tuple, tiling_info: TilingInfo, tiling_key: str):
    compile_cmd = gen_compile_cmd_v220(compile_info.gen_kernel_func_file, compile_info.dst_file, \
        compile_option_tuple, arch, tiling_info.tiling_data_file_path)

    if CommonUtility.is_c310() or CommonUtility.is_310r6():
        if compile_info.raw_tiling_key_kernel_type.get(str(tiling_key)) == KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2:
            compile_cmd += [f"-D__ASCENDC_ENABLE_VEC_TAIL_TILING_COPY__"]
        if compile_info.raw_tiling_key_kernel_type.get(str(tiling_key)) == KernelMetaType.KERNEL_TYPE_AIC_ONLY:
            compile_cmd += [f"-DRAW_AIC_ONLY_DUMP_TENSOR"]

    current_kernel_name = ""
    kernel_type = compile_info.tiling_key_kernel_type[str(tiling_key)]
    if tiling_info.static_shape_flag:
        if kernel_type.value >= 2:
            current_kernel_name = compile_info.kernel_name
            current_kernel_name = gen_sub_kernel_name(current_kernel_name, arch, kernel_type.name,\
                compile_info.dst_file)
            compile_cmd += [f"-Dauto_gen_{compile_info.origin_func_name}_kernel={current_kernel_name}"]
        else:
            current_kernel_name = compile_info.get_kernel_func_name()
            current_kernel_name = gen_sub_kernel_name(current_kernel_name, "AiCore", kernel_type.name,\
                compile_info.dst_file)
            compile_cmd += [f"-Dauto_gen_{compile_info.origin_func_name}_kernel={current_kernel_name}"]
    else:
        if kernel_type.value >= 2:
            current_kernel_name = compile_info.kernel_name[:-7] + tiling_key + compile_info.kernel_name[-8:]
            compile_cmd += [f"-Dauto_gen_{compile_info.origin_func_name}_kernel={current_kernel_name}"]
            set_dynamic_sub_func_names_of_super_kernel_with_kernel_type(tiling_key, arch, kernel_type.name, \
                                                            current_kernel_name)
        else:
            current_kernel_name = compile_info.kernel_name + '_%s' % tiling_key
            compile_cmd += [f"-Dauto_gen_{compile_info.origin_func_name}_kernel={current_kernel_name}"]
            set_dynamic_sub_func_names_of_super_kernel_with_kernel_type(tiling_key, "AiCore", kernel_type.name, \
                                                            current_kernel_name)
    if kernel_type.value >= 6 and kernel_type.value <= 7:
        compile_cmd += [f"-D{MIX_CORE_MACRO}={1}"]
    if kernel_type == KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1:
        compile_cmd += [f"-D__MIX_CORE_AIC_RATION__=1"]
    compile_cmd += [f"-D{TILING_KEY_MACRO}={tiling_key}UL"]
    compile_cmd += [f"-D{compile_info.origin_func_name}={compile_info.origin_func_name}_{tiling_key}_tilingkey"]
    section_content = _generate_section_content(current_kernel_name, tiling_key, kernel_type, tiling_info, compile_info)
    return compile_cmd, section_content


def _compile_ascendc_cce_v220_with_kernel_type_for_static(compile_info: CompileInfo, \
    compile_option_tuple, tiling_info: TilingInfo):
    """call cce-c to compile a AscendC.cce file, generate a binary file and a json file
       for staic shape
    Args:
        compile_info (CompileInfo): compile info for generate .o and .json
        compile_options (list): compile options for bisheng
        tiling_info (TilingInfo): tiling info
    """
    sources = CommonUtility().ascendc_read_file(compile_info.gen_kernel_func_file)
    chip_version = CommonUtility.get_chip_version()

    new_sources = sources[:-1]
    kernel_type = compile_info.tiling_key_kernel_type[str(tiling_info.tiling_key)]
    if kernel_type in [KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1, KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2]:
        cmds_list = []
        dst_file = compile_info.dst_file
        # build cube
        cube_compile_info = _get_sub_compile_info(compile_info, CORE_TYPE_CUBE)
        arch = f"dav-{chip_version}-cube"
        compile_cmd, section_content = _get_compile_cmd_and_section_content(cube_compile_info, arch, \
            compile_option_tuple, tiling_info, tiling_info.tiling_key)
        new_sources += section_content
        cmds_list.append(compile_cmd)
        vec_compile_info = _get_sub_compile_info(compile_info, CORE_TYPE_VEC)
        arch = f"dav-{chip_version}-vec"
        compile_cmd, section_content = _get_compile_cmd_and_section_content(vec_compile_info, arch, \
            compile_option_tuple, tiling_info, tiling_info.tiling_key)
        new_sources += section_content
        cmds_list.append(compile_cmd)
        new_sources += global_var_storage.get_variable("ascendc_meta_info")
        new_sources += "#endif\n"
        # add dfx info section to sourse file
        CommonUtility().ascendc_write_file(compile_info.gen_kernel_func_file, new_sources)
        for cmd in cmds_list:
            CommonUtility.run_cmd_inner(cmd, CompileStage.COMPILE, compile_info.compile_log_path)
        # fatbin 2o->1o
        mix_objs = [cube_compile_info.dst_file, vec_compile_info.dst_file]
        if compile_info.enable_final_super_kernel_compile is True:
            compile_info.super_kernel_objs = mix_objs
        else:
            fatbin_objs(mix_objs, dst_file, compile_info.is_debug, compile_info.compile_log_path)
        _gen_mix_sub_json(compile_info, tiling_info, CORE_TYPE_CUBE)
        if kernel_type.value == 6:
            tiling_info.task_ration = 1
        task_ration_str = f"1:{tiling_info.task_ration}"
        _gen_mix_json_from_seperate_json_for_kernel_type(compile_info.kernel_name, task_ration_str, CORE_TYPE_CUBE, \
            True)
        set_soc_spec("AiCore")
    elif kernel_type in [KernelMetaType.KERNEL_TYPE_MIX_AIV_HARD_SYNC, KernelMetaType.KERNEL_TYPE_MIX_AIC_HARD_SYNC, \
            KernelMetaType.KERNEL_TYPE_MIX_AIV_1_0, KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0]:
        if kernel_type in [KernelMetaType.KERNEL_TYPE_MIX_AIC_HARD_SYNC, KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0]:
            arch = f"dav-{chip_version}-cube"
            code_type = CORE_TYPE_CUBE
        else:
            arch = f"dav-{chip_version}-vec"
            code_type = CORE_TYPE_VEC
        sub_compile_info =  _get_sub_compile_info(compile_info, code_type)
        compile_cmd, section_content = _get_compile_cmd_and_section_content(sub_compile_info, arch, \
            compile_option_tuple, tiling_info, tiling_info.tiling_key)
        new_sources += section_content
        new_sources += global_var_storage.get_variable("ascendc_meta_info")
        new_sources += "#endif\n"
        # add dfx info section to sourse file
        CommonUtility().ascendc_write_file(compile_info.gen_kernel_func_file, new_sources)
        CommonUtility.run_cmd_inner(compile_cmd, CompileStage.COMPILE, compile_info.compile_log_path)
        _gen_mix_sub_json(sub_compile_info, tiling_info, code_type)
        mix_objs = [sub_compile_info.dst_file]
        if compile_info.enable_final_super_kernel_compile is True:
            compile_info.super_kernel_objs = mix_objs
        else:
            fatbin_objs(mix_objs, compile_info.dst_file, compile_info.is_debug, compile_info.compile_log_path)
        task_ration_str = f"1:0" if code_type == CORE_TYPE_CUBE else f"0:1"
        _gen_mix_json_from_seperate_json(compile_info.kernel_name, task_ration_str, code_type, \
            True)
        set_soc_spec("AiCore")
    elif kernel_type in [KernelMetaType.KERNEL_TYPE_AIV_ONLY, KernelMetaType.KERNEL_TYPE_AIC_ONLY]:
        arch = f"dav-{chip_version}-cube" if kernel_type == KernelMetaType.KERNEL_TYPE_AIC_ONLY \
            else f"dav-{chip_version}-vec"
        sub_code_type = f"AIC" if kernel_type == KernelMetaType.KERNEL_TYPE_AIC_ONLY else f"AIV"
        optional_core = f"AiCore" if kernel_type == KernelMetaType.KERNEL_TYPE_AIC_ONLY else f"VectorCore"
        set_soc_spec(optional_core)
        compile_cmd, section_content = _get_compile_cmd_and_section_content(compile_info, \
            arch, compile_option_tuple, tiling_info, tiling_info.tiling_key)
        new_sources += section_content
        new_sources += global_var_storage.get_variable("ascendc_meta_info")
        new_sources += "#endif\n"
        # add dfx info section to sourse file
        CommonUtility().ascendc_write_file(compile_info.gen_kernel_func_file, new_sources)
        CommonUtility.run_cmd_inner(compile_cmd, CompileStage.COMPILE, compile_info.compile_log_path)
        _gen_non_mix_sub_json(compile_info, tiling_info, sub_code_type)


def _compile_ascendc_cce_v220_with_kernel_type_for_dynamic(compile_info: CompileInfo, \
    compile_option_tuple, tiling_info: TilingInfo):
    """call cce-c to compile a AscendC.cce file, generate a binary file and a json file
       for dynamic shape
    Args:
        compile_info (CompileInfo): compile info for generate .o and .json
        compile_options (list): compile options for bisheng
        tiling_info (TilingInfo): tiling info
    """
    sources = CommonUtility().ascendc_read_file(compile_info.gen_kernel_func_file)
    chip_version = CommonUtility.get_chip_version()
    new_sources = sources[:-1]
    obj_files = []
    cmds_list_vec = []
    tiling_key_vec = []
    cmds_list_cube = []
    tiling_key_cube = []
    for tiling_key in compile_info.tiling_key_list:
        kernel_type = compile_info.tiling_key_kernel_type[tiling_key]
        if kernel_type.value >= 6 and kernel_type.value <= 7:
            cube_compile_info = _get_sub_compile_info(compile_info, CORE_TYPE_CUBE)
            cube_compile_info.dst_file = cube_compile_info.dst_file[:-2] + "_%s.o" % tiling_key
            arch = f"dav-{chip_version}-cube"
            compile_cmd, section_content = _get_compile_cmd_and_section_content(cube_compile_info, arch, \
                compile_option_tuple, tiling_info, tiling_key)
            new_sources += section_content
            cmds_list_cube.append(compile_cmd)
            obj_files.append(cube_compile_info.dst_file)
            tiling_key_cube.append(tiling_key)
            vec_compile_info = _get_sub_compile_info(compile_info, CORE_TYPE_VEC)
            vec_compile_info.dst_file = vec_compile_info.dst_file[:-2] + "_%s.o" % tiling_key
            arch = f"dav-{chip_version}-vec"
            compile_cmd, section_content = _get_compile_cmd_and_section_content(vec_compile_info, arch, \
                compile_option_tuple, tiling_info, tiling_key)
            new_sources += section_content
            cmds_list_vec.append(compile_cmd)
            obj_files.append(vec_compile_info.dst_file)
            tiling_key_vec.append(tiling_key)
        elif kernel_type.value >= 2 and kernel_type.value <= 5:
            if kernel_type in [KernelMetaType.KERNEL_TYPE_MIX_AIC_HARD_SYNC, KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0]:
                arch = f"dav-{chip_version}-cube"
                code_type = CORE_TYPE_CUBE
            else:
                arch = f"dav-{chip_version}-vec"
                code_type = CORE_TYPE_VEC
            sub_compile_info =  _get_sub_compile_info(compile_info, code_type)
            sub_compile_info.dst_file = sub_compile_info.dst_file[:-2] + '_%s.o' % tiling_key
            compile_cmd, section_content = _get_compile_cmd_and_section_content(sub_compile_info, arch, \
                compile_option_tuple, tiling_info, tiling_key)
            new_sources += section_content
            if code_type == CORE_TYPE_CUBE:
                cmds_list_cube.append(compile_cmd)
                obj_files.append(sub_compile_info.dst_file)
                tiling_key_cube.append(tiling_key)
            else:
                cmds_list_vec.append(compile_cmd)
                obj_files.append(sub_compile_info.dst_file)
                tiling_key_vec.append(tiling_key)
        elif kernel_type.value >= 0 and kernel_type.value <= 1:
            arch = f"dav-{chip_version}-cube" if kernel_type == KernelMetaType.KERNEL_TYPE_AIC_ONLY else \
                f"dav-{chip_version}-vec"
            sub_compile_info = copy.deepcopy(compile_info)
            sub_compile_info.dst_file = sub_compile_info.dst_file[:-2] + '_%s.o' % tiling_key
            compile_cmd, section_content = _get_compile_cmd_and_section_content(sub_compile_info, \
                arch, compile_option_tuple, tiling_info, tiling_key)
            new_sources += section_content
            if arch == f"dav-{chip_version}-cube":
                cmds_list_cube.append(compile_cmd)
                obj_files.append(sub_compile_info.dst_file)
                tiling_key_cube.append(tiling_key)
            else:
                cmds_list_vec.append(compile_cmd)
                obj_files.append(sub_compile_info.dst_file)
                tiling_key_vec.append(tiling_key)
        else:
            raise Exception(f"current kernel type is not suport {kernel_type}")
    new_sources += global_var_storage.get_variable("ascendc_meta_info")
    new_sources += "#endif\n"
    # add dfx info section to sourse file
    CommonUtility().ascendc_write_file(compile_info.gen_kernel_func_file, new_sources)

    if len(cmds_list_vec) != 0:
        compile_multi_tilingkey(tiling_key_vec, cmds_list_vec, \
            os.path.basename(compile_info.dst_file)[:-2] + "_tmp_aiv", compile_info.compile_log_path)

    if len(cmds_list_cube) != 0:
        compile_multi_tilingkey(tiling_key_cube, cmds_list_cube, \
            os.path.basename(compile_info.dst_file)[:-2] + "_tmp_aic", compile_info.compile_log_path)
    fatbin_objs(obj_files, compile_info.dst_file, compile_info.is_debug, compile_info.compile_log_path)
    _generate_final_json(compile_info, tiling_info)


def _compile_ascendc_cce_v220_with_kernel_type(compile_info: CompileInfo, compile_option_tuple,\
    tiling_info: TilingInfo):
    """call cce-c to compile a AscendC.cce file, generate a binary file and a json file with kernel type

    Args:
        compile_info (CompileInfo): compile info for generate .o and .json
        compile_options (list): compile options for bisheng
        tiling_info (TilingInfo): tiling info
    """
    if tiling_info.static_shape_flag:
        _compile_ascendc_cce_v220_with_kernel_type_for_static(compile_info, compile_option_tuple, tiling_info)
    else:
        _compile_ascendc_cce_v220_with_kernel_type_for_dynamic(compile_info, compile_option_tuple, tiling_info)


def _compile_ascendc_cce_v200_with_kernel_type_for_static(compile_info: CompileInfo, \
    compile_option_tuple, tiling_info: TilingInfo):
    """call cce-c to compile a AscendC.cce file, generate a binary file and a json file
       for staic shape
    Args:
        compile_info (CompileInfo): compile info for generate .o and .json
        compile_options (list): compile options for bisheng
        tiling_info (TilingInfo): tiling info
    """
    kernel_type = compile_info.tiling_key_kernel_type[str(tiling_info.tiling_key)]
    if kernel_type in \
            [KernelMetaType.KERNEL_TYPE_MIX_AICORE, KernelMetaType.KERNEL_TYPE_MIX_VECTOR_CORE]:
        # build Aicore
        set_soc_spec("AiCore")
        dst_file = compile_info.dst_file
        aicore_compile_info = _get_sub_compile_info(compile_info, CORE_TYPE_CUBE)
        arch = "dav-m200"
        call_bisheng_v200_static(aicore_compile_info, compile_option_tuple, tiling_info, arch,\
            kernel_type)
        # build vector
        set_soc_spec("VectorCore")
        vec_compile_info = _get_sub_compile_info(compile_info, CORE_TYPE_VEC)
        arch = "dav-m200-vec"
        if kernel_type is KernelMetaType.KERNEL_TYPE_MIX_VECTOR_CORE:
            compile_option_tuple.compile_options.append('-D__ENABLE_VECTOR_CORE__')
        call_bisheng_v200_static(vec_compile_info, compile_option_tuple, tiling_info, arch, kernel_type)
        # fatbin 2o->1o
        mix_objs = [aicore_compile_info.dst_file, vec_compile_info.dst_file]
        fatbin_objs(mix_objs, dst_file, compile_info.is_debug, compile_info.compile_log_path)
        # gen main json
        _gen_static_json_for_mix_v200(compile_info, tiling_info, kernel_type)
    elif kernel_type in [KernelMetaType.KERNEL_TYPE_AICORE]:
        arch = "dav-m200"
        set_soc_spec("AiCore")
        call_bisheng_v200_static(compile_info, compile_option_tuple, tiling_info, arch, \
            kernel_type)
        # gen json for v200
        _gen_static_json_for_no_mix_v200(compile_info, tiling_info, kernel_type)
    else:
        raise Exception(f'current kernel core type is not support')
    return


def _compile_ascendc_cce_v200_with_kernel_type_for_dynamic(compile_info: CompileInfo, \
    compile_option_tuple, tiling_info: TilingInfo, final_kernel_type):
    """call cce-c to compile a AscendC.cce file, generate a binary file and a json file
       for dynamic shape
    Args:
        compile_info (CompileInfo): compile info for generate .o and .json
        compile_options (list): compile options for bisheng
        tiling_info (TilingInfo): tiling info
    """
    obj_files = []
    cmds_list_vec = []
    tiling_key_vec = []
    cmds_list_aicore = []
    tiling_key_aicore = []
    sources = CommonUtility().ascendc_read_file(compile_info.gen_kernel_func_file)

    new_sources = sources[:-1]
    for tiling_key in compile_info.tiling_key_list:
        kernel_type = compile_info.tiling_key_kernel_type[tiling_key]
        if kernel_type in \
            [KernelMetaType.KERNEL_TYPE_MIX_AICORE, KernelMetaType.KERNEL_TYPE_MIX_VECTOR_CORE]:
            # build Aicore
            set_soc_spec("AiCore")
            dst_file = compile_info.dst_file
            aicore_compile_info = _get_sub_compile_info(compile_info, CORE_TYPE_CUBE)
            arch = "dav-m200"
            param = SingleTilingKeyCompileParams(tiling_key, aicore_compile_info, arch, \
                                                tiling_info, compile_info.code_channel, compile_option_tuple)
            dst_file, compile_cmd, section_content = call_bisheng_v200_dynamic(param, kernel_type)
            new_sources += section_content
            cmds_list_aicore.append(compile_cmd)
            obj_files.append(dst_file)
            tiling_key_aicore.append(tiling_key)
            # build vector
            set_soc_spec("VectorCore")
            vec_compile_info = _get_sub_compile_info(compile_info, CORE_TYPE_VEC)
            arch = "dav-m200-vec"
            if kernel_type is KernelMetaType.KERNEL_TYPE_MIX_VECTOR_CORE:
                compile_option_tuple.compile_options.append('-D__ENABLE_VECTOR_CORE__')
            param = SingleTilingKeyCompileParams(tiling_key, vec_compile_info, arch, \
                                                tiling_info, compile_info.code_channel, compile_option_tuple)
            dst_file, compile_cmd, section_content = call_bisheng_v200_dynamic(param, kernel_type)
            new_sources += section_content
            cmds_list_vec.append(compile_cmd)
            obj_files.append(dst_file)
            tiling_key_vec.append(tiling_key)
        elif kernel_type in [KernelMetaType.KERNEL_TYPE_AICORE]:
            arch = "dav-m200"
            set_soc_spec("AiCore")
            param = SingleTilingKeyCompileParams(tiling_key, compile_info, arch, \
                                                tiling_info, compile_info.code_channel, compile_option_tuple)
            dst_file, compile_cmd, section_content = call_bisheng_v200_dynamic(param, kernel_type)
            new_sources += section_content
            cmds_list_aicore.append(compile_cmd)
            obj_files.append(dst_file)
            tiling_key_aicore.append(tiling_key)
        else:
            raise Exception(f'current kernel core type is not support')
                # gen main json
    new_sources += "#endif\n"
    # add dfx info section to sourse file
    CommonUtility().ascendc_write_file(compile_info.gen_kernel_func_file, new_sources)

    if len(cmds_list_vec) != 0:
        compile_multi_tilingkey(tiling_key_vec, cmds_list_vec, \
            os.path.basename(compile_info.dst_file)[:-2] + "_tmp_aiv", compile_info.compile_log_path)

    if len(cmds_list_aicore) != 0:
        compile_multi_tilingkey(tiling_key_aicore, cmds_list_aicore, \
            os.path.basename(compile_info.dst_file)[:-2] + "_tmp_aic", compile_info.compile_log_path)

    fatbin_objs(obj_files, compile_info.dst_file, compile_info.is_debug, compile_info.compile_log_path)
    _gen_dynamic_json_for_v200(compile_info, tiling_info, final_kernel_type)
    return


def _compile_ascendc_cce_v200_with_kernel_type(compile_info: CompileInfo,\
                                                compile_option_tuple, tiling_info: TilingInfo):
    """call cce-c to compile a AscendC.cce file, generate a binary file and a json file

    Args:
        compile_info (CompileInfo): compile info for generate .o and .json
        compile_options (list): compile options for bisheng
        tiling_info (TilingInfo): tiling info
    """
    from .ascendc_compile_v200 import judge_valid_for_v200
    final_kernel_type = judge_valid_for_v200(compile_info.tiling_key_kernel_type)
    if tiling_info.static_shape_flag:
        _compile_ascendc_cce_v200_with_kernel_type_for_static(compile_info, compile_option_tuple, tiling_info)
    else:
        _compile_ascendc_cce_v200_with_kernel_type_for_dynamic(compile_info, \
            compile_option_tuple, tiling_info, final_kernel_type)


def _compile_ascendc_cce_v220(compile_info: CompileInfo, compile_option_tuple, tiling_info: TilingInfo):
    """call cce-c to compile a AscendC.cce file, generate a binary file and a json file

    Args:
        compile_info (CompileInfo): compile info for generate .o and .json
        compile_options (list): compile options for bisheng
        tiling_info (TilingInfo): tiling info
    """
    chip_version = CommonUtility.get_chip_version()
    if compile_info.code_channel == CORE_TYPE_MIX:
        # build cube
        set_soc_spec("AiCore")
        dst_file = compile_info.dst_file
        cube_compile_info = _get_sub_compile_info(compile_info, CORE_TYPE_CUBE)
        arch = f"dav-{chip_version}-cube"
        tiling_key_list = call_bisheng_v220(cube_compile_info, compile_option_tuple, tiling_info, arch,\
            compile_info.code_channel)
        _gen_mix_sub_json(cube_compile_info, tiling_info, CORE_TYPE_CUBE)
        # build vector
        set_soc_spec("VectorCore")
        vec_compile_info = _get_sub_compile_info(compile_info, CORE_TYPE_VEC)
        arch = f"dav-{chip_version}-vec"
        call_bisheng_v220(vec_compile_info, compile_option_tuple, tiling_info, arch, compile_info.code_channel)
        # fatbin 2o->1o
        mix_objs = [cube_compile_info.dst_file, vec_compile_info.dst_file]
        fatbin_objs(mix_objs, dst_file, compile_info.is_debug, compile_info.compile_log_path)
        # gen main json
        task_ration_str = f"1:{tiling_info.task_ration}"
        _gen_mix_json_from_seperate_json(compile_info.kernel_name, task_ration_str, CORE_TYPE_CUBE, True)
        set_soc_spec("AiCore")
        if not tiling_info.static_shape_flag:
            _dynamic_kernel_list_to_json(compile_info.kernel_name, tiling_key_list, \
                compile_info.enable_deterministic, compile_info.tiling_key_deterministic)
    elif compile_info.hard_sync and compile_info.code_channel in [CORE_TYPE_VEC, CORE_TYPE_CUBE]:
        dst_file = compile_info.dst_file
        single_side_compile_info = _get_sub_compile_info(compile_info, compile_info.code_channel)
        arch = f"dav-{chip_version}-vec" if compile_info.code_channel == CORE_TYPE_VEC else f"dav-{chip_version}-cube"
        tiling_key_list = call_bisheng_v220(single_side_compile_info, compile_option_tuple, \
            tiling_info, arch, compile_info.code_channel)
        _gen_mix_sub_json(single_side_compile_info, tiling_info, compile_info.code_channel)
        mix_objs = [single_side_compile_info.dst_file]
        fatbin_objs(mix_objs, dst_file, compile_info.is_debug, compile_info.compile_log_path)
        # gen main json
        task_ration_str = f"1:0" if compile_info.code_channel == CORE_TYPE_CUBE else f"0:1"
        _gen_mix_json_from_seperate_json(compile_info.kernel_name, task_ration_str, compile_info.code_channel, True)
        set_soc_spec("AiCore")
        if not tiling_info.static_shape_flag:
            _dynamic_kernel_list_to_json(compile_info.kernel_name, tiling_key_list, \
                compile_info.enable_deterministic, compile_info.tiling_key_deterministic)
    else:
        if compile_info.code_channel == CORE_TYPE_CUBE:
            arch = f"dav-{chip_version}-cube"
            sub_core_type = "AIC"
            optional_core = "AiCore"
        elif compile_info.code_channel == CORE_TYPE_VEC:
            arch = f"dav-{chip_version}-vec"
            sub_core_type = "AIV"
            optional_core = "VectorCore"  # do the same work with SetOptionalCoreType in cpp
        else:
            raise Exception(f"invalid code_channel = {compile_info.code_channel}")
        set_soc_spec(optional_core)
        tiling_key_list = call_bisheng_v220(compile_info, compile_option_tuple, tiling_info, arch, \
            compile_info.code_channel)
        _gen_non_mix_sub_json(compile_info, tiling_info, sub_core_type)
        if not tiling_info.static_shape_flag:
            _dynamic_kernel_list_to_json(compile_info.kernel_name, tiling_key_list, \
                compile_info.enable_deterministic, compile_info.tiling_key_deterministic)


def _compile_ascendc_cce_m510(compile_info: CompileInfo, compile_option_tuple, tiling_info: TilingInfo):
    """call cce-c to compile a AscendC.cce file, generate a binary file and a json file

    Args:
        compile_info (CompileInfo): compile info for generate .o and .json
        compile_options (list): compile options for bisheng
        tiling_info (TilingInfo): tiling info
    """
    sub_core_type = "AIC"
    optional_core = "AiCore"
    arch = None
    set_soc_spec(optional_core)
    tiling_key_list = call_bisheng_v220(compile_info, compile_option_tuple, tiling_info, arch, \
        compile_info.code_channel)
    _gen_non_mix_sub_json(compile_info, tiling_info, sub_core_type)
    if not tiling_info.static_shape_flag:
        _dynamic_kernel_list_to_json(compile_info.kernel_name, tiling_key_list, \
            compile_info.enable_deterministic, compile_info.tiling_key_deterministic)


def _compile_ascendc_cce_regbase(compile_info: CompileInfo, compile_option_tuple, tiling_info: TilingInfo):
    """call cce-c to compile a AscendC.cce file, generate a binary file and a json file

    Args:
        compile_info (CompileInfo): compile info for generate .o and .json
        compile_options (list): compile options for bisheng
        tiling_info (TilingInfo): tiling info
    """
    soc_arch_map = {"Ascend310B": "dav-m300", "Ascend610Lite": "dav-m310"}
    arch = soc_arch_map.get(global_var_storage.get_variable("ascendc_short_soc_version"))
    value = get_soc_spec("cube_vector_combine")
    value_str_list = value.split(",")
    enable_mix_for_profiling = False
    if value_str_list[0] == 'unknown' or ("fuse" in value_str_list and len(value_str_list)) == 1:
        enable_mix_for_profiling = True
    if enable_mix_for_profiling:
        sub_core_type = "AIC"
        optional_core = "AiCore"
    else:
        sub_core_type = "AIV"
        optional_core = "VectorCore"  # do the same work with SetOptionalCoreType in cpp
    set_soc_spec(optional_core)
    tiling_key_list = _call_bisheng_regbase(compile_info, compile_option_tuple, tiling_info, arch, \
        compile_info.code_channel)
    _gen_non_mix_sub_json(compile_info, tiling_info, sub_core_type)
    if not tiling_info.static_shape_flag:
        _dynamic_regbase_kernel_list_to_json(compile_info.kernel_name, tiling_key_list, \
            compile_info.enable_deterministic, enable_mix_for_profiling, compile_info.tiling_key_deterministic)
    else:
        _static_regbase_kernel_list_to_json(compile_info.kernel_name)


def _get_sub_compile_info(compile_info: CompileInfo, core_type: int):
    sub_compile_info = copy.deepcopy(compile_info)
    core_type_marker = "_mix_aic" if core_type == CORE_TYPE_CUBE else "_mix_aiv"
    # i.e. change demo_kernel.o to demo_kernel_mix_aic.o
    sub_compile_info.dst_file = compile_info.dst_file[:-2] + core_type_marker + compile_info.dst_file[-2:]
    sub_compile_info.kernel_name = compile_info.kernel_name + core_type_marker
    sub_compile_info.sub_core_type = core_type
    return sub_compile_info


def _gen_compile_cmd_regbase(src_file: str, dst_file: str, compile_option_tuple, sub_arch: str, tiling_file: str,\
                                         with_tiling_file: bool = True):
    """
    Generate the compile command for the V300 compiler.
    :param src_file: the source file
    :param dst_file: the destination file
    :param extra_options: the extra options
    :param with_tiling_file: whether with the tiling file
    :return: the compile command
    """
    if global_var_storage.get_variable("ascendc_enable_ccache") == True:
        compile_cmd = [os.environ.get("ASCENDC_CCACHE_EXECUTABLE"), \
            global_var_storage.get_variable("ascendc_compiler_path"), '-c', '-O3']
    else:
        compile_cmd = [global_var_storage.get_variable("ascendc_compiler_path"), '-c', '-O3']

    for option in compile_option_tuple.compile_options:
        compile_cmd += [option]
    compile_cmd += [src_file, "--cce-aicore-arch=%s" % sub_arch,
                    "--cce-aicore-only", "-o", dst_file,
                    "-mllvm", "-cce-aicore-function-stack-size=16000",
                    "-mllvm", "-cce-aicore-addr-transform",
                    "-mllvm", "--cce-aicore-or-combine=false",
                    "-mllvm", "-instcombine-code-sinking=false",
                    "-mllvm", "-cce-aicore-jump-expand=false",
                    "-mllvm", "-cce-aicore-mask-opt=false"]
    for opt in compile_option_tuple.mllvm_options:
        compile_cmd += [opt]

    if with_tiling_file:
        compile_cmd += ["-include", tiling_file]
    compile_cmd += ["-std=c++17"]
    if "oom" in get_current_build_config("tir.op_debug_config"):
        compile_cmd += [f"-D{ASCENDC_OOM}={1}"]
    return compile_cmd


def _compile_single_tiling_regbase(param : SingleTilingKeyCompileParams):
    dst_file = param.compile_info.dst_file[:-2] + '_%s.o' % param.tiling_key
    compile_cmd = _gen_compile_cmd_regbase(param.compile_info.gen_kernel_func_file, dst_file, \
                                               param.compile_option_tuple, param.sub_arch, \
                                               param.tiling_info.tiling_data_file_path)
    compile_cmd += [f"-D{TILING_KEY_MACRO}={param.tiling_key}UL"]
    compile_cmd += \
        [f"-D{param.compile_info.origin_func_name}={param.compile_info.origin_func_name}_{param.tiling_key}_tilingkey"]
    kernel_func_name = param.compile_info.kernel_name + '_%s' % param.tiling_key
    compile_cmd += [f"-Dauto_gen_{param.compile_info.origin_func_name}_kernel={kernel_func_name}"]
    section_content = DFXSectionGenerator().generate_dfx_section(param.tiling_key, \
                                                param.tiling_info, kernel_func_name, param.compile_info, True)
    return compile_cmd, section_content


def _mssanitizer_link(src_file, dst_file, compile_log_path=None):
    """Build the mssanitize link command before link.
    Parameters
    ----------
    src_file : str
        The src object file.

    dst_file : str
        The dst object file.
    """
    short_soc_version = global_var_storage.get_variable("ascendc_short_soc_version")
    if short_soc_version not in global_var_storage.get_variable("ascendc_asan_obj_path"):
        raise Exception("asan config file not support asan.a path")
    if not isinstance(src_file, list):
        src_file = [src_file]
    cmd = [CCECInfo.get_exe("ld.lld"), "-m", "aicorelinux", "-r", "-Ttext=0"]
    cmd.extend(src_file)
    cmd.extend(['--dependent-libraries'])
    cmd.extend(global_var_storage.get_variable("ascendc_asan_obj_path")[short_soc_version])
    cmd.extend([
        "-r",
        "-o",
        "%s" % dst_file,
        ])
    CommonUtility.run_cmd_inner(cmd, CompileStage.FATBIN, compile_log_path)


def _call_bisheng_regbase(compile_info: CompileInfo, compile_option_tuple, tiling_info: TilingInfo, sub_arch: str,\
    code_channel: int):
    """generate bisheng cmd instead of _build_aicore_compile_cmd, since tbe set davinci-m300-{sub_core} in build_cce.cc

    Args:
        compile_info (CompileInfo): compile info for generate .o and .json
        compile_options (list): compile options for bisheng
        tiling_info (TilingInfo): tiling info
        sub_arch (str): m300 arch info
    """
    sources = CommonUtility().ascendc_read_file(compile_info.gen_kernel_func_file)

    new_sources = sources[:-1]
    if tiling_info.static_shape_flag:
        compile_cmd = _gen_compile_cmd(compile_info.gen_kernel_func_file, compile_info.dst_file, compile_option_tuple, \
                                       tiling_info.tiling_data_file_path)
        # tbe-pass add "__kernel0" in tbe-codegen and json, we use -D to change function name
        compile_cmd += [f"-Dauto_gen_{compile_info.origin_func_name}_kernel={compile_info.get_kernel_func_name()}"]
        compile_cmd += [f"-D{TILING_KEY_MACRO}={tiling_info.tiling_key}UL"]
        new_sources += DFXSectionGenerator().generate_dfx_section(str(tiling_info.tiling_key), \
                                            tiling_info, compile_info.get_kernel_func_name(), compile_info, True)
        new_sources += "#endif\n"
        # add dfx info section to sourse file
        CommonUtility().ascendc_write_file(compile_info.gen_kernel_func_file, new_sources)

        CommonUtility.run_cmd_inner(compile_cmd, CompileStage.COMPILE, compile_info.compile_log_path)
        target = "cce_core"
        core_type_info = {var("core_type"): var("")}

        tvm_callback_cce_postproc(target, compile_info.kernel_name, tiling_info.block_dim, \
                                    0, "", None, None, core_type_info, None, None, False, 1, None, None)
    else:
        obj_files = []
        for tiling_key in compile_info.tiling_key_list:
            dst_file = compile_info.dst_file[:-2] + '_%s.o' % tiling_key
            obj_files.append(dst_file)
        cmds_list = []
        for tiling_key in compile_info.tiling_key_list:
            param = SingleTilingKeyCompileParams(\
                tiling_key, compile_info, sub_arch, tiling_info, code_channel, compile_option_tuple)
            compile_cmd, section_content = _compile_single_tiling_regbase(param)
            cmds_list.append(compile_cmd)
            new_sources += section_content
        new_sources += "#endif\n"
        # add dfx info section to sourse file
        CommonUtility().ascendc_write_file(compile_info.gen_kernel_func_file, new_sources)
        # compile binary
        compile_multi_tilingkey(compile_info.tiling_key_list, cmds_list, \
            os.path.basename(compile_info.dst_file)[:-2], compile_info.compile_log_path)
        fatbin_objs(obj_files, compile_info.dst_file, compile_info.is_debug, compile_info.compile_log_path)
        return compile_info.tiling_key_list


def replay_op(op_info: OpInfo, entry_obj: str, code_channel: int, src_file: str, compile_options: list):
    """replay_op feature is at sunset
    """
    return True, "success"


def get_code_channel(src_file: str, kernel_name: str, optype: str, compile_options_input: list = None):
    # replay function needs, so it is reserved
    return CORE_TYPE_MIX