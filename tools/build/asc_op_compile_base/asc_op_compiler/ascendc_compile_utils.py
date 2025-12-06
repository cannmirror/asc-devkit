#!/usr/bin/python
# -*- coding: utf-8 -*-
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
compile check
"""

from typing import List
from asc_op_compile_base.common.buildcfg import get_current_build_config
from asc_op_compile_base.common.context import get_context
from asc_op_compile_base.common.utils.log_utils import AscendCLogLevel
from .get_op_tiling import OpInfo
from .template_tiling import extract_decl_param_options
from .ascendc_common_utility import CommonUtility
from .ascendc_constants import KernelMetaType, KERNEL_TYPE_TO_STR, KERNEL_TYPE_STR_TO_TPL, \
    STR_TO_KERNEL_TYPE_V220, STR_TO_KERNEL_TYPE_V200

GEN_PLACE_HOLDER_STR = 'gen_placeholder'


def get_kernel_meta_type(value):
    for member in KernelMetaType:
        if member.value == value:
            return member
    return None


def check_custom_dcci_end_false(compile_option_tuple):
    has_dcci_end_false: bool = False
    for option_list in [compile_option_tuple.mllvm_options, compile_option_tuple.compile_options]:
        del_ids = []
        for opt_id, option in enumerate(option_list):
            if not option.startswith('-cce-aicore-dcci-before-kernel-end=false'):
                continue
            has_dcci_end_false = True
            if opt_id != 0 and option_list[opt_id - 1] == '-mllvm':
                del_ids.append(opt_id - 1)
            del_ids.append(opt_id)
        for i in reversed(del_ids):
            del option_list[i]
    if has_dcci_end_false:
        compile_option_tuple.compile_options.append('--cce-no-dcache-flush')


def check_if_gen_placehoder(op_info: OpInfo, is_input: bool) -> bool:
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


def check_tpl_kernel_type(kernel_type_name):
    tpl_kernel_type_name = KERNEL_TYPE_STR_TO_TPL.get(kernel_type_name, "UNKNOWN_KERNEL_TYPE")
    if CommonUtility.is_v220() or CommonUtility.is_c310() or CommonUtility.is_310r6():
        if kernel_type_name in STR_TO_KERNEL_TYPE_V220.keys():
            return STR_TO_KERNEL_TYPE_V220[kernel_type_name]
        else:
            raise Exception(f"The current kernel type: {tpl_kernel_type_name} is not supported in current core version")
    elif CommonUtility.is_v200():
        if kernel_type_name in STR_TO_KERNEL_TYPE_V200.keys():
            return STR_TO_KERNEL_TYPE_V200[kernel_type_name]
        else:
            raise Exception(f"The current kernel type: {tpl_kernel_type_name} is not supported in current core version")
    else:
        raise Exception(f"The TPL kernel type set is not supported in current core version.")
    return None


def tpl_tilingkey_kernel_type_check(
    tiling_key_list, decode_tiling_result, tiling_key_kernel_type
):
    tpl_set_kernel_type_cnt = 0
    for k in decode_tiling_result.keys():
        internal_dict = decode_tiling_result[k]
        if "kernelType" in internal_dict:
            tpl_set_kernel_type_cnt += 1
            tpl_kernel_type = get_kernel_meta_type(
                internal_dict["kernelType"]
            )
            if tpl_kernel_type is not None:
                kernel_type_name = KERNEL_TYPE_TO_STR.get(tpl_kernel_type, "UNKNOWN_KERNEL_TYPE")
                if kernel_type_name == "UNKNOWN_KERNEL_TYPE":
                    raise Exception(f"UNKNOWN TPL KERNEL_TYPE, value is {tpl_kernel_type}")
                else:
                    check_result = check_tpl_kernel_type(kernel_type_name)
                    if check_result is not None and check_result == tpl_kernel_type:
                        tiling_key_kernel_type[str(k)] = tpl_kernel_type
                    else:
                        raise Exception(f"check_tpl_kernel_type no pass: check_result {check_result} \
                            is no equal to {tpl_kernel_type}")
            else:
                raise Exception(f"get_kernel_meta_type return tpl_kernel_type is None, kernel_type value \
                    is {internal_dict['kernelType']}")
                    
    if tpl_set_kernel_type_cnt != 0 and tpl_set_kernel_type_cnt != len(tiling_key_list):
        raise Exception(f"All ASCENDC_TPL_ARGS_SEL must set ASCENDC_TPL_KERNEL_TYPE_SEL simultaneously!")

    return tiling_key_list, decode_tiling_result


def tpl_tilingkey_deterministic_extract(
    tiling_key_list,
    decode_tiling_result,
    tiling_key_deterministic
):
    expect_tilingkey_set = set()
    cur_deterministic_flag = get_current_build_config("enable_deterministic_mode") == 1
    deter_flag = False
    for k, v in decode_tiling_result.items():
        if "deterministic" in v:
            tiling_key_deterministic[str(k)] = v["deterministic"]
            deterministic_flag = True if v["deterministic"].lower() == "true" else False
            deter_flag = deter_flag or deterministic_flag
            if deterministic_flag == cur_deterministic_flag:
                expect_tilingkey_set.add(str(k))
    if len(tiling_key_deterministic) > 0 and deter_flag is False:
        tiling_key_deterministic.clear()
        expect_tilingkey_set.clear()
    if len(expect_tilingkey_set) > 0 and len(decode_tiling_result) > 0:
        tiling_key_list = [x for x in tiling_key_list if x in expect_tilingkey_set]
        decode_tiling_result = {
            k: v
            for k, v in decode_tiling_result.items()
            if str(k) in expect_tilingkey_set
        }
    return tiling_key_list, decode_tiling_result


def tpl_tilingkey_native_extract(tiling_key_list, decode_tiling_result, op_info):
    decl_dtype_indexes, decl_dtype_select_indexes = extract_decl_param_options(
        op_info, "dtype"
    )
    decl_format_indexes, decl_format_select_indexes = extract_decl_param_options(
        op_info, "format"
    )
    post_filter_tilingkey_list = []
    for x in tiling_key_list:
        if _filter_sel_match_by_verify_option(
            x,
            decode_tiling_result,
            decl_dtype_indexes,
            verify_params="dtypeParams",
            verify_indexes=decl_dtype_select_indexes,
        ) and _filter_sel_match_by_verify_option(
            x,
            decode_tiling_result,
            decl_format_indexes,
            verify_params="formatParams",
            verify_indexes=decl_format_select_indexes,
        ):
            post_filter_tilingkey_list.append(x)
    tiling_key_list = post_filter_tilingkey_list
    decode_tiling_result = {
        k: v for k, v in decode_tiling_result.items() if str(k) in tiling_key_list
    }
    return tiling_key_list, decode_tiling_result


def _filter_sel_match_by_verify_option(
    tiling_key: str,
    decode_tiling_map: dict,
    value_list: List[str] = None,
    verify_params: str = "dtypeParams",
    verify_indexes: List[bool] = None,
) -> bool:
    if value_list is None:
        return True
    if (
        int(tiling_key) not in decode_tiling_map
        or verify_params not in decode_tiling_map[int(tiling_key)]
        or not decode_tiling_map[int(tiling_key)][verify_params]
    ):
        return True
    target_params = value_list
    verify_params = decode_tiling_map[int(tiling_key)][verify_params]
    if verify_indexes is not None:
        verify_params = [
            verify_params[i] for i, x in enumerate(verify_indexes) if x == True
        ]
    if "unknown" in verify_params:
        CommonUtility.print_compile_log(
            "",
            f"Tiling key: '{tiling_key}' {verify_params} exist 'unknown' Params, please check it. {verify_params}",
            AscendCLogLevel.LOG_ERROR,
        )
    if len(target_params) != len(verify_params):
        CommonUtility.print_compile_log(
            "",
            f"Tiling key: '{tiling_key}' {verify_params} length do not match, "
            f"expect is {len(target_params)}, but is {len(verify_params)}",
            AscendCLogLevel.LOG_ERROR,
        )
    return target_params == verify_params