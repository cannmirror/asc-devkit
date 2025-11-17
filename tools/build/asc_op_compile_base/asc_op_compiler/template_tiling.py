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

import copy
import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Any
from asc_op_compile_base.common.utils.log_utils import AscendCLogLevel
from .ascendc_common_utility import CommonUtility

TILING_DECLARE_MAP = []
TILING_SELECT_MAP = []
ONE_BYTE_BIT = 8

UI_RANGE = 0
UI_LIST = 1
UI_MIX = 2


def extract_num(s):
    numbers = re.findall(r'\d+', s)
    return [int(num) for num in numbers]


def extract_str(s):
    strs = re.findall(r'\w+', s)
    return strs


class TilingParamType(Enum):
    TPL_DTYPE = auto()
    TPL_FORMAT = auto()
    TPL_UINT = auto()
    TPL_BOOL = auto()
    TPL_TILING_STRUCT = auto()
    TPL_KERNEL_TYPE = auto()
    TPL_DETERMINISTIC = auto()
    TPL_NONE = auto()



@dataclass
class TilingTemplateParams:
    name: str
    param_type: TilingParamType
    values: List[Any]
    bit_width: int = 0
    macro_type: str = ""

    def __post_init__(self):
        if self.param_type == TilingParamType.TPL_UINT:
            if len(self.values) < 2:
                raise RuntimeError("Length of ASCENDC_TPL_UINT_{} {} is too short.".format(self.macro_type, self.name))
            if self.bit_width == 0:
                self.bit_width = self.values[0]
                uint_declare_flag = self.values[1]
                vals = self.values[2:]
            else:
                uint_declare_flag = self.values[0]
                vals = self.values[1:]
            if uint_declare_flag == UI_LIST:
                self.values = vals
                self.__check_valid()
                return
            if uint_declare_flag != UI_RANGE and uint_declare_flag != UI_MIX:
                raise RuntimeError("Cannot find flag in ASCENDC_TPL_UINT_{} {}! Value should be in"
                " [ASCENDC_TPL_UI_RANGE, ASCENDC_TPL_UI_LIST, ASCENDC_TPL_UI_MIX].".format(self.macro_type, self.name))
            range_num = vals[0]
            extend_num = []
            for i in range(range_num):
                if 1 + i * 2 >= len(vals) - 1:
                    raise RuntimeError("UI_RANGE declare parse failed, "
                        "because the announced length of ASCENDC_TPL_UINT_{} {}"
                        " is greater than actual values' length.".format(self.macro_type, self.name))
                extend_num.extend(range(vals[1 + 2 * i], vals[2 + i * 2] + 1))
            if uint_declare_flag == UI_MIX:
                extend_num.extend(vals[1 + range_num * 2:])
            self.values = extend_num
        self.__check_valid()

    def __check_valid(self):
        if not self.values:
            raise RuntimeError("values of ASCENDC_TPL_{}_{} {} is empty!".format(self.get_param_type_str(),
                self.macro_type, self.name))
        if len(set(self.values)) != len(self.values):
            raise RuntimeError("There is duplicated number in ASCENDC_TPL_{}_{} {}!"
                " Duplicated List: {}.".format(self.macro_type, self.get_param_type_str(), self.name, self.values))
        if self.param_type == TilingParamType.TPL_BOOL:
            if not set(self.values).issubset({0, 1}):
                raise RuntimeError("There is invalid number in ASCENDC_TPL_BOOL_{} {}!"
                    " Value should only be in [0, 1].".format(self.macro_type, self.name))
        elif self.param_type == TilingParamType.TPL_DETERMINISTIC:
            if not set(self.values).issubset({'true', 'false'}):
                raise RuntimeError("There is invalid number in ASCENDC_TPL_DETERMINISTIC_SEL!"
                    "Value should only be in [0, 1, true, false].")
        else:
            if self.bit_width <= 0:
                raise RuntimeError("Bit width in ASCENDC_TPL_{}_{} {}"
                " cannot be less than zero!".format(self.get_param_type_str(), self.macro_type, self.name))
            if 2 ** self.bit_width < len(self.values):
                raise RuntimeError("Bit width:{} in ASCENDC_TPL_{}_{} {} is not enough to represent all values: {}!"\
                    " Please make sure 2^bitWidth is greater than or equal to the number of values."
                    .format(self.bit_width, self.get_param_type_str(), self.macro_type, self.name, self.values))

    def get_encodes(self):
        encodes = dict()
        if self.param_type == TilingParamType.TPL_UINT:
            for i, value in enumerate(self.values):
                encodes[bin(i)[2:].rjust(self.bit_width, '0')] = value
        elif self.param_type == TilingParamType.TPL_DTYPE or \
            self.param_type == TilingParamType.TPL_FORMAT:
            for value in self.values:
                encodes[bin(value)[2:].rjust(self.bit_width, '0')] = value
        else:
            encodes['0'] = 0
            encodes['1'] = 1
        return encodes

    def get_param_type_str(self) -> str:
        name_dict = {TilingParamType.TPL_DTYPE: "DTYPE",
                    TilingParamType.TPL_BOOL: "BOOL",
                    TilingParamType.TPL_FORMAT: "FORMAT",
                    TilingParamType.TPL_UINT: "UINT",
                    TilingParamType.TPL_DETERMINISTIC: "DETERMINISTIC",
                    TilingParamType.TPL_NONE: ""
        }
        return name_dict.get(self.param_type, "")


def extract_template_tiling_info(tiling_declare_str: str, tiling_select_str: str) -> None:
    global TILING_DECLARE_MAP
    global TILING_SELECT_MAP
    declare_param_list = tiling_declare_str.split("@@")
    select_param_list = tiling_select_str.split("@@")
    TILING_DECLARE_MAP = extract_template_tiling_params(declare_param_list)[0]
    bit_map = {}
    for declare_param in TILING_DECLARE_MAP:
        bit_map[declare_param.name] = declare_param.bit_width
    TILING_SELECT_MAP = extract_template_tiling_params(select_param_list, bit_map)
    name_order = [declare_param.name for declare_param in TILING_DECLARE_MAP]
    for tiling_param_list in TILING_SELECT_MAP:
        check_valid_select_param(tiling_param_list, name_order)

    for select_params in TILING_SELECT_MAP:
        select_params.sort(key=lambda x: name_order.index(x.name) if x.name in name_order else -1)


def check_valid_select_param(tiling_param_list: List[TilingTemplateParams], name_order: List[str]) -> None:
    tiling_param_name_list = [tiling_param.name for tiling_param in tiling_param_list if
                              tiling_param.param_type != TilingParamType.TPL_TILING_STRUCT]
    for tiling_declare_param in TILING_DECLARE_MAP:
        if tiling_declare_param.name not in tiling_param_name_list:
            raise RuntimeError("There is missing marco define: {} in ASCENDC_TPL_{}_SEL.".format(
                tiling_declare_param.name, tiling_declare_param.get_param_type_str()))
    total_bit_length = 0
    for tiling_param in tiling_param_list:
        if tiling_param.param_type == TilingParamType.TPL_TILING_STRUCT or \
            tiling_param.param_type == TilingParamType.TPL_KERNEL_TYPE or \
            tiling_param.param_type == TilingParamType.TPL_DETERMINISTIC:
            continue
        name = tiling_param.name
        total_bit_length += tiling_param.bit_width
        if total_bit_length > 64:
            raise RuntimeError(f"name:{tiling_param.name}, type:{tiling_param.get_param_type_str()}, \
Total bit width cannot be greater than 64!")
        tpl_type = tiling_param.param_type
        vals = tiling_param.values
        matched_param = None
        for param in TILING_DECLARE_MAP:
            if param.name == name:
                matched_param = param
                break
        if matched_param is None:
            raise RuntimeError(
                "Cannot find ASCENDC_TPL_{}_SEL name: {} in ASCENDC_TPL_{}_DECL.".format(
                tiling_param.get_param_type_str(), name, tiling_param.get_param_type_str()))
        if tpl_type != matched_param.param_type:
            raise RuntimeError(
                "{} has different type in ASCENDC_TPL_{}_SEL and ASCENDC_TPL_{}_DECL.".format(name,
                tiling_param.get_param_type_str(), matched_param.get_param_type_str()))
        if not set(vals).issubset(set(matched_param.values)):
            raise RuntimeError(
                "Cannot find value {} from ASCENDC_TPL_{}_SEL {} in ASCENDC_TPL_{}_DECL,"
                " please check your macro define.".format(set(vals) - set(matched_param.values),
                tiling_param.get_param_type_str(),
                name, tiling_param.get_param_type_str()))


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def extract_template_tiling_params(tiling_param_list: List[str], bit_map: dict = None) -> list:
    res = []
    tiling_param = []
    for i, _ in enumerate(tiling_param_list):
        sub_str = tiling_param_list[i]
        if not sub_str:
            continue
        if sub_str == '{' and tiling_param:
            res.append(tiling_param)
            tiling_param = []
        macro_type = "SEL" if sub_str.startswith("ASCENDC_TPL_DTYPE_SEL_") or \
            sub_str.startswith("ASCENDC_TPL_FORMAT_SEL_") or \
            sub_str.startswith("ASCENDC_TPL_UINT_SEL_") or \
            sub_str.startswith("ASCENDC_TPL_BOOL_SEL_") or \
            sub_str.startswith("ASCENDC_TPL_TILING_STRUCT_SEL_") or \
            sub_str.startswith("ASCENDC_TPL_KERNEL_TYPE_SEL") or \
            sub_str.startswith("ASCENDC_TPL_DETERMINISTIC_SEL") else \
            "DECL"
        if sub_str.startswith("ASCENDC_TPL_DTYPE"):
            dtype_list = extract_num(tiling_param_list[i + 1])
            name = remove_prefix(sub_str, 'ASCENDC_TPL_DTYPE_{}_'.format(macro_type))
            tiling_param.append(TilingTemplateParams(
                name, TilingParamType.TPL_DTYPE, dtype_list, 8, macro_type))
        elif sub_str.startswith("ASCENDC_TPL_FORMAT"):
            format_list = extract_num(tiling_param_list[i + 1])
            name = remove_prefix(sub_str, 'ASCENDC_TPL_FORMAT_{}_'.format(macro_type))
            tiling_param.append(TilingTemplateParams(
                name, TilingParamType.TPL_FORMAT, format_list, 8, macro_type))
        elif sub_str.startswith("ASCENDC_TPL_KERNEL_TYPE_SEL"):
            format_list = extract_num(tiling_param_list[i + 1])
            name = "KERNEL_TYPE"
            tiling_param.append(TilingTemplateParams(
                name, TilingParamType.TPL_KERNEL_TYPE, format_list, 8, macro_type))
        elif sub_str.startswith("ASCENDC_TPL_DETERMINISTIC_SEL"):
            cur_deter_flag = extract_str(tiling_param_list[i + 1])
            if len(cur_deter_flag) != 1:
                raise RuntimeError("ASCENDC_TPL_DETERMINISTIC_SEL can only one value can be specified")
            if cur_deter_flag[0] == "1":
                cur_deter_flag = ["true"]
            if cur_deter_flag[0] == "0":
                cur_deter_flag = ["false"]
            tiling_param.append(TilingTemplateParams("DETERMINISTIC", \
                TilingParamType.TPL_DETERMINISTIC, cur_deter_flag, 1, macro_type))
        elif sub_str.startswith("ASCENDC_TPL_UINT"):
            uint_list = extract_num(tiling_param_list[i + 1])
            name = remove_prefix(sub_str, 'ASCENDC_TPL_UINT_{}_'.format(macro_type))
            if macro_type == "SEL":
                tiling_param.append(TilingTemplateParams(name, \
                    TilingParamType.TPL_UINT, uint_list, bit_map[name], macro_type))
            else:
                tiling_param.append(TilingTemplateParams(name, TilingParamType.TPL_UINT, uint_list, 0, macro_type))
        elif "ASCENDC_TPL_BOOL" in sub_str:
            bool_list = extract_num(tiling_param_list[i + 1])
            name = remove_prefix(sub_str, 'ASCENDC_TPL_BOOL_{}_'.format(macro_type))
            tiling_param.append(TilingTemplateParams(name, TilingParamType.TPL_BOOL, bool_list, 1, macro_type))
        elif "ASCENDC_TPL_TILING_STRUCT" in sub_str:
            # 用-1占位
            name = remove_prefix(sub_str, 'ASCENDC_TPL_TILING_STRUCT_{}_'.format(macro_type))
            tiling_param.append(TilingTemplateParams(name, TilingParamType.TPL_TILING_STRUCT, [-1], 1))
    if tiling_param:
        res.append(tiling_param)
    return res


def get_concated_tiling_key(template_param_list: List[TilingTemplateParams], \
    index: int, tiling_key: str, tiling_args: List[int], encode_book: dict, data: dict) -> dict:
    result = dict()
    if index == len(template_param_list):
        data.update({"paramArgs": tiling_args})
        result[int(tiling_key, 2)] = copy.copy(data)
        return result
    template_param = template_param_list[index]
    for val in template_param.values:
        if template_param.param_type == TilingParamType.TPL_TILING_STRUCT:
            data["tilingStruct"] = template_param.name
            encode_ = ""
            tiling_args_temp = tiling_args
        elif template_param.param_type == TilingParamType.TPL_KERNEL_TYPE:
            data["kernelType"] = template_param.values[0]
            encode_ = ""
            tiling_args_temp = tiling_args
        elif template_param.param_type == TilingParamType.TPL_DETERMINISTIC:
            data['deterministic'] = template_param.values[0]
            encode_ = ""
            tiling_args_temp = tiling_args
        else:
            encode_ = encode_book[template_param.name][val]
            tiling_args_temp = tiling_args + [val]
        result.update(get_concated_tiling_key(template_param_list, index + 1, encode_ + tiling_key, tiling_args_temp,
                                              encode_book, data))
    return result


def decode_tiling(tiling_key: int = None) -> dict:
    encode_book = dict()
    for param in TILING_DECLARE_MAP:
        encode_book[param.name] = param.get_encodes()
    reversed_encode_book = {}
    CommonUtility.print_compile_log("", "[Template Tiling] Encode Book is {}.".format(encode_book), \
        AscendCLogLevel.LOG_INFO)
    for k, v in encode_book.items():
        reversed_book = dict()
        for bin_encodes, actual_val in v.items():
            reversed_book[actual_val] = bin_encodes
        reversed_encode_book[k] = reversed_book
    decode_results = dict()
    for tiling_template_select_param in TILING_SELECT_MAP:
        decode_result = get_concated_tiling_key(
            tiling_template_select_param, 0, '', [], reversed_encode_book, {})
        length_before = len(decode_result) + len(decode_results)
        decode_results.update(decode_result)
        if length_before != len(decode_results):
            raise RuntimeError("ASCENDC_TPL_SELECT has duplicated definitions!")
    if tiling_key is None:
        return decode_results
    if tiling_key not in decode_results.keys():
        CommonUtility.print_compile_log("", "Cannot find any matched template tiling keys.", \
            AscendCLogLevel.LOG_WARNING)
        return {}
    else:
        return {tiling_key: decode_results[tiling_key]}
