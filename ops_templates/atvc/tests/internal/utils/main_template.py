#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
ELEWISE_MAIN_CODE = """
#include "acl/acl.h"
#include <iostream>
#include "data_utils.h"
#include "elewise/elewise_host.h"

void {kernel_func}(uint32_t blockDim, void* stream, {uint8_param});
{op_traits};
int32_t main(int32_t argc, char* argv[])
{{
    int32_t eleNum = std::stoi(std::string(argv[1]));
    bool enableProf = std::string(argv[2]) == "1";
{scalar_define_params}
{declare_input_shape}
{declare_output_shape}

    ATVC::EleWiseParam param;
    if (!ATVC::Host::CalcEleWiseTiling<OP_TRAITS>(eleNum, param)) {{
        printf("EleWise tiling error.");
        return -1;
    }}
    CHECK_ACL(aclInit({{}}));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    uint8_t* paramDevice;
{acl_calls}
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return 0;
}}
"""


device_type_2_host = {
    "half": "int16_t",
    "bfloat16_t": "int16_t",
    "float16_t": "int16_t"
}


def build_op_traits(test_case_info):
    op_traits = "using OP_TRAITS = ATVC::OpTraits<ATVC::OpInputs<{}>, ATVC::OpOutputs<{}>, ATVC::OpTemps<{}>>;"
    input_types = ",".join([device_type_2_host.get(input_info["dtype"], input_info["dtype"]) \
        for input_info in test_case_info.get("inputs", [])])
    output_types = ",".join([device_type_2_host.get(output_info["dtype"], output_info["dtype"])\
        for output_info in test_case_info.get("outputs", [])])
    temp_types = ",".join([i for i in test_case_info.get("op_temps", [])])
    return op_traits.format(input_types, output_types, temp_types)


def build_acl_api(case_name, test_case_info):
    """
    构建ACL API调用的函数

    参数:
    case_name: 测试用例的名称
    test_case_info: 测试用例的信息，包含输入、输出、形状等信息

    返回值:
    acl_calls: 构建的ACL API调用的字符串
    """
    # 提取输入和输出信息
    input_names = [inp["name"] for inp in test_case_info["inputs"]]
    output_names = [out["name"] for out in test_case_info["outputs"]]
    non_inplace_outputs = [out["name"] for out in test_case_info["outputs"] 
                          if out["name"] not in input_names]

    # 生成输入初始化代码
    input_initialization = '\n'.join([
        f'\tCHECK_ACL(aclrtMallocHost((void**)(&{name}Host), {name}ByteSize));\n'\
        f'\tCHECK_ACL(aclrtMalloc((void**)&{name}Device, {name}ByteSize, ACL_MEM_MALLOC_HUGE_FIRST));\n'\
        f'\tReadFile("./input/input_{name}.bin", {name}ByteSize, {name}Host, {name}ByteSize);\n'\
        f'\tCHECK_ACL(aclrtMemcpy({name}Device, {name}ByteSize, {name}Host,'
        f' {name}ByteSize, ACL_MEMCPY_HOST_TO_DEVICE));'\
            .format(name)
         for name in input_names])


    # 生成输出初始化代码
    output_initialization = '\n'.join([
        f'\tCHECK_ACL(aclrtMallocHost((void**)(&{name}Host), {name}ByteSize));\n'\
        f'\tCHECK_ACL(aclrtMalloc((void**)&{name}Device, {name}ByteSize, ACL_MEM_MALLOC_HUGE_FIRST));\n'.format(name)
         for name in non_inplace_outputs]) if non_inplace_outputs else ''


    # 生成kernel调用参数
    tensor_ptrs = [f'{name}Device' for name in input_names + non_inplace_outputs]
    scalar_args = [f'{scalar["name"]}' for scalar in test_case_info.get("scalars", [])]
    kernel_args = f'{", ".join(tensor_ptrs)}, paramDevice'
    if scalar_args:
        kernel_args += f', {", ".join(scalar_args)}'

    # 生成输出处理代码
    output_handling = '\n'.join([f'\tCHECK_ACL(aclrtMemcpy({name}Host, {name}ByteSize, {name}Device, {name}ByteSize, '\
                                 f'ACL_MEMCPY_DEVICE_TO_HOST));\n'\
                                 f'\tWriteFile("./output/output_{name}.bin", {name}Host, {name}ByteSize);'.format(name)
                                 for name in output_names])

    # 生成内存释放代码
    memory_release = '\n'.join([f'\tCHECK_ACL(aclrtFree({name}Device));\n'\
        f'\tCHECK_ACL(aclrtFreeHost({name}Host));\n'.format(name) for name in input_names + non_inplace_outputs])
    memory_release += "\tCHECK_ACL(aclrtFree(paramDevice));\n"
    # 拼接所有代码部分
    acl_calls = "".join([f'\tuint8_t* {name}Device;\n' for name in input_names + non_inplace_outputs])

    acl_calls += f"{input_initialization}\n"
    if output_initialization:
        acl_calls += f"{output_initialization}\n\n"
    acl_calls += f"\tauto elementParamSize = sizeof(param);\n"\
        "\tCHECK_ACL(aclrtMalloc((void**)&paramDevice, elementParamSize, ACL_MEM_MALLOC_HUGE_FIRST));\n"\
        "\tCHECK_ACL(aclrtMemcpy(paramDevice, elementParamSize, reinterpret_cast<uint8_t*>(&param),"\
        " elementParamSize, ACL_MEMCPY_HOST_TO_DEVICE));\n"
    acl_calls += (
        "int loopCnt = 1; if(enableProf) {loopCnt = 20;};\n"\
        "for (int i = 0; i < loopCnt; ++i) {\n"\
        f"\t{test_case_info['kernel_func']}(param.tilingData.blockNum, stream, {kernel_args});\n"
        "}\n"
        "\tCHECK_ACL(aclrtSynchronizeStream(stream));\n\n"
        f"{output_handling}\n\n"
        f"{memory_release}"
    )

    return acl_calls


def build_main_file(case_name, test_case_info, exec_args):
    """
    构建主函数,生成对应的main.cpp文件

    参数:
    case_name: 测试用例名称
    test_case_info: 测试用例信息，包含输入、输出、形状、标量等信息
    """
    if "reduce_dim" in test_case_info:
        build_reduce_main_file(case_name, test_case_info, exec_args)
    elif "broadcast" in test_case_info:
        build_broadcast_main_file(case_name, test_case_info, exec_args)
    else:
        build_elewise_main_file(case_name, test_case_info, exec_args)


def build_reduce_main_file(case_name, test_case_info, exec_args):
    import os
    from .common import run_cmds
    base_path = os.path.dirname(os.path.abspath(__file__))
    run_cmds(" ".join(["cp", f"{base_path}/reduce_main.cpp", "./" + case_name + "/main.cpp"]))


def build_broadcast_main_file(case_name, test_case_info, exec_args):
    import os
    from .common import run_cmds
    base_path = os.path.dirname(os.path.abspath(__file__))
    run_cmds(" ".join(["cp", f"{base_path}/broadcast_main.cpp", "./" + case_name + "/main.cpp"]))


def build_elewise_main_file(case_name, test_case_info, exec_args):
    uint8_param = ""
    # 获取输入和输出的名称
    input_names = [input_info["name"]
                   for input_info in test_case_info["inputs"]]
    output_names = [output_info["name"]
                    for output_info in test_case_info["outputs"]]
    # 遍历输入和输出的名称，构建相关参数
    for i in set(input_names + output_names):
        uint8_param += " uint8_t* {}, ".format(i)
    uint8_param += "uint8_t* param"
    exclude_output_names = [output_name for output_name in output_names if output_name not in input_names]

    # 初始化声明输入输出大小的字符串
    declare_input_shape = ""
    declare_output_shape = ""
    # 构建标量参数的定义
    scalar_define_params = "".join(["\t{} {} = {};\n".format(scalar["dtype"], scalar["name"], scalar["value"])
                                    for scalar in test_case_info.get("scalars", [])])
    # 构建标量参数的声明
    scalar_declare_param = "".join([", {} {}".format(scalar["dtype"], scalar["name"])
                                    for scalar in test_case_info.get("scalars", [])])
    uint8_param += scalar_declare_param
    # 遍历输入信息，构建相关参数
    declare_input_shape = "".join([f'\tuint8_t* {name}Host;\n' for name in input_names + exclude_output_names])
    for input_info in test_case_info.get("inputs", []):
        declare_input_shape += "\tsize_t {}ByteSize "\
            "= eleNum * sizeof({});\n".format(input_info["name"],
                                                    device_type_2_host.get(input_info["dtype"], input_info["dtype"]))

    # 遍历输出信息，构建相关参数
    for output_info in test_case_info.get("outputs", []):
        if output_info["name"] in exclude_output_names:
            declare_output_shape += "\tsize_t {}ByteSize = eleNum * sizeof({});\n".\
                    format(output_info["name"],
                   device_type_2_host.get(output_info["dtype"], output_info["dtype"]))
    # 构建ACL API
    acl_src_line = build_acl_api(case_name, test_case_info)

    # 写入main.cpp文件
    with open("./{}/main.cpp".format(case_name), mode="w") as f:
        f.write(ELEWISE_MAIN_CODE.format(kernel_func=test_case_info["kernel_func"],
                                    uint8_param=uint8_param, scalar_define_params=scalar_define_params,
                                    declare_input_shape=declare_input_shape,
                                    declare_output_shape=declare_output_shape,
                                    op_traits=build_op_traits(test_case_info),
                                    acl_calls=acl_src_line))