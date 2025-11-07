#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import sys
import os
import subprocess
import re


def get_file_content(file):
    data = []
    with open(file, 'rb') as fd:
        buf = fd.read()
        start = 0
        while start + 8 <= len(buf):
            value = int.from_bytes(buf[start : start + 8], byteorder='little')
            data.append(f'0x{value:016x}')
            start += 8
    return data


def get_kernel_funcs(file):
    elftool = os.path.join(os.environ.get('ASCEND_HOME_PATH'), 'toolkit/toolchain/hcc/bin', \
        'aarch64-target-linux-gnu-readelf')
    command = [elftool, '--dyn-syms', '-W', file]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'get_kernel_funcs errno {result.returncode}: {result.stderr}')
    funcs = []
    for line in result.stdout.splitlines():
        sym_infos = line.split()
        if len(sym_infos) <= 7:
            continue
        is_valid_symbol = (
            sym_infos[2].isnumeric() and int(sym_infos[2]) > 0
            and sym_infos[3] == 'FUNC'
            and sym_infos[5] == 'DEFAULT'
        )
        is_valid_binding = (sym_infos[4] == 'GLOBAL' or sym_infos[4] == 'WEAK')
        if is_valid_symbol and is_valid_binding:
            funcs.append(sym_infos[7])
    return funcs


def gen_code(so_file, src):
    data = get_file_content(so_file)
    src_name = os.path.splitext(os.path.basename(src))[0]

    code = '''
#include <acl/acl.h>
#include <cstring>
#include <mutex>
#include <thread>

#ifdef __cplusplus
extern "C" {
#endif
aclrtBinHandle AicpuLoadBinaryFromBuffer(const unsigned long *aicpuFileBuf, size_t fileSize);
aclrtFuncHandle AicpuRegFunctionByName(const aclrtBinHandle binHandle, const char *funcName);
void AicpuLaunchKernel(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream, void *arg, size_t argSize);

'''
    code += f'static const unsigned long __aicpu_file_buf__[{len(data)}] '\
            f'__attribute__ ((section (".ascend.kernel.aicpu"))) = {{\n'
    start = 0
    while start + 8 <= len(data):
        code += f'    ' + ', '.join(data[start : start + 8]) + ',\n'
        start += 8
    if start != len(data):
        code += f'    ' + ', '.join(data[start : len(data)]) + ',\n'
    code += '};\n'
    code += f'''
static __thread aclrtBinHandle __aicpu_bin_handle__;

static void *__aicpu_get_bin_hdl_{src_name}__(void)
{{
  if (__aicpu_bin_handle__ == nullptr) {{
    __aicpu_bin_handle__ = AicpuLoadBinaryFromBuffer(__aicpu_file_buf__, sizeof(__aicpu_file_buf__));
  }}
  return (void *)__aicpu_bin_handle__;
}}
'''
    funcs = get_kernel_funcs(so_file)
    # gen c style handle get function
    for func in funcs:
        code += f'\n__attribute__((alias("__aicpu_get_bin_hdl_{src_name}__")))\n'
        code += f'static void *__aicpu_get_func_bin_hdl_{func}__(void);\n'
    code += '\n'
    # c++ style function instances
    for func in funcs:
        # weak function for kernel stub
        code += '__attribute__((weak))\n'
        code += f'void {func}(uint32_t block_dim, void *stream, void *arg, size_t arg_size) {{\n'
        code += f'''
    static thread_local aclrtBinHandle bin_handle = (aclrtBinHandle)__aicpu_get_func_bin_hdl_{func}__();
    static thread_local aclrtFuncHandle fun_handle = AicpuRegFunctionByName(bin_handle, \"{func}\");
    AicpuLaunchKernel(fun_handle, block_dim, stream, arg, arg_size);
}}
'''
    code += '''
#ifdef __cplusplus
}
#endif
'''
    with open(src, 'w+') as fd:
        fd.write(code)


if __name__ == '__main__':
    so_file = sys.argv[1]
    src_file = sys.argv[2]
    gen_code(so_file, src_file)
