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
ascendc compile base
"""
import os
import stat
import shutil
import subprocess
import multiprocessing
import re
from collections import namedtuple
from dataclasses import dataclass
from asc_op_compile_base.asc_op_compiler import cce_runtime
from asc_op_compile_base.common.buildcfg import get_current_build_config
from asc_op_compile_base.common.error_mgr import raise_tbe_python_err, TBE_DEFAULT_PYTHON_ERROR_CODE
from asc_op_compile_base.common.ccec import CCECInfo
from .ascendc_common_utility import CommonUtility, write_mk, is_enable_ascendc_cov, \
    is_enable_build_log, is_enable_sanitizer
from .global_storage import global_var_storage
from asc_op_compile_base.common.utils.log_utils import AscendCLogLevel, CompileStage
from .get_op_tiling import OpInfo
from .ascendc_constants import KernelMetaType, CORE_TYPE_MIX, CORE_TYPE_CUBE, CORE_TYPE_VEC


def compile_pre_process(op_info: OpInfo, compile_options: list):
    cce_runtime.TBE_WORKSPACE_SIZE_LIST.local_list = []
    cce_runtime.TBE_WORKSPACE_IND_LIST.local_list = []
    cce_runtime.MULTI_CORE_SYNC_WORKSPACE_SIZE_LIST.local_list = []
    cce_runtime.TBE_ATUO_ATOMIC_IND_LIST.local_list = []
    CommonUtility.get_ascendc_compiler_path()
    if global_var_storage.get_variable("ascendc_enable_ccache") == True:
        CommonUtility.remove_options(compile_options, ["-x", "cce"])
        compile_options.append("--cce-aicore-lang")
    from asc_op_compile_base.common.buildcfg.buildcfg_mapping import op_debug_config
    op_debug_config_val = get_current_build_config(op_debug_config)
    compile_options.append("--cce-disable-kernel-global-attr-check")
    global_var_storage.set_variable("ascendc_enable_super_kernel", False)
    global_var_storage.set_variable("ascendc_sub_super_kernel_params", "")
    global_var_storage.set_variable("ascendc_sub_super_kernel_type", "")
    global_var_storage.set_variable("ascendc_sub_super_kernel_fun_names", {})
    global_var_storage.set_variable("ascendc_compile_debug_config", "dump_cce" in op_debug_config_val)
    global_var_storage.set_variable("ascendc_dump_disable_compile_options", "-DASCENDC_DUMP=0" in compile_options)
    global_var_storage.set_variable("ascendc_debug_compile_options", "-DASCENDC_DEBUG" in compile_options)
    global_var_storage.set_variable("ascendc_enable_sanitizer", is_enable_sanitizer(compile_options))
    global_var_storage.set_variable("ascendc_enable_build_log", is_enable_build_log())
    global_var_storage.set_variable("ascendc_enable_coverage", is_enable_ascendc_cov())
    global_var_storage.set_variable("ascendc_time_stamp_compile_options", "-DASCENDC_TIME_STAMP_ON" in compile_options)
    global_var_storage.set_variable("ascendc_enable_super_kernel", \
      (bool(get_current_build_config("enable_super_kernel")) and CommonUtility.is_support_super_kernel()))
    global_var_storage.set_variable(
        "ascendc_enable_aicore_exception_restart", "-DAICORE_EXCEPTION_RESTART" in compile_options)
    if global_var_storage.get_variable("ascendc_enable_coverage"):
        compile_options.append("-g")
    return compile_options


def get_actual_kernel_type(tiling_key, compile_info, need_ffts, kernel_name):
    code_type = compile_info.code_channel
    default_kernel_type = compile_info.default_kernel_type
    kernel_type = compile_info.tiling_key_kernel_type[
        tiling_key] if tiling_key in compile_info.tiling_key_kernel_type else default_kernel_type
    if kernel_type in [KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0]:
        return CORE_TYPE_CUBE
    elif kernel_type in [KernelMetaType.KERNEL_TYPE_MIX_AIV_1_0]:
        return CORE_TYPE_VEC
    elif kernel_type in [KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2]:
        return CORE_TYPE_MIX
    if compile_info.no_set_kernel_type and need_ffts:
        return code_type
    else:
        CommonUtility.print_compile_log(
            kernel_name, "Aicore Exception Restart not support this kernel type", AscendCLogLevel.LOG_ERROR)
        raise Exception(f"Aicore Exception Restart not support this kernel type")


SingleTilingKeyCompileParams = namedtuple('SingleTilingKeyCompileParams', \
    ['tiling_key', 'compile_info', 'sub_arch', 'tiling_info', 'code_channel', 'compile_option_tuple'])


def fatbin_objs(obj_files: list, dst_file: str, is_debug: bool, compile_log_path=None):
    if global_var_storage.get_variable("ascendc_enable_super_kernel") is True and \
        global_var_storage.get_variable("ascendc_is_static_op"):
        return
    compile_cmd = [CCECInfo.get_exe("ld.lld"), '-m', 'aicorelinux', '-r', '-Ttext=0', '-q']
    if not is_debug:
        compile_cmd.append('-x')
    for obj in obj_files:
        compile_cmd += [obj]
    compile_cmd += ['-static', '-o', "%s" % dst_file]
    CommonUtility.run_cmd_inner(compile_cmd, CompileStage.FATBIN, compile_log_path)
    if not global_var_storage.get_variable("ascendc_compile_debug_config") and \
        not global_var_storage.get_variable("super_kenel_save_sub_op_files"):
        for obj in obj_files:
            os.remove(obj)


def link_relocatable(bin_file_path, compile_log_path=None):
    short_soc_version = global_var_storage.get_variable("ascendc_short_soc_version")
    if short_soc_version == "Ascend310B":
        link_cmd = [CCECInfo.get_exe("ld.lld"),
                    "-m",
                    "aicorelinux",
                    "-Ttext=0",
                    "%s" % bin_file_path,
                    "-static",
                    "-o",
                    "%s" % bin_file_path,
                    '-q',
                    ]
    else:
        link_cmd = [CCECInfo.get_exe("ld.lld"),
                    "-m",
                    "aicorelinux",
                    "-Ttext=0",
                    "%s" % bin_file_path,
                    "-static",
                    "-o",
                    "%s" % bin_file_path,
                    '-q',
                    ]
    CommonUtility.run_cmd_inner(link_cmd, CompileStage.LINKRELOCATE, compile_log_path)


def link_relocatable_meta_file(bin_file_path, meta_file_path, compile_log_path=None):
    link_cmd = [CCECInfo.get_exe("ld.lld"),
                "-m",
                "aicorelinux",
                "-Ttext=0",
                "%s" % bin_file_path,
                "%s" % meta_file_path,
                "-static",
                "-o",
                "%s" % bin_file_path,
                '-q',
                ]
    CommonUtility.run_cmd_inner(link_cmd, CompileStage.LINKRELOCATE, compile_log_path)


def compile_multi_tilingkey(tiling_key_list, cmds_list, dstfile_name, compile_log_path):
    parallel_compile_check = os.getenv('TILINGKEY_PAR_COMPILE')
    if parallel_compile_check not in [None, '1', '0']:
        CommonUtility.print_compile_log("", "TILINGKEY_PAR_COMPILE ONLY SUPPORT 0 OR 1, current \
TILINGKEY_PAR_COMPILE is {}".format(parallel_compile_check), AscendCLogLevel.LOG_WARNING)
    ci_big_makefile_par_switch = os.getenv('TILINGKEY_PAR_COMPILE') == '1'
    ascendc_self_par_job = os.getenv('ASCENDC_PAR_COMPILE_JOB')
    ascendc_self_par_job_num = 0
    if ascendc_self_par_job is not None:
        if ascendc_self_par_job == '1':
            cpu_db_num = 2 * multiprocessing.cpu_count() - 2
            ascendc_self_par_job_num = cpu_db_num if cpu_db_num > 0 else 1
        else:
            ascendc_self_par_job_num = int(ascendc_self_par_job)

    if ci_big_makefile_par_switch or ascendc_self_par_job_num > 0:
        dstfile_with_pid = dstfile_name + str(os.getpid())
        write_mk(tiling_key_list, cmds_list, dstfile_with_pid, compile_log_path)
        # when TILINGKEY_PARALLEL_COMPILATION_SWITCH and ASCENDC_PAR_COMPILE_JOB conflicts
        # TILINGKEY_PARALLEL_COMPILATION_SWITCH first
        mk_file = f'{dstfile_with_pid}.mk'
        if ci_big_makefile_par_switch:
            cmd = ['make', '-f', mk_file]
        else:
            cmd = ['make', '-f', mk_file, '-j', f'{ascendc_self_par_job_num}']
        cmd_str = ' '.join(cmd)
        file_name = ""
        if global_var_storage.get_variable("ascendc_enable_build_log") is True:
            file_name, kernel_name, hash_name = CommonUtility.get_build_file_name(cmds_list[0], CompileStage.COMPILE)
            try:
                with open(file_name, mode="at") as f:
                    os.chmod(file_name, stat.S_IRUSR + stat.S_IWUSR)
                    f.write("%s\n" % (cmd_str))
                    cmd.append('2>&1')
                    cmd.append('|')
                    cmd.append('tee -a')
                    cmd.append(file_name)
                    cmd_str = ' '.join(cmd)
            except Exception as err:
                raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, ("write log failed, reason is:", err))
        ret = os.system(f'{cmd_str} > /dev/null')
        if ret != 0 and global_var_storage.get_variable("ascendc_enable_build_log") is True:
            file_name_parts = file_name.split('.')
            new_file_name = file_name_parts[0] + "_error." + file_name_parts[-1]
            os.rename(file_name, new_file_name)
            CommonUtility.print_compile_log("", "Operator {}_{}: errors occurred during compile phase \
of {}, See also {}".format(kernel_name, hash_name, \
str(CompileStage.COMPILE), new_file_name), AscendCLogLevel.LOG_ERROR)
            raise Exception("An error occurred during compile phases of {}".format(str(CompileStage.COMPILE)))
        if not global_var_storage.get_variable("ascendc_compile_debug_config"):
            CommonUtility.remove_temp_file(mk_file)
    else:
        for cmds in cmds_list:
            CommonUtility.run_cmd_inner(cmds, CompileStage.COMPILE, compile_log_path)


def search_in_line(line, keywords):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b')
    matches = pattern.findall(line)
    if matches:
        return True, f"{', '.join(matches)}"
    return False, ""


def extract_file_path(line):
    pattern = re.compile(r'"([^"]+)"')
    matches = pattern.findall(line)
    return matches[0]