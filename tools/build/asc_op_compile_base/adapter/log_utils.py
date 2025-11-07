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
AscendC compile log
"""
import os
import stat
import threading
import time
import sys
import inspect
from enum import Enum
from types import MappingProxyType
from datetime import datetime, timezone
from .global_storage import global_var_storage
from tbe.common.platform.platform_info import set_current_compile_soc_info
from tbe.tvm.error_mgr import raise_tbe_python_err, TBE_DEFAULT_PYTHON_ERROR_CODE
from tbe.common.repository_manager.utils.repository_manager_log import LOG_INSTANCE


COMPILE_STAGE_MSG_INFO = {
    "compile_op_start": "compile op start",
    "preprocess_start": "preprocess start",
    "preprocess_end": "preprocess end",
    "generate_tiling_start": "generate tiling start",
    "generate_tiling_end": "generate tiling end",
}
COMPILE_STAGE_MSG_INFO = MappingProxyType(COMPILE_STAGE_MSG_INFO)


CompileStage = Enum('CompileStage', ('PRECOMPILE', 'INFERCHANNEL', 'DEBUG_PRECOMPILE', \
    'DEBUG_ASSEMBLE', 'COMPILE', 'FATBIN', 'LINKRELOCATE', 'SPLIT_SUB_OBJS', 'SPK_INPUT', 'PACK', 'UNPACK'))


class AscendCLogLevel(Enum):
    LOG_DEBUG = 0
    LOG_INFO = 1
    LOG_WARNING = 2
    LOG_ERROR = 3


LOG_LEVEL_TO_STR = {
    AscendCLogLevel.LOG_DEBUG : "\033[32mDEBUG\033[0m",
    AscendCLogLevel.LOG_INFO : "\033[32mINFO\033[0m",
    AscendCLogLevel.LOG_WARNING : "\033[93mWARNING\033[0m",
    AscendCLogLevel.LOG_ERROR : "\033[31mERROR\033[0m",
}


class LogUtil:
    """
    This class defines some common tool function methods.
    """
    class Option(Enum):
        DEFAULT = 0
        NON_SOC = 1

    def __init__(self):
        pass

    # write the cmpile_cmd to log
    @staticmethod
    def dump_compile_log(compile_cmd, stage: CompileStage, log_file=None):
        if log_file is None or compile_cmd is None:
            return
        flags = os.O_RDWR | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        try:
            with os.fdopen(os.open(log_file, flags, modes), 'a') as f:
                f.write(f'// Stage: {stage}\n')
                f.write(" ".join(str(cmd) for cmd in compile_cmd))
                f.write("\n\n")
        except Exception as err:
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, ("write log failed, reason:", err))

    @staticmethod
    def set_soc_version(soc_version):
        set_current_compile_soc_info(soc_version)

    @staticmethod
    def dump_log(log_str, log_file=None, level="[INFO] : "):
        if log_file is None or log_str is None:
            return
        flags = os.O_RDWR | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        try:
            with os.fdopen(os.open(log_file, flags, modes), 'a') as f:
                f.write(f'// : log:\n')
                f.write(f"{level} {log_str}")
                f.write("\n\n")
        except Exception as err:
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, ("write log failed, reason:", err))

    # print log with level judge
    @staticmethod
    def print_compile_log(kernel_name: str, msg_info: str, log_level: AscendCLogLevel, option: Option = Option.DEFAULT):
        default_log_level = AscendCLogLevel.LOG_WARNING.value
        plog_switch = os.environ.get("ASCEND_SLOG_PRINT_TO_STDOUT")
        if plog_switch is None and log_level.value < default_log_level:
            return
        if plog_switch is not None and int(plog_switch) == 0 and log_level.value < default_log_level:
            return
        plog_level = os.environ.get("ASCEND_GLOBAL_LOG_LEVEL")
        if plog_level is None and log_level.value < default_log_level:
            return
        if plog_level is not None and log_level.value < int(plog_level):
            return
        LogUtil.log_print(kernel_name, msg_info, log_level, option)
        LogUtil.plog_print(kernel_name, msg_info, log_level, option)
        return


    # print log without level judge
    @staticmethod
    def log_print(kernel_name: str, msg_info: str, log_level: AscendCLogLevel, option: Option = Option.DEFAULT):
        short_soc_version = global_var_storage.get_variable("ascendc_short_soc_version")
        current_time = datetime.now(tz=timezone.utc)
        tim_head = "[\033[32m{}-{}-{} {}:{}:{}\033[0m]".format(current_time.year, 
                                                               current_time.month,
                                                               current_time.day,
                                                               current_time.hour,
                                                               current_time.minute,
                                                               current_time.second)
        level_info = " [{}]".format(LOG_LEVEL_TO_STR[log_level])
        log_msg = tim_head + level_info
        if option is not LogUtil.Option.NON_SOC:
            log_msg += " [{}]".format(short_soc_version.lower())
        if kernel_name != "":
            log_msg += " {}".format(kernel_name)
        log_msg += " {}".format(msg_info)
        print(log_msg, flush=True)

    @staticmethod
    def detail_log_print(kernel_name: str, msg_info: str, log_level: AscendCLogLevel, option: Option = Option.DEFAULT):
        plog_switch = os.environ.get("ASCEND_GLOBAL_EVENT_ENABLE")
        if plog_switch is not None and int(plog_switch) == 1:
            logpid = os.getpid()
            python_exe = os.path.basename(sys.executable)
            thread_id = threading.currentThread().ident
            current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
            microsecond = int(time.time_ns() % 1_000_000_000)
            milliseconds = microsecond // 1_000_000
            microseconds = (microsecond % 1_000_000) // 1000
            tim_head = f" {current_time}.{milliseconds:03d}.{microseconds:03d}"
            nanoseconds = time.monotonic_ns()
            level_info = "[INFO] "
            log_msg = f"{level_info}ASC({logpid}, {python_exe}):{tim_head}"
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            if caller_frame:
                filename = os.path.basename(caller_frame.f_code.co_filename)
                line_number = caller_frame.f_lineno
                file_line = f"[{filename}:{line_number}]"
            else:
                file_line = "[unknown:0]"
            del frame
    
            log_msg += " {} [tid: {}] {}".format(file_line, thread_id, msg_info)
            log_msg += " , timestamp: {}ns".format(nanoseconds)
            print(log_msg, flush=True)

            
    @staticmethod
    def plog_print(kernel_name: str, msg_info: str, log_level: AscendCLogLevel, option: Option = Option.DEFAULT):
        # plog print
        short_soc_version = global_var_storage.get_variable("ascendc_short_soc_version")
        plog_log_msg = "[AscendCCompiler] "
        if option is not LogUtil.Option.NON_SOC:
            plog_log_msg = "[{}] ".format(short_soc_version.lower())
        if kernel_name != "":
            plog_log_msg += " {} ".format(kernel_name)
        plog_log_msg += msg_info
        if log_level == AscendCLogLevel.LOG_DEBUG:
            LOG_INSTANCE.debug(plog_log_msg)
        elif log_level == AscendCLogLevel.LOG_INFO:
            LOG_INSTANCE.info(plog_log_msg)
        elif log_level == AscendCLogLevel.LOG_WARNING:
            LOG_INSTANCE.warn(plog_log_msg)
        elif log_level == AscendCLogLevel.LOG_ERROR:
            LOG_INSTANCE.error(plog_log_msg)
