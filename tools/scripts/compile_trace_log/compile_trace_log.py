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
import re
import argparse
import json
from pathlib import Path
import inspect
import sys


def extract_info_lines(filename):
    matching_lines = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                stripped_line = line.strip()
                if '[INFO] ASC(' in stripped_line:
                    matching_lines.append(stripped_line)
    except FileNotFoundError:
        frame = inspect.currentframe()
        print(f"ERROR: {frame.f_code.co_filename}:line {frame.f_lineno}: File '{filename}' not found.", file=sys.stderr)
        raise
    except PermissionError:
        frame = inspect.currentframe()
        print(
            f"ERROR: {frame.f_code.co_filename}:line {frame.f_lineno}: "
            f"Permission denied when reading '{filename}'.", file=sys.stderr
            )
        raise
    except UnicodeDecodeError as e:
        frame = inspect.currentframe()
        print(
            f"ERROR: {frame.f_code.co_filename}:line {frame.f_lineno}: "
            f"Failed to decode '{filename}' with UTF-8 encoding: {e}",
            file=sys.stderr
        )
        raise

    if matching_lines:
        print(f"INFO: Found {len(matching_lines)} matching log rows：")
    else:
        frame = inspect.currentframe()
        raise RuntimeError(f"INFO: {frame.f_code.co_filename}:line {frame.f_lineno}: No log starting with [INFO] "
                    "ASC was found.Please check if the log file is empty or if the ASCEND_GLOBAL_EVENT_ENABLE"
                    " environment variable for controlling the compilation time stamp is not set to 1. ")
    
    return matching_lines


def save_info_lines(trace_events, output_file):
    # 构建最终 JSON 结构
    data = {
        "traceEvents": trace_events
    }

    # 写入 JSON 文件
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except PermissionError:
        frame = inspect.currentframe()
        print(
            f"ERROR: {frame.f_code.co_filename}:line {frame.f_lineno}: "
            f"Permission denied when writing to file '{output_file}'.",
            file=sys.stderr
        )
        raise
    except FileNotFoundError:
        frame = inspect.currentframe()
        print(
            f"ERROR: {frame.f_code.co_filename}:line {frame.f_lineno}: "
            f"The specified path does not exist (may be a directory path): '{output_file}'.", file=sys.stderr
            )
        raise
    except IsADirectoryError:
        frame = inspect.currentframe()
        print(
            f"ERROR: {frame.f_code.co_filename}:line {frame.f_lineno}: "
            f"The output path is a directory, not a file: '{output_file}'.", file=sys.stderr
            )
        raise
    except Exception as e:
        frame = inspect.currentframe()
        print(
            f"ERROR: {frame.f_code.co_filename}:line {frame.f_lineno}: "
            f"Unknown error: failed to save data to file '{output_file}', details: {e}", file=sys.stderr
            )
        raise
    return


def build_traceEvents(len_pid, optype, trace_events, timestamp, pid, tid, tilingtype, compile_stage):
    for i in range(len_pid):
        num = int(i / 12) * 7# 除12是因为，打12个时间点，只有7个tiling信息
        idx = i % 12
        if idx == 11: 
            continue
        if idx == 0:
            name = optype[num] + "   compile op"
            trace_events.append({
                "optype": optype[num],
                "name": name,
                "cat": "compile_op",
                "ph": "B",
                "ts": timestamp[i],
                "pid": pid[i],
                "tid": tid[i],
                "args": { 
                    "tiling_key": tilingtype[num]  
                } 
                })
            last_i = i + 11 
            trace_events.append({
                "optype": optype[num],
                "name": name,
                "cat": "compile_op",
                "ph": "E",
                "ts": timestamp[last_i],
                "pid": pid[last_i],
                "tid": tid[last_i]
                })
        else:
            common_trace_event(
                trace_events,
                compile_stage[idx],
                optype[num],
                timestamp[i],
                pid[i],
                tid[i],
                tilingtype[num]
            )

    return trace_events


def common_trace_event(trace_events, compile_stage, optype, timestamp, pid, tid, tilingtype):
    name, stage = compile_stage.rsplit(' ', 1)
    if (stage == "start"):
        trace_events.append({ 
            "optype": optype,
            "name": name,
            "cat": "compile_op",
            "ph": "B",
            "ts": timestamp,
            "pid": pid,
            "tid": tid,
            "args": { 
                "tiling_key": tilingtype
            }
            })
    if (stage == "end"):
        trace_events.append({ 
            "optype": optype,
            "name": name,
            "cat": "compile_op",
            "ph": "E",
            "ts": timestamp,
            "pid": pid,
            "tid": tid
            })
            
    return trace_events


def compile_trace(input_file, output_file):
    matching_lines = extract_info_lines(input_file)
    pid = []
    tid = []
    timestamp = []
    optype = []
    tilingtype = []
    for line in matching_lines:
        # 提取第一个数字（pid）
        numbers = re.findall(r'\b\d+\b', line)
        p = int(numbers[0])
        pid.append(p)
        #timestamp
        match = re.search(r'timestamp:\s*(\d+)ns', line)
        ts = int(match.group(1))
        timestamp.append(ts)
        #tid
        match = re.search(r'\[tid:\s*(\d+)\]', line)
        t = int(match.group(1))
        tid.append(t)
        # 提取第一个 <...> 内容（optype）
        first_lt = line.find('<')
        first_gt = line.find('>', first_lt)
        if first_lt != -1 and first_gt != -1:
            op = line[first_lt + 1:first_gt]
            optype.append(op)
        # 提取第二个 <...> 内容（tilingtype）
        second_lt = line.find('<', first_gt + 1)
        second_gt = line.find('>', second_lt)
        if second_lt != -1 and second_gt != -1:
            til = line[second_lt + 1:second_gt]
            tilingtype.append(til)
    compile_stage = ["compile op start", \
                "preprocess start", \
                "preprocess end", \
                "generate tiling start", \
                "generate tiling end", \
                "generate kernel stub start", \
                "generate kernel stub end", \
                "compile kernel start", \
                "compile kernel end", \
                "link kernel start", \
                "link kernel end", \
                "compile op end"]
    # 构建 traceEvents 列表
    trace_events = []
    len_pid = len(pid)
    build_traceEvents(len_pid, optype, trace_events, timestamp, pid, tid, tilingtype, compile_stage)
    # 构建最终 JSON 结构
    save_info_lines(trace_events, output_file)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建命令行解析器
    parser = argparse.ArgumentParser(
        description=(
            "Extract the line starting with [INFO] ASC from the "
            "log file and output it as a JSON file."
        )
    )

    # 添加命令行参数
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input log file (e.g., out_log.txt)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="out_log_trace_output.json",
        help="Path to the output JSON file (e.g., out_log_trace_output.json)"
    )

    # 解析参数
    args = parser.parse_args()

    # 调用主函数
    try:
        compile_trace(args.input, args.output)
        print(f"[SUCCESS]: JSON file generated: {args.output}")
    except Exception as e:
        print(f"{e}")
        exit(1)
    