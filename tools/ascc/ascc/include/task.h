/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file task.h
 * \brief
 */

#ifndef _ASCC_TASK_H_
#define _ASCC_TASK_H_

#include <string>
#include <vector>
#include <llvm/Support/Process.h>

namespace Ascc {

enum class TaskStep : uint8_t {
    CODEGEN = 0,      // preprocess and generate code stage
    COMPILE,          // compile .o files
    DEVICE_LINK,      // link device files
    PACK_KERNEL,      // pack multiple .o files into one .o file
    LINK              // link .o files into .so or executable
};

struct Task {
    explicit Task(const std::string& path) : execPath(path) {}

    // Add optional arguments to the tasks (from user / default options)
    Task& AddArg(const std::string& arg)
    {
        args.emplace_back(arg);
        return *this;
    }

    Task& AddArg(const std::vector<std::string>& argsIn)
    {
        for (const auto& arg : argsIn) {
            args.emplace_back(arg);
        }
        return *this;
    }

    std::string GetCommand() const
    {
        std::string command = execPath;
        for (const auto& arg : args) {
            command += " " + arg;
        }
        return command;
    }

    std::string execPath;                 // executable path
    std::vector<std::string> args;        // total compile options for the tasks
    llvm::sys::ProcessStatistics stats;   // thread stats
};

struct TaskGroup {
    explicit TaskGroup(TaskStep ts) : step(ts) {}

    // add thread task
    void AddTask(const Task& task)
    {
        tasks.emplace_back(task);
    }

    TaskStep step;
    std::vector<Task> tasks;
};

}  // namespace Ascc

#endif  // _ASCC_TASK_H_