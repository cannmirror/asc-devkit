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
 * \file task_executor.cpp
 * \brief
 */
#include "task_executor.h"

#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <future>

#include "ascc_utils.h"

namespace Ascc {

void TaskExecutor::AddTaskGroup(const TaskGroup& group)
{
    tasks_[static_cast<int>(group.step)].emplace_back(group);
}

bool TaskExecutor::ExecuteTasks()
{
    if (enablePrintTasks_) {
        PrintTasks();
    }
    for (auto& kv : tasks_) {
        // execute all task groups at current level in parallel
        std::vector<std::future<bool>> futures;
        for (auto& group : kv.second) {
            futures.emplace_back(
                std::async(std::launch::async,
                TaskExecutor::ExecuteGroup,
                std::ref(group), enableStatistics_)
            );
        }

        // collect result of current step
        // Note: must wait until all tasks are executed, then return res. Otherwise might cause double free
        bool taskRes = true;
        for (auto& future : futures) {
            auto groupResult = future.get();
            taskRes = taskRes && groupResult;
        }
        if (!taskRes) {
            return false;
        }
    }

    if (enableStatistics_) {
        ReportStatistics();
    }
    return true;
}

void TaskExecutor::PrintTasks() const
{
    for (auto& [step, taskGroupVec] : tasks_) {
        for (auto& taskGroup : taskGroupVec) {
            for (auto& task : taskGroup.tasks) {
                printf("=== Process Command ===\n");
                printf("Step %d : #$ %s\n\n", step, task.GetCommand().c_str());
            }
        }
    }
}

bool TaskExecutor::ExecuteGroup(TaskGroup& group, bool enableStatistics)
{
    for (auto& task : group.tasks) {
        // 1. combine args into command
        std::vector<llvm::StringRef> argsRef;
        argsRef.reserve(task.args.size() + 1);  // executable path + args
        argsRef.emplace_back(task.execPath);    // first arg must be the executable path

        for (const auto& arg : task.args) {
            argsRef.emplace_back(arg);
        }

        llvm::ArrayRef<llvm::StringRef> argsArray = llvm::makeArrayRef(argsRef);

        // 2. execute command
        std::string errorMsg;
        bool executionFailed = false;
        llvm::Optional<llvm::sys::ProcessStatistics> procStats;
        int exitCode = llvm::sys::ExecuteAndWait(
            llvm::StringRef(task.execPath),
            argsArray,
            llvm::None,
            {}, /* Redirects */
            0, /* SecondsToWait */
            0, /* MemoryLimit */
            &errorMsg,
            &executionFailed,
            enableStatistics ? &procStats : nullptr,  // process stats (optional)
            nullptr /* AffinityMask */
        );
        // 3. report launch result
        // executionFailed: only shows whether subprocess is launched successfully
        // exitCode: shows the result of execution (might launch successfully but encounter error during execution)
        if (executionFailed || exitCode != 0) {
            Ascc::HandleError("Execute command failed: " + task.GetCommand());
            Ascc::HandleError("Error message: " + errorMsg);
            Ascc::HandleError("ExitCode: " + std::to_string(exitCode));
            return false;
        }

        if (enableStatistics) {
            if (procStats.hasValue()) {
                task.stats = procStats.getValue();
            } else {
                Ascc::HandleErrorAndCheckLog("Failed to get stats for task: " + task.execPath);
            }
        }
    }

    return true;
}

void TaskExecutor::EnablePrintTasks()
{
    enablePrintTasks_ = true;
}

void TaskExecutor::EnableStatistics()
{
    enableStatistics_ = true;
}

void TaskExecutor::ReportStatistics() const
{
    std::cout << "\n=== Process Statistics ===\n";
    for (const auto& [step, taskGroupVec] : tasks_) {
        std::cout << "Step " << static_cast<int>(step) << ":\n";
        for (auto& taskGroup : taskGroupVec) {
            for (const auto& task : taskGroup.tasks) {
                // convert to millisecond (double)
                double userTime = task.stats.UserTime.count() / 1e3;
                double totalTime = task.stats.TotalTime.count() / 1e3;
                // use std::ostringstream for formatting
                std::ostringstream stream;
                constexpr int PRECISION = 2;
                stream << std::fixed << std::setprecision(PRECISION) << userTime;
                std::string formattedUserTime = stream.str();
                stream.str(std::string());
                stream << std::fixed << std::setprecision(PRECISION) << totalTime;
                std::string formattedTotalTime = stream.str();
                std::cout << "  Task: " << task.execPath << "\n"
                          << "    UserTime:  " << formattedUserTime << "ms\n"
                          << "    TotalTime: " << formattedTotalTime << "ms\n"
                          << "    Peak Memory: " << task.stats.PeakMemory << " KB" << std::endl;
            }
        }
    }
}

} // namespace Ascc
