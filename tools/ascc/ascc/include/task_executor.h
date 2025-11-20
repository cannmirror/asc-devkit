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
 * \file task_executor.h
 * \brief
 */

#ifndef _ASCC_TASK_EXECUTOR_H_
#define _ASCC_TASK_EXECUTOR_H_


#include <vector>
#include <string>
#include <map>
#include <mutex>
#include "task.h"

namespace Ascc {
// if need to link multiple cpp files into one object, create multiple TaskExecutor and recursively call ExecuteTasks
class TaskExecutor {
public:
    // add a group of task. Tasks within a group are executed in serial, across groups are executed in parallel
    void AddTaskGroup(const TaskGroup& group);

    // execute all tasks
    bool ExecuteTasks();

    // calculate statistics of tasks. Enable by --time(-time)
    // should be turned on before ExecuteTasks
    void EnableStatistics();

    // print compile commands. Enable by --verbose(-v)
    void EnablePrintTasks();
private:
    static bool ExecuteGroup(TaskGroup& group, bool enableStatistics);
    // print commands of all tasks
    void PrintTasks() const;
    void ReportStatistics() const;

    std::map<int, std::vector<TaskGroup>> tasks_;
    bool enableStatistics_ = false;                     // related to --time
    bool enablePrintTasks_ = false;                     // related to --verbose
};

} // namespace Ascc

#endif // _ASCC_TASK_EXECUTOR_H_