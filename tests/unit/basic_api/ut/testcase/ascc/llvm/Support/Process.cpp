/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "Process.h"
#include <vector>
#include <cstdio>
#include <string>
#include <memory>


namespace llvm {
namespace sys {


int ExecuteAndWait(StringRef Program,  ///< Path of the program to be executed. It is
                                       ///< presumed this is the result of the findProgramByName method.
    ArrayRef<StringRef> Args,  ///< An array of strings that are passed to the
    ///< program.  The first element should be the name of the program.
    ///< The array should **not** be terminated by an empty StringRef.
    Optional<ArrayRef<StringRef>> Env,  ///< An optional vector of
    ///< strings to use for the program's environment. If not provided, the
    ///< current program's environment will be used.  If specified, the
    ///< vector should **not** be terminated by an empty StringRef.
    ArrayRef<Optional<StringRef>> Redirects,  ///<
    ///< An array of optional paths. Should have a size of zero or three.
    ///< If the array is empty, no redirections are performed.
    ///< Otherwise, the inferior process's stdin(0), stdout(1), and stderr(2)
    ///< will be redirected to the corresponding paths, if the optional path
    ///< is present (not \c llvm::None).
    ///< When an empty path is passed in, the corresponding file descriptor
    ///< will be disconnected (ie, /dev/null'd) in a portable way.
    unsigned SecondsToWait,  ///< If non-zero, this specifies the amount
    ///< of time to wait for the child process to exit. If the time
    ///< expires, the child is killed and this call returns. If zero,
    ///< this function will wait until the child finishes or forever if
    ///< it doesn't.
    unsigned MemoryLimit,  ///< If non-zero, this specifies max. amount
    ///< of memory can be allocated by process. If memory usage will be
    ///< higher limit, the child is killed and this call returns. If zero
    ///< - no memory limit.
    std::string *ErrMsg,  ///< If non-zero, provides a pointer to a
    ///< string instance in which error messages will be returned. If the
    ///< string is non-empty upon return an error occurred while invoking the
    ///< program.
    bool *ExecutionFailed,
    Optional<ProcessStatistics> *ProcStat,  ///< If non-zero,
    /// provides a pointer to a structure in which process execution
    /// statistics will be stored.
    BitVector *AffinityMask  ///< CPUs or processors the new
                                       /// program shall run on.
)
{
    std::vector<StringRef> ArgsV(Args.Data, Args.Data + Args.Length);
    std::string cmd = Program.str() + " ";
    for (auto S: ArgsV) {
        cmd += S.str() + " ";
    }
    // 打开管道读取命令输出
    std::unique_ptr<FILE, decltype(&pclose)> pipe(
        popen(cmd.c_str(), "r"), 
        pclose
    );
    
    if (!pipe) {
        *ErrMsg = "launch failed\n";
        return 1;
    }
    
    // 读取输出
    std::string result;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        result += buffer;
    }
    
    // 获取退出状态
    int status = pclose(pipe.release());
    int exit_code = WEXITSTATUS(status);
    
    if (exit_code == 0) {
        return 0;
    } else {
        *ErrMsg = "exit_code not 0 \n";
        return exit_code;
    }
    if (ProcStat != nullptr) {
        ProcStat->setValue();
    }
    return 0;
}
ProcessInfo ExecuteNoWait(StringRef Program, ArrayRef<StringRef> Args, Optional<ArrayRef<StringRef>> Env,
    ArrayRef<Optional<StringRef>> Redirects, unsigned MemoryLimit, std::string *ErrMsg, bool *ExecutionFailed,
    BitVector *AffinityMask)
{
    return ProcessInfo();
}

ProcessInfo Wait(const ProcessInfo &PI, unsigned SecondsToWait, bool WaitUntilTerminates, std::string *ErrMsg,
    Optional<ProcessStatistics> *ProcStat)
{
    if (ProcStat != nullptr) {
        ProcStat->setValue();
    }
    return ProcessInfo();
}

}  // namespace sys
}  // namespace llvm