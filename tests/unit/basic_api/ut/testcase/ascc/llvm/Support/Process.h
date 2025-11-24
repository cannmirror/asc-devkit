/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef LLVM_SUPPORT_PROCESS_H
#define LLVM_SUPPORT_PROCESS_H

#include <chrono>
#include <string>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
template <typename T>
class Optional {
public:
    constexpr Optional() = default;
    constexpr Optional(NoneType)
    {}
    constexpr bool hasValue() const
    {
        return has_val;
    }
    constexpr const T &getValue() const &
    {
        return val;
    }
    void setValue() {has_val = true;}
    T val;
    bool has_val = false;//for gtest
    T &operator*() & { return val; }
    T *operator->() { return &val; }
};
class BitVector {};

namespace sys {
typedef ::pid_t procid_t;
typedef procid_t process_t;
/// This struct encapsulates information about a process.
struct ProcessInfo {
    enum : procid_t { InvalidPid = 0 };

    procid_t Pid;       /// The process identifier.
    process_t Process;  /// Platform-dependent process object.

    /// The return code, set after execution.
    int ReturnCode;
};

struct ProcessStatistics {
    std::chrono::microseconds TotalTime;
    std::chrono::microseconds UserTime;
    uint64_t PeakMemory = 0;  ///< Maximum resident set size in KiB.
};

int ExecuteAndWait(StringRef Program,  ///< Path of the program to be executed. It is
    ///< presumed this is the result of the findProgramByName method.
    ArrayRef<StringRef> Args,  ///< An array of strings that are passed to the
    ///< program.  The first element should be the name of the program.
    ///< The array should **not** be terminated by an empty StringRef.
    Optional<ArrayRef<StringRef>> Env = None,  ///< An optional vector of
    ///< strings to use for the program's environment. If not provided, the
    ///< current program's environment will be used.  If specified, the
    ///< vector should **not** be terminated by an empty StringRef.
    ArrayRef<Optional<StringRef>> Redirects = {},  ///<
    ///< An array of optional paths. Should have a size of zero or three.
    ///< If the array is empty, no redirections are performed.
    ///< Otherwise, the inferior process's stdin(0), stdout(1), and stderr(2)
    ///< will be redirected to the corresponding paths, if the optional path
    ///< is present (not \c llvm::None).
    ///< When an empty path is passed in, the corresponding file descriptor
    ///< will be disconnected (ie, /dev/null'd) in a portable way.
    unsigned SecondsToWait = 0,  ///< If non-zero, this specifies the amount
    ///< of time to wait for the child process to exit. If the time
    ///< expires, the child is killed and this call returns. If zero,
    ///< this function will wait until the child finishes or forever if
    ///< it doesn't.
    unsigned MemoryLimit = 0,  ///< If non-zero, this specifies max. amount
    ///< of memory can be allocated by process. If memory usage will be
    ///< higher limit, the child is killed and this call returns. If zero
    ///< - no memory limit.
    std::string *ErrMsg = nullptr,  ///< If non-zero, provides a pointer to a
    ///< string instance in which error messages will be returned. If the
    ///< string is non-empty upon return an error occurred while invoking the
    ///< program.
    bool *ExecutionFailed = nullptr,
    Optional<ProcessStatistics> *ProcStat = nullptr,  ///< If non-zero,
    /// provides a pointer to a structure in which process execution
    /// statistics will be stored.
    BitVector *AffinityMask = nullptr  ///< CPUs or processors the new
                                       /// program shall run on.
);
ProcessInfo ExecuteNoWait(StringRef Program, ArrayRef<StringRef> Args, Optional<ArrayRef<StringRef>> Env,
    ArrayRef<Optional<StringRef>> Redirects = {}, unsigned MemoryLimit = 0, std::string *ErrMsg = nullptr,
    bool *ExecutionFailed = nullptr, BitVector *AffinityMask = nullptr);

ProcessInfo Wait(const ProcessInfo &PI, unsigned SecondsToWait, bool WaitUntilTerminates, std::string *ErrMsg,
    Optional<ProcessStatistics> *ProcStat);
}  // namespace sys
}  // namespace llvm

#endif  // LLVM_SUPPORT_PROCESS_H