/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef LLVM_SUPPORT_RAW_OSTREAM_H
#define LLVM_SUPPORT_RAW_OSTREAM_H

#include <system_error>
#include <cstring>
#include <string>
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/ADT/StringRef.h"

#define STDERR_FILENO	2

namespace llvm {
class raw_ostream {
public:
    raw_ostream() = default;
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const char* str) {
    return os;
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const std::string& str) {
    return os;
}

class raw_svector_ostream : public raw_ostream {
    SmallVectorImpl<char> &OS;
public:
    raw_svector_ostream() = default;

    raw_svector_ostream(SmallVectorImpl<char> &O) : OS(O) {}
};

class raw_string_ostream : public raw_ostream {
    std::string &OS;
public:
    raw_string_ostream(std::string &O) : OS(O) {}
    void flush() { return; }
};

class raw_fd_ostream : public raw_ostream {
private:
    int fd_;
    bool shouldClose_;
    StringRef Filename_;
    std::error_code EC_;
    sys::fs::OpenFlags Flags_;
public:
    raw_fd_ostream(StringRef Filename, std::error_code &EC, sys::fs::OpenFlags Flags)
        : Filename_(Filename), EC_(EC), Flags_(Flags) {}
    raw_fd_ostream(int fd, bool shouldClose, bool unbuffered=false)
        : fd_(fd), shouldClose_(shouldClose) {}
};

inline raw_ostream &errs() {
    static raw_fd_ostream S(STDERR_FILENO, false, true);
    return S;
}
} // llvm namespace
#endif