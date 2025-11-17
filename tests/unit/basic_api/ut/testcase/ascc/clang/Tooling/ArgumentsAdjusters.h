/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef CLANG_TOOLING_ARGUMENTS_ADJUSTERS_H
#define CLANG_TOOLING_ARGUMENTS_ADJUSTERS_H
#include <vector>
#include <string>
#include <functional>
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace tooling {
using CommandLineArguments = std::vector<std::string>;
using ArgumentsAdjuster = std::function<CommandLineArguments(
    const CommandLineArguments &, llvm::StringRef Filename)>;
}
}
#endif