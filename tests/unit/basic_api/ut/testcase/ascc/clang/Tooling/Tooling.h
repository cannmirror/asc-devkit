/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef CLANG_TOOLING_TOOLING_H
#define CLANG_TOOLING_TOOLING_H
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Frontend/FrontendAction.h"
#include "llvm/ADT/ArrayRef.h"
#include <string>

namespace clang {
namespace tooling {
class ToolAction {
public:
    ToolAction() = default;
};

class ClangTool {
public:
    ClangTool(const CompilationDatabase &Compilations, llvm::ArrayRef<std::string> SourcePaths) {}
    void appendArgumentsAdjuster(ArgumentsAdjuster Adjuster) {
        Adjuster(CommandLineArguments(), llvm::StringRef());
    }
    int run(ToolAction *Action) { return 0; }
};

class FrontendActionFactory : public ToolAction {
public:
    virtual FrontendAction *create() = 0;
};

template <typename T>
std::unique_ptr<FrontendActionFactory> newFrontendActionFactory() {
    class SimpleFrontendActionFactory : public FrontendActionFactory {
    public:
        FrontendAction *create() override { return new T; }
    };

    return std::unique_ptr<FrontendActionFactory>(
        new SimpleFrontendActionFactory);
}
}
}
#endif