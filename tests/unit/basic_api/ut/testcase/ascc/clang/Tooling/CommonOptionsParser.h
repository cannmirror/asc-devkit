/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef CLANG_TOOLING_COMMON_OPTIONS_PARSER_H
#define CLANG_TOOLING_COMMON_OPTIONS_PARSER_H
#include <string>
#include <utility>
#include "llvm/Support/CommandLine.h"
#include "clang/Tooling/CompilationDatabase.h"

namespace clang {
namespace tooling {
class CommonOptionsParser {
private:
    std::unique_ptr<CompilationDatabase> Compilations;
    std::vector<std::string> SourcePathList;
public:
    CommonOptionsParser() = default;
    static std::unique_ptr<CommonOptionsParser>
    create(int &argc, const char **argv, llvm::cl::OptionCategory &Category)
    {
        if (std::string(argv[0]) == std::string("ABCDEFG"))
        {
            return nullptr;
        }
        auto parser = std::make_unique<CommonOptionsParser>();
        parser->Compilations = std::make_unique<CompilationDatabase>();
        return std::move(parser);
    }

    CompilationDatabase &getCompilations() {
        return *Compilations;
    }
    const std::vector<std::string> &getSourcePathList() const {
        return SourcePathList;
    }
};
}
}
#endif