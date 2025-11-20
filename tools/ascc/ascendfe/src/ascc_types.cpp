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
 * \file ascc_types.cpp
 * \brief
 */

#include "ascc_types.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace Ascc {
static const std::unordered_map<CompileType, const char*> COMPILE_TYPE_TO_OPTION = {
    {CompileType::OBJECT, "-c"},
    {CompileType::SHARED, "-shared"},
    {CompileType::STATIC, "-static"}
};

std::string CompileArgs::GetCmd(
    const std::string &compiler, bool isClang, CompileType type, bool isAst) const
{
    const std::vector<std::string>& cmdVec = GetCmdVector(compiler, isClang, type, isAst);
    std::string cmd = std::string();
    for (const auto& str : cmdVec) {
        cmd += str + " ";
    }
    return cmd;
}

std::vector<std::string> CompileArgs::GetCmdVector(
    const std::string &compiler, bool isClang, CompileType type, bool isAst) const
{
    std::vector<std::string> cmdVec = {compiler};
    std::vector<std::string> compileOption;
    for (const auto &opt : this->options) {
        compileOption.emplace_back(opt);
    }
    if (type != CompileType::NONE) {
        compileOption.emplace_back(COMPILE_TYPE_TO_OPTION.at(type));
    }
    for (const auto &def : this->definitions) {
        compileOption.emplace_back(std::string("-D" + def));
    }
    for (const auto &inc : this->incPaths) {
        compileOption.emplace_back(std::string("-I" + inc));
    }
    for (const auto &lib : this->linkFiles) {
        compileOption.emplace_back(std::string("-l" + lib));
    }
    for (const auto &path : this->linkPath) {
        compileOption.emplace_back(std::string("-L" + path));
    }
    for (const auto &header : this->incFiles) {
        compileOption.emplace_back("-include");
        compileOption.emplace_back(header);
    }
    if (!this->outputPath.empty()) {
        compileOption.emplace_back("-o");
        compileOption.emplace_back(this->outputPath);
    }
    if (isAst) {
        cmdVec.emplace_back(this->file);
        if (isClang) {
            cmdVec.emplace_back("--");
        }
        cmdVec.insert(cmdVec.end(), compileOption.begin(), compileOption.end());
    } else {
        cmdVec.insert(cmdVec.end(), compileOption.begin(), compileOption.end());
        if (isClang) {
            cmdVec.emplace_back("--");
        }
        cmdVec.emplace_back(this->file);
    }
    return cmdVec;
}
}