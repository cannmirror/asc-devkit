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
 * \file asc_ast_utils.cpp
 * \brief
 */

#include "asc_ast_utils.h"

namespace AscPlugin {
std::string CompileArgs::GetCmd(const std::string& compiler, bool isClang, bool isAst) const
{
    const std::vector<std::string>& cmdVec = GetCmdVector(compiler, isClang, isAst);
    std::string cmd = std::string();
    for (const auto& str : cmdVec) {
        cmd += str + " ";
    }
    return cmd;
}

// return a vector that contains all compile options in CompileArgs
std::vector<std::string> CompileArgs::GetCmdVector(const std::string& compiler, bool isClang, bool isAst) const
{
    std::vector<std::string> cmdVec = {compiler};
    std::vector<std::string> compileOption;
    // considered as compile object file
    compileOption.emplace_back("-c");
    compileOption.insert(compileOption.end(), this->options.begin(), this->options.end());
    compileOption.insert(compileOption.end(), this->definitions.begin(), this->definitions.end());
    compileOption.insert(compileOption.end(), this->includePaths.begin(), this->includePaths.end());
    compileOption.insert(compileOption.end(), this->includeFiles.begin(), this->includeFiles.end());
    compileOption.insert(compileOption.end(), this->linkFiles.begin(), this->linkFiles.end());
    compileOption.insert(compileOption.end(), this->linkPaths.begin(), this->linkPaths.end());
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

void CompileArgs::RemoveOptions(std::vector<std::string>& removeOpts) {
    for (auto it = this->definitions.begin(); it != this->definitions.end(); ) {
        bool needRemove = false;
        for (const auto& opt : removeOpts) {
            if (*it == opt) {
                needRemove = true;
                break;
            }
        }
        if (needRemove) {
            it = this->definitions.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace AscPlugin