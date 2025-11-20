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
 * \file asc_ast_utils.h
 * \brief ast related struct and functions
 */
#ifndef __INCLUDE_INTERNAL_ASC_AST_UTILS_H__
#define __INCLUDE_INTERNAL_ASC_AST_UTILS_H__
#include <string>
#include <vector>

namespace AscPlugin {

class CompileArgs {
public:
    std::vector<std::string> definitions;
    std::vector<std::string> hostDefinitions;       // using -Xhost-start -Xhost-end
    std::vector<std::string> includePaths;
    std::vector<std::string> options;
    std::vector<std::string> includeFiles;
    std::vector<std::string> linkFiles;
    std::vector<std::string> linkPaths;
    std::string file;
    std::string customOption;
    std::string outputPath;

    std::string GetCmd(const std::string& compiler, bool isClang = true, bool isAst = false) const;
    std::vector<std::string> GetCmdVector(const std::string& compiler, bool isClang = true, bool isAst = false) const;
    void RemoveOptions(std::vector<std::string>& removeOpts);
};

}  // namespace AscPlugin
#endif // __INCLUDE_INTERNAL_ASC_AST_UTILS_H__