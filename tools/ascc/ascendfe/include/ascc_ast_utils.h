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
 * \file ascc_ast_utils.h
 * \brief
 */


#ifndef __INCLUDE_ASCC_AST_UTILS_H__
#define __INCLUDE_ASCC_AST_UTILS_H__
#include <string>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <llvm/Support/JSON.h>
#include "ascc_info_function.h"

namespace Ascc {

/**
 * @brief chech whether variable declaration is in namespace XXX
 * @param var variable declaration
 * @param namespaceName namespace where var should be in
 * @return true(is in namespace) / false (not in namespace)
 */
bool IsVarInNamespace(const clang::VarDecl* var, const std::string& namespaceName);

/**
 * @brief get the variable dtype
 * @param varDecl variable declaration
 * @return variable dtype name
 */
const std::string FindVarTypeStr(const clang::VarDecl *varDecl);

/**
 * @brief update function in AsccInfoStorage with kernel type
 * @param funcDecl function declaration
 * @param kernelTypeStr kernel type
 */
void SetFuncFileKernelType(const clang::FunctionDecl *funcDecl, const std::string &kernelTypeStr);

/**
 * @brief update function in AsccInfoStorage with info: whether has kfc
 * @param funcDecl function declaration
 * @param hasKfc whether has kfc server in this function
 */
void SetFuncFileKernelHasKfc(const clang::FunctionDecl *funcDecl, bool hasKfc);

/**
 * @brief Generate json object based on given info
 * @param rootJsonObj root json object
 * @param funcInfo function information
 * @param stubFilePath function related stub_xxxx.cpp
 * @param inputFile src .cpp file
 * @param enableDFX whether enable DFX (printf, dumpTensor, assert)
 * @param dumpSize dumpSize 1024(only assert) / 1024 * 1024 (printf)
 */
bool GenerateJsonObj(llvm::json::Object& rootJsonObj, const AsccInfoFunction::FunctionInfo& funcInfo,
    const std::string& stubFilePath, const std::string& inputFile, bool enableDFX, uint32_t dumpSize);

/**
 * @brief Generate a json file based on given jsonObj
 * @param jsonObj the json object that needs to be stored into json file
 * @param jsonName json file name
 * @return whether success or not
 */
AsccStatus MergeJsonObjs(llvm::json::Object jsonObj, const std::string& jsonName);

/**
 * @brief Warn users that xxx <<<>>> function has not been called
 * @param funcInfo function info that needs to check whether has been called
 */
void WarnNoFuncCalls(const AsccInfoFunction::FunctionInfo& funcInfo);

} // Ascc
#endif // __INCLUDE_ASCC_AST_UTILS_H__