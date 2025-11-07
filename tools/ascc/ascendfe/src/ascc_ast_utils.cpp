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

/*!
 * \file ascc_ast_utils.cpp
 * \brief
 */
#include "ascc_ast_utils.h"
#include "ascc_common_utils.h"
#include "ascc_info_storage.h"
#include "ascc_global_env_manager.h"
#include "ascc_argument_manager.h"
#include "ascc_match_global_info.h"
#include "ascc_log.h"

namespace Ascc {
static const std::unordered_map<std::string, Ascc::CodeMode> KERNEL_TYPE_MAP_V220 = {
    {"KERNEL_TYPE_AIV_ONLY", Ascc::CodeMode::KERNEL_TYPE_AIV_ONLY},
    {"KERNEL_TYPE_AIC_ONLY", Ascc::CodeMode::KERNEL_TYPE_AIC_ONLY},
    {"KERNEL_TYPE_MIX_AIV_1_0", Ascc::CodeMode::KERNEL_TYPE_MIX_AIV_1_0},
    {"KERNEL_TYPE_MIX_AIC_1_0", Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_0},
    {"KERNEL_TYPE_MIX_AIC_1_1", Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_1},
    {"KERNEL_TYPE_MIX_AIC_1_2", Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_2}
};
// chech whether variable declaration is in namespace XXX
bool IsVarInNamespace(const clang::VarDecl* var, const std::string& namespaceName) {
    const clang::DeclContext* ctx = var->getDeclContext();   // get variable declaration context
    while (ctx != nullptr) {
        // check whether context is namespace
        const auto *ns = clang::dyn_cast<clang::NamespaceDecl>(ctx);
        if (ns != nullptr) {
            if (ns->getName() == namespaceName) {
                return true;
            }
        }
        ctx = ctx->getParent();   // check parent scope
    }
    return false;
}

// find the dtype of variable
const std::string FindVarTypeStr(const clang::VarDecl *varDecl)
{
    return varDecl->getType().getAsString();
}

void SetFuncFileKernelType(const clang::FunctionDecl *funcDecl, const std::string &kernelTypeStr)
{
    clang::SourceManager &srcMgr = funcDecl->getASTContext().getSourceManager();
    clang::SourceLocation funcLoc = funcDecl->getLocation();
    clang::FullSourceLoc fullLoc(funcLoc, srcMgr);
    std::string fileName = fullLoc.getFileEntry()->getName().str();
    uint32_t lineNo = static_cast<uint32_t>(fullLoc.getSpellingLineNumber());
    auto &storage = AsccInfoStorage::GetInstance();
    if (auto allFuncInfo = std::dynamic_pointer_cast<AsccInfoFunction>(
            storage.GetInfo(fileName, AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION))) {
        std::string key = funcDecl->getNameAsString() + ":" + std::to_string(lineNo);
        if (auto funcInfo = allFuncInfo->GetFunctionInfo(key)) {
            if (KERNEL_TYPE_MAP_V220.find(kernelTypeStr) != KERNEL_TYPE_MAP_V220.end()) {
                funcInfo->kernelType = KERNEL_TYPE_MAP_V220.at(kernelTypeStr);
                for (auto& [manglingName, mangledFunc] : funcInfo->mangledToInstFuncInfo) {
                    (void)manglingName;
                    mangledFunc->kernelType = KERNEL_TYPE_MAP_V220.at(kernelTypeStr);
                }
            } else {
                HandleError(std::string("kernel type set error, func name: " + funcDecl->getNameAsString() +
                                        ", line: " + std::to_string(lineNo) + ", kernel type: " + kernelTypeStr));
            }
        }
    } else {
        HandleError(std::string("funcInfo not found, func name: " + fileName + ", line: " + std::to_string(lineNo)));
    }
}

void SetFuncFileKernelHasKfc(const clang::FunctionDecl *funcDecl, bool hasKfc)
{
    clang::SourceManager &srcMgr = funcDecl->getASTContext().getSourceManager();
    clang::SourceLocation funcLoc = funcDecl->getLocation();
    clang::FullSourceLoc fullLoc(funcLoc, srcMgr);
    std::string fileName = fullLoc.getFileEntry()->getName().str();
    uint32_t lineNo = static_cast<uint32_t>(fullLoc.getSpellingLineNumber());
    auto &storage = AsccInfoStorage::GetInstance();
    if (auto allFuncInfo = std::dynamic_pointer_cast<AsccInfoFunction>(
            storage.GetInfo(fileName, AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION))) {
        std::string key = funcDecl->getNameAsString() + ":" + std::to_string(lineNo);
        if (auto funcInfo = allFuncInfo->GetFunctionInfo(key)) {
            funcInfo->hasKfcServer = hasKfc;
            for (auto& [manglingName, mangledFunc] : funcInfo->mangledToInstFuncInfo) {
                (void)manglingName;
                mangledFunc->hasKfcServer = hasKfc;
            }
        }
    } else {
        HandleError(std::string("funcInfo not found, func name: " + fileName + ", line: " + std::to_string(lineNo)));
    }
}

void WarnNoFuncCalls(const AsccInfoFunction::FunctionInfo& funcInfo)
{
    std::string fileName = AsccArgumentManager::GetInstance().GetInputFile();
    ASC_LOG_ASC_WARN(DEVICE_STUB, "Function [%s] in file %s at line %u does not have function call!",
        funcInfo.funcName.c_str(), fileName.c_str(), funcInfo.lineNo);
}

bool GenerateJsonObj(llvm::json::Object& rootJsonObj, const AsccInfoFunction::FunctionInfo& funcInfo,
    const std::string& stubFilePath, const std::string& inputFile, bool enableDFX, uint32_t dumpSize)
{
    // if <<<>>> is not called, do not need to generate jsob obj
    if (!AsccMatchGlobalInfo::GetInstance().IsCalled(funcInfo.manglingName)) {
        WarnNoFuncCalls(funcInfo);
        return false;
    }
    auto& manager = Ascc::AsccGlobalEnvManager::GetInstance();
    llvm::json::Object kernelFuncObj;
    kernelFuncObj["func_name"] = funcInfo.funcName;
    kernelFuncObj["stub_filename"] = stubFilePath + "/stub_" + Ascc::GetFileName(inputFile);
    kernelFuncObj["kernel_type"] = static_cast<uint32_t>(funcInfo.kernelType);
    kernelFuncObj["enable_dfx"] = enableDFX;
    kernelFuncObj["dump_size"] = dumpSize;
    kernelFuncObj["mix_num_lineno"] = manager.mixNumLineNum;
    kernelFuncObj["dump_workspace_lineno"] = manager.dumpWorkspaceLineNum;
    kernelFuncObj["dump_uint_lineno"] = manager.dumpUintLineNum;
    rootJsonObj[funcInfo.manglingName] = std::move(kernelFuncObj);
    ASC_LOG_ASC_DEBUG(DEVICE_STUB, "Function [%s] has function call!", funcInfo.manglingName.c_str());
    return true;
}

// after all json objs added to jsonObj, generate the final json file
AsccStatus MergeJsonObjs(llvm::json::Object jsonObj, const std::string& jsonName)
{
    std::error_code errorCode;
    // Generate json files when preprocess tasks
    if (Ascc::AsccArgumentManager::GetInstance().GetPreTaskType() != PreTaskType::NONE) {
        std::string jsonPath = Ascc::AsccArgumentManager::GetInstance().GetModulePath() + "/" + GetFileName(jsonName);
        llvm::raw_fd_ostream osJson(jsonPath, errorCode, llvm::sys::fs::OF_Text);
        llvm::json::Value root = std::move(jsonObj);
        ASCC_CHECK((!errorCode), {llvm::errs() << "Error: " << errorCode.message() << "\n";});
        osJson << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
    }
    return AsccStatus::SUCCESS;
}

} // namespace Ascc