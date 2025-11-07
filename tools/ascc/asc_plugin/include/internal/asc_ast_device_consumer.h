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
 * \file asc_ast_device_consumer.h
 * \brief
 */
#ifndef __INCLUDE_ASC_AST_DEVICE_CONSUMER_H__
#define __INCLUDE_ASC_AST_DEVICE_CONSUMER_H__

#include <utility>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/Tooling.h>
#include <clang/AST/ASTContext.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceManager.h>
#include <clang/AST/Mangle.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Hashing.h>

#include "asc_interface.h"
#include "asc_utils.h"
#include "asc_auto_identify_ktype.h"

namespace llvm {
template<>
struct DenseMapInfo<AscPlugin::KernelFuncInfo> {
    static AscPlugin::KernelFuncInfo getEmptyKey() {
        return {"", "", 0, 0};
    }
    static AscPlugin::KernelFuncInfo getTombstoneKey() {
        return {"", "", ~0U, ~0U};
    }
    static unsigned getHashValue(const AscPlugin::KernelFuncInfo &info) {
        return hash_combine(hash_value(info.mangledName),
                            hash_value(info.fileName),
                            info.lineNum,
                            info.colNum);
    }
    static bool isEqual(const AscPlugin::KernelFuncInfo &lhs, const AscPlugin::KernelFuncInfo &rhs) {
        return lhs == rhs;
    }
};
}   // namespace llvm

namespace AscPlugin {

void StoreFuncKernelType(const AscPlugin::KernelFuncInfo &kernelKey, const std::string &kernelTypeStr);
std::vector<KernelMetaType> GetBishengKType(const KernelInfo& kernelInfo);
std::pair<std::vector<KernelMetaType>, KfcScene> GetKernelFuncScene(const AscPlugin::KernelInfo& kernelInfo);
void GetMangledName(const clang::FunctionDecl *funcDecl);
KernelFuncInfo GetKernelInfo(const clang::FunctionDecl *funcDecl);
extern llvm::DenseMap<KernelFuncInfo, std::pair<std::vector<KernelMetaType>, KfcScene>> g_kernelFuncType;

class ASTDeviceVisitor : public clang::RecursiveASTVisitor<ASTDeviceVisitor> {
public:
    ASTDeviceVisitor(clang::ASTContext &context, clang::CompilerInstance &compiler);
    bool VisitFunctionDecl(const clang::FunctionDecl *funcDecl);
    bool VisitVarDecl(clang::VarDecl *varDecl);
    bool VisitCallExpr(clang::CallExpr *exprCall);
private:
    clang::ASTContext &context_;
    clang::CompilerInstance &compiler_;
    clang::SourceManager &srcManager_;
    llvm::DenseMap<AscPlugin::KernelFuncInfo, llvm::StringRef> kernelFuncsKernelType_;
    uint32_t GetLineNumber(const clang::Stmt *stmt) const;
    void FindKernelType(clang::VarDecl *varDecl);
    void FindMatmulObjRegister(clang::VarDecl *varDecl);
    void FindOpSystemCfg(const clang::VarDecl *varDecl) const;
};

class ASTDeviceConsumer : public clang::ASTConsumer {
public:
    ASTDeviceConsumer(clang::ASTContext &context, clang::CompilerInstance &compiler);
    void HandleTranslationUnit(clang::ASTContext &context) override;
private:
    ASTDeviceVisitor visitor_;
    clang::CompilerInstance &compiler_;
};

class DeviceAnalyzeAction : public clang::ASTFrontendAction {
public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance &compiler, llvm::StringRef inputFile) override;
};

class AscDiagnostic : public clang::DiagnosticConsumer {
public:
    explicit AscDiagnostic() = default;
    void HandleDiagnostic(clang::DiagnosticsEngine::Level diagLevel, const clang::Diagnostic &info) override;
    bool IncludeInDiagnosticCounts() const override
    {
        return true;  // 允许错误计入总数
    }
private:
    std::string GetSourceLine(const clang::SourceManager &srcManager, const clang::SourceLocation &srcLoc) const;
};
}  // namespace AscPlugin
#endif