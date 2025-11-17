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
 * \file ascc_ast_device_consumer.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_AST_DEVICE_CONSUMER_H__
#define __INCLUDE_ASCC_AST_DEVICE_CONSUMER_H__
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/Tooling.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Mangle.h>
#include <clang/Basic/Diagnostic.h>

namespace Ascc {
class ASTDeviceVisitor : public clang::RecursiveASTVisitor<ASTDeviceVisitor> {
public:
    ASTDeviceVisitor(clang::ASTContext &context, clang::CompilerInstance &compiler);
    bool VisitFunctionDecl(const clang::FunctionDecl *funcDecl);
    bool VisitCallExpr(clang::CallExpr *exprCall);
    unsigned GetLineNumber(const clang::Stmt *stmt) const;
    bool VisitVarDecl(clang::VarDecl *varDecl);
private:
    void UpdateKernelKfcInfo(clang::VarDecl *varDecl);
    clang::ASTContext &context_;
    clang::CompilerInstance &compiler_;
    clang::SourceManager &srcManager_;
    llvm::DenseMap<const clang::FunctionDecl*, llvm::StringRef> globalAicoreFuncsKernelType_;
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
}  // namespace Ascc
#endif