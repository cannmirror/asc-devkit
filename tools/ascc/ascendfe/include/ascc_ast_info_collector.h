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
 * \file ascc_ast_info_collector.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_AST_INFO_COLLECTOR_H__
#define __INCLUDE_ASCC_AST_INFO_COLLECTOR_H__
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Tooling.h>
#include <clang/AST/Mangle.h>
#include <clang/AST/ExprCXX.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/CompilerInstance.h>
#include <optional>

#include "ascc_log.h"
#include "ascc_global_env_manager.h"
#include "ascc_info_storage.h"
#include "ascc_info_function.h"

namespace Ascc {
class KerenelInfoCollector : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
    void run(const clang::ast_matchers::MatchFinder::MatchResult &result) override;

private:
    bool IsTemplate(const clang::FunctionDecl* funcDecl) const;
    std::shared_ptr<AsccInfoBase> GetStorageInfo(const std::string &file, const AscCursorTypes& infoType) const;
    std::string GetQualifiedScope(const clang::FunctionDecl *funcDecl) const;
    void KernelCallHandle(
        const clang::CUDAKernelCallExpr *kernelCall, const clang::ast_matchers::MatchFinder::MatchResult &result) const;
    void KernelDeclHandle(
        const clang::FunctionDecl *funcDecl, const clang::ast_matchers::MatchFinder::MatchResult &result) const;
    std::optional<std::string> GetEnumNameForValue(const clang::FunctionDecl* funcDecl, unsigned index, int64_t val,
        const clang::PrintingPolicy& policy) const;
};

class AsccDiagnostic : public clang::DiagnosticConsumer {
public:
    explicit AsccDiagnostic() = default;
    void HandleDiagnostic(clang::DiagnosticsEngine::Level diagLevel, const clang::Diagnostic &info) override;
    bool IncludeInDiagnosticCounts() const override
    {
        return true;  // 允许错误计入总数
    }
private:
    std::string GetSourceLine(const clang::SourceManager &srcManager, const clang::SourceLocation &srcLoc) const;
};

class AsccASTConsumer : public clang::ASTConsumer {
public:
    explicit AsccASTConsumer();
    void HandleTranslationUnit(clang::ASTContext &context) override;

private:
    clang::ast_matchers::MatchFinder finder_;
    KerenelInfoCollector callback_;
};

class AsccFrontendAction : public clang::ASTFrontendAction {
public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance &compileInst, llvm::StringRef inputFile) override;
};

} // namespace Ascc
#endif