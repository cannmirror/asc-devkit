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
#ifndef LLVM_CLANG_ASTMATCHERS_ASTMATCHFINDER_H
#define LLVM_CLANG_ASTMATCHERS_ASTMATCHFINDER_H

#include "llvm/Support/Process.h"
#include "llvm/ADT/StringRef.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
namespace clang {
namespace ast_matchers {
static DeclarationMatcher functionDecl(int has)
{
    return DeclarationMatcher();
}
static StatementMatcher cudaKernelCallExpr()
{
    return StatementMatcher();
}
static int hasAttr(uint32_t attr)
{
    return 0;
}
class MatchFinder {
public:
    struct MatchResult {
        MatchResult(const BoundNodes &Nodes, clang::ASTContext *Context)
            : Nodes(Nodes),      // 初始化 const BoundNodes
              Context(Context),  // 初始化 const 指针
              SourceManager(&Context->getSourceManager())
        {}
        const BoundNodes Nodes;
        clang::ASTContext * const Context;
        clang::SourceManager * const SourceManager;
    };

    class MatchCallback {
    public:
        virtual ~MatchCallback() = default;

        /// Called on every match by the \c MatchFinder.
        virtual void run(const MatchResult &Result) = 0;

    };
    void addMatcher(const DeclarationMatcher &NodeMatch, MatchCallback *Action) {};
    void addMatcher(const StatementMatcher &NodeMatch, MatchCallback *Action) {};
    void matchAST(ASTContext &Context) {};
};
}
}
#endif