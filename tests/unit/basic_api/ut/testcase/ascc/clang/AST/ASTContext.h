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
#ifndef CLANG_AST_ASTCONTEXT
#define CLANG_AST_ASTCONTEXT

#include "clang/Basic/SourceManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/PrettyPrinter.h"

namespace clang {
class MangleContext;
class TranslationUnitDecl;
class ASTContext {
public:
    SourceManager& getSourceManager() { return SourceMgr; }
    const SourceManager& getSourceManager() const { return SourceMgr; }
    MangleContext *createMangleContext() { return nullptr; };
    TranslationUnitDecl *getTranslationUnitDecl() const { return TUDecl; }
    const LangOptions& getLangOpts() const { return LangOpts; }
    const TargetInfo &getTargetInfo() const { return *Target; }
    const clang::PrintingPolicy &getPrintingPolicy() const {
        return PrintingPolicy;
    }
    void setPrintingPolicy(const clang::PrintingPolicy &Policy) { PrintingPolicy = Policy; }
private:
    TranslationUnitDecl *TUDecl;
    const TargetInfo *Target = nullptr;
    clang::PrintingPolicy PrintingPolicy;
    SourceManager SourceMgr;
    LangOptions LangOpts;
};
}
#endif