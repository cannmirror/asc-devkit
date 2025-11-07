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
#ifndef CLANG_AST_ASTCONSUMER_H
#define CLANG_AST_ASTCONSUMER_H

// here is some headfile not find entry
#include "clang/AST/Stmt.h"
#include "clang/AST/APValue.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/FileManager.h"
#include "clang/AST/DeclTemplate.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace clang {
class ASTConsumer {
    class ASTContext;
public:
    ASTConsumer() = default;
    virtual ~ASTConsumer() = default;
    virtual void HandleTranslationUnit(clang::ASTContext &Ctx) {}
};
}

#endif // CLANG_AST_ASTCONSUMER_H