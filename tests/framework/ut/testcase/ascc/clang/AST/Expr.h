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
#ifndef CLANG_AST_EXPR_H
#define CLANG_AST_EXPR_H

#include "clang/Basic/SourceLocation.h"
#include "clang/AST/Decl.h"
namespace clang {
class Expr : public Stmt {
public:
    SourceLocation getExprLoc();
    bool isValueDependent() const { return false;}
    Expr *IgnoreImpCasts() {}
    Expr *IgnoreParens() {}
};

class CallExpr : public Expr {
public:
    FunctionDecl *getDirectCallee() const { return nullptr; }
    SourceLocation getBeginLoc() const { return sLoc; }
    SourceLocation getEndLoc() const;
    Expr *getCallee() { return &expr; }
private:
    Expr expr;
    SourceLocation sLoc;
};
}
#endif // CLANG_AST_EXPR_H
