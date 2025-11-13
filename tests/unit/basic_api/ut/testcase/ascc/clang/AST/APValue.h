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
#ifndef CLANG_AST_APVALUE_H
#define CLANG_AST_APVALUE_H

#include "clang/AST/ExprCXX.h"
#include "clang/AST/Decl.h"

namespace clang {
template <class T = UnresolvedLookupExpr>
UnresolvedLookupExpr* dyn_cast(Expr* expr) { return nullptr;}

template <class T = DeclRefExpr>
DeclRefExpr* dyn_cast_or_null(Expr * expr) {}

template <class T = NamespaceDecl>
NamespaceDecl* dyn_cast(NamespaceDecl * decl) {}

template <typename T>
clang::FunctionDecl *dyn_cast(const clang::DeclContext *) { return nullptr; }
}

#endif