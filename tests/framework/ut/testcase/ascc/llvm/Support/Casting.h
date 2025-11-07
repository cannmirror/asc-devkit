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
#ifndef LLVM_SUPPORT_CASTING_H
#define LLVM_SUPPORT_CASTING_H
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"

namespace llvm {
template <typename T>
clang::NamedDecl *dyn_cast(clang::Expr *) {}

template <typename T>
clang::AnnotateAttr *dyn_cast(const clang::AnnotateAttr *) {}

template <typename T>
clang::FunctionTemplateDecl *dyn_cast_or_null(const clang::NamedDecl *) {}

template <typename T>
clang::TemplateTypeParmDecl *dyn_cast(clang::NamedDecl *) {}
}
template <typename To>
inline bool isa(const clang::DeclContext *dc) {
  return false;
}

#endif