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
#ifndef LLVM_CLANG_ASTMATCHERS_ASTMATCHERS_H
#define LLVM_CLANG_ASTMATCHERS_ASTMATCHERS_H
#include "llvm/ADT/StringRef.h"
namespace clang {
namespace attr {
enum Kind {
    CUDAGlobal = 0
};
}
namespace ast_matchers {
class DeclarationMatcher {
    int matchKind;
public:
    DeclarationMatcher bind(const char *str) { return *this; }
};
class StatementMatcher {
    int matchKind;
public:
    StatementMatcher bind(const char *str) { return *this; }
};

}
class FunctionDecl;
class CUDAKernelCallExpr;
class BoundNodes {
public:
    template <typename T>
    const T *getNodeAs(llvm::StringRef ID) const { return nullptr; };
};
}
#endif