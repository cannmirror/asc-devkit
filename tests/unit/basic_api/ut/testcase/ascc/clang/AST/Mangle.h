/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef CLANG_AST_MANGLE_H
#define CLANG_AST_MANGLE_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "llvm/Support/raw_ostream.h"
namespace clang {
class MangleContext {
public:
    bool shouldMangleDeclName(const NamedDecl *D) { return false; }
    void mangleName(const NamedDecl *D, llvm::raw_ostream &Out) { return; }
    void mangleName(const FunctionTemplateDecl *D, llvm::raw_ostream &Out) { return; }
    bool shouldMangleCXXName(const NamedDecl *D);
    bool shouldMangleCXXName(const FunctionTemplateDecl *D);
};
}
#endif