/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef LLVM_CLANG_AST_ASTCONTEXTALLOCATE_H
#define LLVM_CLANG_AST_ASTCONTEXTALLOCATE_H

#include <cstddef>

namespace clang {

class ASTContext;

} // namespace clang

// Defined in ASTContext.h
void *operator new(size_t Bytes, const clang::ASTContext &C,
                   size_t Alignment = 8);
void *operator new[](size_t Bytes, const clang::ASTContext &C,
                     size_t Alignment = 8);

// It is good practice to pair new/delete operators.  Also, MSVC gives many
// warnings if a matching delete overload is not declared, even though the
// throw() spec guarantees it will not be implicitly called.
void operator delete(void *Ptr, const clang::ASTContext &C, size_t);
void operator delete[](void *Ptr, const clang::ASTContext &C, size_t);

#endif // LLVM_CLANG_AST_ASTCONTEXTALLOCATE_H
