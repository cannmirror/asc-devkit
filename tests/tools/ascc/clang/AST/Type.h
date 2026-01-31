/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef LLVM_CLANG_AST_TYPE_H
#define LLVM_CLANG_AST_TYPE_H
#include "clang/AST/PrettyPrinter.h"
#include "llvm/ADT/APSInt.h"
namespace clang {

class EnumDecl;
class TagDecl;
class EnumConstantDecl;

class TagType {
public:
    TagDecl *getDecl() const {};
};
class EnumType : public TagType {
public:
    EnumType() = default;
    const EnumDecl *getDecl() const { return nullptr; };
};

class QualType {
public:
    QualType() = default;
    std::string getAsString() const {
        return std::string();
    }
    std::string getAsString(PrintingPolicy policy) const {
        return std::string();
    }
    QualType getNonReferenceType() const
    {
        return *this;
    }
    QualType getUnqualifiedType() const
    {
        return *this;
    }
    template <typename T = EnumType>
    const T *getAs() const { return nullptr; };
    const QualType& getCanonicalType() const { return *this; }
    bool isPointerType() const { return true; }
    QualType* operator->() {
        return this; // 返回指针或可递归访问的对象
    }
};
}
#endif