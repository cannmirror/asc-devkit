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
#ifndef CLANG_AST_DECLTEMPLATE_H
#define CLANG_AST_DECLTEMPLATE_H
#include <string>
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include "clang/AST/TemplateName.h"
#include "llvm/ADT/APSInt.h"

namespace clang {
/// A template argument list.
class ValueDecl;
class NamedDecl;
class TemplateArgument {
public:
    enum ArgKind {
        /// Represents an empty template argument, e.g., one that has not
        /// been deduced.
        Null = 0,

        /// The template argument is a type.
        Type,

        /// The template argument is a declaration that was provided for a pointer,
        /// reference, or pointer to member non-type template parameter.
        Declaration,

        /// The template argument is a null pointer or null pointer to member that
        /// was provided for a non-type template parameter.
        NullPtr,

        /// The template argument is an integral value stored in an llvm::APSInt
        /// that was provided for an integral non-type template parameter.
        Integral,

        /// The template argument is a template name that was provided for a
        /// template template parameter.
        Template,

        /// The template argument is a pack expansion of a template name that was
        /// provided for a template template parameter.
        TemplateExpansion,

        /// The template argument is an expression, and we've not resolved it to one
        /// of the other forms yet, either because it's dependent or because we're
        /// representing a non-canonical template argument (for instance, in a
        /// TemplateSpecializationType).
        Expression,

        /// The template argument is actually a parameter pack. Arguments are stored
        /// in the Args struct.
        Pack
    };
    TemplateArgument() = default;
    ArgKind getKind() const {
        return ak;
    }
    QualType getAsType() const {
        return qt;
    }
    llvm::APSInt getAsIntegral() const {
        return apsi;
    }
    ValueDecl *getAsDecl() const {
        return vdp;
    }
    TemplateName getAsTemplate() const {
        return tn;
    }
public:
    ArgKind ak;
    QualType qt;
    llvm::APSInt apsi;
    ValueDecl *vdp;
    TemplateName tn;
};

class TemplateArgumentList {
public:
    const TemplateArgument &operator[](unsigned Idx) const { return vta[Idx]; }
    size_t size() const { return vta.size(); }
    std::vector<TemplateArgument> vta;
};

class TemplateParameterList {
public:
    TemplateParameterList(std::vector<NamedDecl *> vnd_) : vnd(vnd_) {}
    TemplateParameterList() = default;
    size_t size() const { return vnd.size(); }
    const NamedDecl *getParam(unsigned int Idx) const { return vnd[Idx]; }
    std::vector<NamedDecl *> vnd;
};

class TemplateTypeParmDecl {
public:
    TemplateTypeParmDecl() = default;
};

class FunctionTemplateDecl {
public:
    FunctionTemplateDecl() { this->tplp = &tpl; }
    std::string getQualifiedNameAsString() {
        return "";
    }
    std::string getNameAsString() {
        return "";
    }
    SourceRange getSourceRange() const {
        return range;
    }
    std::vector<FunctionTemplateDecl*> specializations() { return {}; }
    SourceLocation getPointOfInstantiation() { return SourceLocation(); }
    FunctionTemplateDecl *getAsFunction() { return nullptr; }
    ASTContext &getASTContext() {}
    TemplateParameterList *getTemplateParameters() const {
        return tplp;
    }
public:
    TemplateParameterList* tplp;
    TemplateParameterList tpl;
    SourceRange range;
};

class FunctionTemplateSpecializationInfo {
public:
    FunctionTemplateSpecializationInfo() { this->TemplateArguments = &tal; };
    const FunctionTemplateDecl *getTemplate() const { return &ftd; }
public:
    FunctionTemplateDecl ftd;
    const TemplateArgumentList* TemplateArguments;
    TemplateArgumentList tal = TemplateArgumentList();
};

class NonTypeTemplateParmDecl {
  // FIXME: Collapse this into TemplateParamPosition; or, just move depth/index
  // down here to save memory.

  /// Whether this non-type template parameter is a parameter pack.
  bool ParameterPack;

  /// Whether this non-type template parameter is an "expanded"
  /// parameter pack, meaning that its type is a pack expansion and we
  /// already know the set of types that expansion expands to.
  bool ExpandedParameterPack = false;

  /// The number of types in an expanded parameter pack.
  unsigned NumExpandedTypes = 0;
};
class TemplateTemplateParmDecl {
public:
    TemplateTemplateParmDecl() = default;
};
}
#endif