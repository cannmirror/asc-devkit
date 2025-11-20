/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef CLANG_AST_DECL_H
#define CLANG_AST_DECL_H
#include <string>
#include <vector>
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/ASTContextAllocate.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
class alignas(4) Decl {
public:
    Decl() = default;
    SourceLocation getLocation() const { return Loc; }
    void dump(llvm::raw_ostream &Out, bool Deserialize = false) const {}
    SourceLocation Loc;
};

class alignas(4) DeclContext {
public:
    DeclContext() = default;
    DeclContext(const DeclContext&) = delete;
    DeclContext *getParent() const { return nullptr; }
};

class alignas(4) NamedDecl : public DeclContext, public Decl {
public:
    NamedDecl() = default;
    NamedDecl(const NamedDecl&) = delete;
    std::string getNameAsString() const { return ""; }
    std::string getQualifiedNameAsString() { return ""; }
    std::vector<clang::NamedDecl*> decls() { return {}; }
    void printQualifiedName(llvm::raw_ostream &OS, const PrintingPolicy &P) const {}
    llvm::StringRef getName() const {
        return llvm::StringRef("");
    }
};

class alignas(4) ValueDecl : public NamedDecl {
public:
    ValueDecl() = default;
};

class alignas(4) DeclaratorDecl : public ValueDecl {
public:
    DeclaratorDecl() = default;
};

class AnnotateAttr {
public:
    AnnotateAttr() {}
    AnnotateAttr(const AnnotateAttr&) = delete;
    llvm::StringRef getAnnotation() const { return llvm::StringRef(""); }
};

class Expr;
class VarDecl {
public:
    VarDecl() = default;
    QualType getType() const { return qt; }
    llvm::StringRef getName() const { return llvm::StringRef(""); }
    DeclContext *getDeclContext() { return &declContext; }
    const DeclContext *getDeclContext() const { return &declContext; }
    SourceLocation getLocation() const { return sLoc; }
    Expr *getInit() const { return nullptr; }
    ASTContext getASTContext() const { return Ctx; }
private:
    DeclContext declContext;
    SourceLocation sLoc;
    ASTContext Ctx;
    QualType qt;
};

class ParmVarDecl : public VarDecl {
public:
    ParmVarDecl() = default;
    std::string getNameAsString() const { return str; }
    const ParmVarDecl& getCanonicalType() const { return *this; }
public:
    std::string str;
};

class FunctionDecl : public DeclaratorDecl {
public:
    FunctionDecl(ASTContext &ctx) : Ctx(ctx) {
        pv.push_back(&pvd);
        ftdp = &ftd;
        tplp = &tpl;
    }
    FunctionDecl(const FunctionDecl&) = delete;
    std::vector<clang::AnnotateAttr*> attrs() const { return {}; }
    ASTContext &getASTContext() const { return Ctx; }
    QualType getReturnType() const {
        return qt;
    }
    SourceLocation getEndLoc() const {
        return Loc;
    }
    bool isTemplateInstantiation() const { return isTemplateInst; }
    bool isFunctionTemplateSpecialization() const { return isTemplateSpec; }
    std::vector<ParmVarDecl *> parameters() const {
        return pv;
    }
    const FunctionTemplateSpecializationInfo *getTemplateSpecializationInfo() const
    {
        return &ftsi;
    }
    FunctionTemplateDecl *getPrimaryTemplate() const
    {
        return ftdp;
    }
    FunctionTemplateDecl *getDescribedFunctionTemplate() const {
        return ftdp;
    }
    QualType getType() const {
        return qt;
    }
    TemplateParameterList *getTemplateParameters() const {
        return tplp;
    }
    const DeclContext *getDeclContext() const {
        return &dc;
    }
    bool isAnonymousNamespace() const { return false; }
    SourceRange getSourceRange() const { return range; }
public:
    std::vector<ParmVarDecl *> pv;
    FunctionTemplateSpecializationInfo ftsi;
    ParmVarDecl pvd;
    QualType qt;
    ASTContext &Ctx;
    FunctionTemplateDecl ftd;
    FunctionTemplateDecl* ftdp;
    TemplateParameterList tpl;
    TemplateParameterList* tplp;
    DeclContext dc;
    SourceRange range;
    bool isTemplateInst = false;
    bool isTemplateSpec = false;
};

class alignas(4) TranslationUnitDecl : public Decl {
public:
    TranslationUnitDecl() = default;
};

class NamespaceDecl {
public:
    std::string getName() const {return "";};
};

class DeclRefExpr {
public:
    DeclRefExpr() = default;
    clang::NamedDecl *getDecl() const { return nullptr; }
};

class TagDecl {
public:
    TagDecl() = default;
    TagDecl *getDefinition() const {};
};

class EnumDecl : public TagDecl {
public:
    EnumDecl() = default;
    using enumerator_iterator = const EnumConstantDecl**;
    EnumDecl *getDefinition() const {
    }
    enumerator_iterator enumerator_begin() const {
        return enumerator_iterator();
    }
    enumerator_iterator enumerator_end() const {
        return enumerator_iterator();
    }
};

class EnumConstantDecl {
public:
    const llvm::APSInt &getInitVal() const {
        return llvm::APSInt();
    }

    std::string getNameAsString() const {
        return std::string();
    }
};
}
#endif
