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
#include <gtest/gtest.h>
#include <unistd.h>
#include <mockcpp/mockcpp.hpp>
#include <set>
#include <string>
#define private public
#include "ascc_types.h"
#include "ascc_utils.h"
#include "ascc_ast_info_collector.h"
#include "ascc_ast_analyzer.h"
#include "ascc_argument_manager.h"
#include <clang/AST/Decl.h>

class TEST_ASCC_AST_INFO_COLLECTOR : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_KernelDeclHandlerTemplateTypeParmDecl)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    static clang::AnnotateAttr attrA;
    static clang::AnnotateAttr attrB;
    Ascc::KerenelInfoCollector collector;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    NamedDecl nd;
    std::vector<NamedDecl *> vnd = {&nd};
    TemplateParameterList tal(vnd);
    MOCKER(&clang::ASTContext::createMangleContext).stubs()
        .will(returnValue(new MangleContext()));
    MOCKER(&MangleContext::shouldMangleDeclName).stubs().will(returnValue(true));
    MOCKER(&clang::dyn_cast<clang::NamespaceDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    MOCKER(&clang::FunctionTemplateDecl::getTemplateParameters).stubs().will(returnValue(&tal));
    MOCKER(&clang::dyn_cast<clang::TemplateTypeParmDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    EXPECT_NO_THROW(collector.KernelDeclHandle(funcDecl, res));
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_KernelDeclHandlerTemplateTypeParmDecl_nullptr)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    static clang::AnnotateAttr attrA;
    static clang::AnnotateAttr attrB;
    Ascc::KerenelInfoCollector collector;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    NamedDecl nd;
    std::vector<NamedDecl *> vnd = {&nd};
    TemplateParameterList tal(vnd);
    MOCKER(&clang::ASTContext::createMangleContext).stubs()
        .will(returnValue(new MangleContext()));
    MOCKER(&MangleContext::shouldMangleDeclName).stubs().will(returnValue(true));
    MOCKER(&clang::dyn_cast<clang::NamespaceDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    MOCKER(&clang::FunctionTemplateDecl::getTemplateParameters).stubs().will(returnValue(&tal));
    MOCKER(&clang::dyn_cast<clang::TemplateTypeParmDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    Ascc::AsccInfoFunction::FunctionInfo *retNull = nullptr;
    MOCKER(&Ascc::AsccInfoFunction::GetFunctionInfo).stubs().will(returnValue(retNull));
    EXPECT_NO_THROW(collector.KernelDeclHandle(funcDecl, res));
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_KernelDeclHandlerNonTypeTemplateParmDecl)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    static clang::AnnotateAttr attrA;
    static clang::AnnotateAttr attrB;
    Ascc::KerenelInfoCollector collector;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    NamedDecl nd;
    std::vector<NamedDecl *> vnd = {&nd};
    TemplateParameterList tal(vnd);
    MOCKER(&clang::ASTContext::createMangleContext).stubs()
        .will(returnValue(new MangleContext()));
    MOCKER(&clang::dyn_cast<clang::NamespaceDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    MOCKER(&clang::FunctionTemplateDecl::getTemplateParameters).stubs().will(returnValue(&tal));
    MOCKER(&clang::dyn_cast<clang::NonTypeTemplateParmDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    EXPECT_NO_THROW(collector.KernelDeclHandle(funcDecl, res));
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_KernelDeclHandlerTemplateTemplateParmDecl)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    static clang::AnnotateAttr attrA;
    static clang::AnnotateAttr attrB;
    Ascc::KerenelInfoCollector collector;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    NamedDecl nd;
    std::vector<NamedDecl *> vnd = {&nd};
    TemplateParameterList tal(vnd);
    MOCKER(&clang::ASTContext::createMangleContext).stubs()
        .will(returnValue(new MangleContext()));
    MOCKER(&clang::dyn_cast<clang::NamespaceDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    MOCKER(&clang::FunctionTemplateDecl::getTemplateParameters).stubs().will(returnValue(&tal));
    MOCKER(&clang::FunctionDecl::getTemplateParameters).stubs().will(returnValue(&tal));
    MOCKER(&clang::dyn_cast<clang::TemplateTemplateParmDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    EXPECT_NO_THROW(collector.KernelDeclHandle(funcDecl, res));
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_KernelDeclHandlerInst)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    funcDecl->isTemplateInst = true;
    static clang::AnnotateAttr attrA;
    static clang::AnnotateAttr attrB;
    Ascc::KerenelInfoCollector collector;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    TemplateArgument taType, taInt, taDecl, taTemp;
    taType.ak = TemplateArgument::Type;
    taInt.ak = TemplateArgument::Integral;
    taDecl.ak = TemplateArgument::Declaration;
    taTemp.ak = TemplateArgument::Template;
    std::vector<TemplateArgument> vta = {taType, taInt, taDecl, taTemp};
    TemplateArgumentList tal;
    tal.vta = vta;
    FunctionTemplateSpecializationInfo ftsl;
    ftsl.tal = tal;
    ValueDecl vd;
    ValueDecl* vdp = &vd;
    const FunctionTemplateSpecializationInfo *ftslp = &ftsl;
    std::string fileName("test");
    MOCKER(&clang::ASTContext::createMangleContext).stubs().will(returnValue(new MangleContext()));
    MOCKER(&clang::dyn_cast<clang::NamespaceDecl>, clang::FunctionDecl * (const clang::DeclContext *))
        .stubs()
        .will(returnValue(funcDecl));
    MOCKER(&clang::FunctionDecl::getTemplateSpecializationInfo).stubs().will(returnValue(ftslp));
    MOCKER(&clang::TemplateArgument::getAsDecl).stubs().will(returnValue(vdp));
    MOCKER(&clang::NamedDecl::printQualifiedName).stubs().will(ignoreReturnValue());
    MOCKER(&llvm::StringRef::str).stubs().will(returnValue(fileName));
    EXPECT_NO_THROW(collector.KernelDeclHandle(funcDecl, res));
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_KernelDeclHandlerIntEnum)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    funcDecl->isTemplateInst = true;
    Ascc::KerenelInfoCollector collector;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    TemplateArgument taType, taInt, taDecl, taTemp;
    taType.ak = TemplateArgument::Type;
    taInt.ak = TemplateArgument::Integral;
    taDecl.ak = TemplateArgument::Declaration;
    taTemp.ak = TemplateArgument::Template;
    std::vector<TemplateArgument> vta = {taType, taInt, taDecl, taTemp};
    TemplateArgumentList tal;
    tal.vta = vta;
    FunctionTemplateSpecializationInfo ftsl;
    ftsl.tal = tal;
    ValueDecl vd;
    ValueDecl* vdp = &vd;
    const FunctionTemplateSpecializationInfo *ftslp = &ftsl;
    std::string fileName("test");
    MOCKER(&clang::ASTContext::createMangleContext).stubs().will(returnValue(new MangleContext()));
    MOCKER(&clang::dyn_cast<clang::NamespaceDecl>, clang::FunctionDecl * (const clang::DeclContext *))
        .stubs()
        .will(returnValue(funcDecl));
    MOCKER(&clang::FunctionDecl::getTemplateSpecializationInfo).stubs().will(returnValue(ftslp));
    MOCKER(&clang::TemplateArgument::getAsDecl).stubs().will(returnValue(vdp));
    MOCKER(&clang::NamedDecl::printQualifiedName).stubs().will(ignoreReturnValue());
    MOCKER(&Ascc::KerenelInfoCollector::GetEnumNameForValue).stubs()
            .will(returnValue(std::optional<std::string>("test_enum::test")));
    MOCKER(&llvm::StringRef::str).stubs().will(returnValue(fileName));
    EXPECT_NO_THROW(collector.KernelDeclHandle(funcDecl, res));
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_GetEnumNameForValue)
{
    clang::ASTContext context = clang::ASTContext();
    PrintingPolicy policy = context.getPrintingPolicy();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    Ascc::KerenelInfoCollector collector;
    NamedDecl nd;
    std::vector<NamedDecl *> vnd = {&nd};
    TemplateParameterList tal(vnd);
    MOCKER(&clang::FunctionTemplateDecl::getTemplateParameters).stubs().will(returnValue(&tal));
    clang::EnumType ET;
    const clang::EnumType* enumType = &ET;
    clang::EnumDecl ED;
    const clang::EnumDecl* enumDecl = &ED;
    static const clang::EnumConstantDecl ECD1, ECD2;
    static const EnumConstantDecl* enumerators[] = {&ECD1, &ECD2};
    llvm::APSInt aps;

    {
        EXPECT_EQ(collector.GetEnumNameForValue(funcDecl, 0, 0, policy), std::nullopt);
    }

    {
        EXPECT_EQ(collector.GetEnumNameForValue(funcDecl, 0, 0, policy), std::nullopt);
    }

    {
        MOCKER(&clang::dyn_cast<clang::NonTypeTemplateParmDecl>, clang::FunctionDecl *(const clang::DeclContext*))
            .stubs()
            .will(returnValue(funcDecl));
        EXPECT_EQ(collector.GetEnumNameForValue(funcDecl, 0, 0, policy), std::nullopt);
    }

    {
        const clang::EnumConstantDecl** begin_ptr = enumerators;
        const clang::EnumConstantDecl** end_ptr = enumerators + 2;

        MOCKER(&clang::dyn_cast<clang::NonTypeTemplateParmDecl>, clang::FunctionDecl *(const clang::DeclContext*))
            .stubs()
            .will(returnValue(funcDecl));
        MOCKER(&clang::QualType::getAs<clang::EnumType>).stubs().will(returnValue(enumType));
        MOCKER(&clang::EnumType::getDecl).stubs().will(returnValue(enumDecl));
        MOCKER(&clang::EnumDecl::enumerator_begin).stubs().will(returnValue(begin_ptr));
        MOCKER(&clang::EnumDecl::enumerator_end).stubs().will(returnValue(end_ptr));
        MOCKER(&clang::EnumConstantDecl::getInitVal).stubs().will(returnValue(aps));
        MOCKER(static_cast<std::string (clang::QualType::*)(const clang::PrintingPolicy) const>
            (&clang::QualType::getAsString)).stubs().will(returnValue(std::string("test_enum")));
        MOCKER(&clang::EnumConstantDecl::getNameAsString).stubs().will(returnValue(std::string("test")));

        EXPECT_EQ(collector.GetEnumNameForValue(funcDecl, 0, 0, policy), std::string("test_enum::test"));
    }

    {
        const clang::EnumConstantDecl** begin_ptr = enumerators;
        const clang::EnumConstantDecl** end_ptr = enumerators + 2;

        MOCKER(&clang::dyn_cast<clang::NonTypeTemplateParmDecl>, clang::FunctionDecl *(const clang::DeclContext*))
            .stubs()
            .will(returnValue(funcDecl));
        MOCKER(&clang::QualType::getAs<clang::EnumType>).stubs().will(returnValue(enumType));
        MOCKER(&clang::EnumType::getDecl).stubs().will(returnValue(enumDecl));
        MOCKER(&clang::EnumDecl::enumerator_begin).stubs().will(returnValue(begin_ptr));
        MOCKER(&clang::EnumDecl::enumerator_end).stubs().will(returnValue(end_ptr));
        MOCKER(&clang::EnumConstantDecl::getInitVal).stubs().will(returnValue(aps));
        MOCKER(&llvm::APSInt::getExtValue).stubs().will(returnValue(1));
        MOCKER(static_cast<std::string (clang::QualType::*)(const clang::PrintingPolicy) const>
            (&clang::QualType::getAsString)).stubs().will(returnValue(std::string("test_enum")));
        MOCKER(&clang::EnumConstantDecl::getNameAsString).stubs().will(returnValue(std::string("test")));

        EXPECT_EQ(collector.GetEnumNameForValue(funcDecl, 0, 0, policy), std::nullopt);
    }
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_KernelCallHandle)
{
    clang::ASTContext context = clang::ASTContext();
    CUDAKernelCallExpr cudakce;
    Ascc::KerenelInfoCollector collector;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::CallExpr::getDirectCallee).stubs().will(returnValue(funcDecl));
    MOCKER(&clang::ASTContext::createMangleContext).stubs()
        .will(returnValue(new MangleContext()));
    MOCKER(&MangleContext::shouldMangleDeclName).stubs().will(returnValue(true));
    EXPECT_NO_THROW(collector.KernelCallHandle(&cudakce, res));
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_KernelCallHandleNoDeclName)
{
    clang::ASTContext context = clang::ASTContext();
    CUDAKernelCallExpr cudakce;
    Ascc::KerenelInfoCollector collector;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::CallExpr::getDirectCallee).stubs().will(returnValue(funcDecl));
    MOCKER(&clang::ASTContext::createMangleContext).stubs()
        .will(returnValue(new MangleContext()));
    EXPECT_NO_THROW(collector.KernelCallHandle(&cudakce, res));
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_ASTRun)
{
    clang::ASTContext context = clang::ASTContext();
    CUDAKernelCallExpr cudakce;
    Ascc::KerenelInfoCollector collector;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    EXPECT_NO_THROW(collector.run(res));
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_HandleDiagnosticInvalid)
{
    Ascc::AsccGlobalEnvManager::ascendSlogPrintToStdout = 1;
    Ascc::AsccDiagnostic ad;
    Diagnostic diag;
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Level::Warning;
    clang::ASTContext context = clang::ASTContext();
    CUDAKernelCallExpr cudakce;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    EXPECT_NO_THROW(ad.HandleDiagnostic(level, diag));
    Ascc::AsccGlobalEnvManager::ascendSlogPrintToStdout = 0;
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_HandleDiagnosticValid)
{
    Ascc::AsccGlobalEnvManager::ascendSlogPrintToStdout = 1;
    Ascc::AsccDiagnostic ad;
    Diagnostic diag;
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Level::Warning;
    clang::ASTContext context = clang::ASTContext();
    CUDAKernelCallExpr cudakce;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::PresumedLoc::isInvalid).stubs().will(returnValue(false));
    EXPECT_NO_THROW(ad.HandleDiagnostic(level, diag));
    level = clang::DiagnosticsEngine::Level::Error;
    EXPECT_NO_THROW(ad.HandleDiagnostic(level, diag));
    level = clang::DiagnosticsEngine::Level::Fatal;
    EXPECT_NO_THROW(ad.HandleDiagnostic(level, diag));
    Ascc::AsccGlobalEnvManager::ascendSlogPrintToStdout = 0;
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_GetSourceLine)
{
    const char* cch = "test";
    Ascc::AsccDiagnostic ad;
    Diagnostic diag;
    SourceManager sm;
    SourceLocation sl;
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Level::Warning;
    clang::ASTContext context = clang::ASTContext();
    CUDAKernelCallExpr cudakce;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::SourceLocation::isValid).stubs().will(returnValue(true));
    EXPECT_NO_THROW(ad.GetSourceLine(sm, sl));
    MOCKER(&clang::FileID::isInvalid).stubs().will(returnValue(false));
    MOCKER(&clang::SLocEntry::isFile).stubs().will(returnValue(true));
    EXPECT_NO_THROW(ad.GetSourceLine(sm, sl));
    MOCKER(&llvm::StringRef::empty).stubs().will(returnValue(false));
    MOCKER(&llvm::StringRef::data).stubs().will(returnValue(cch));
    EXPECT_NO_THROW(ad.GetSourceLine(sm, sl));
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_HandleTranslationUnit)
{
    Ascc::AsccASTConsumer aac;
    Diagnostic diag;
    SourceManager sm;
    SourceLocation sl;
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Level::Warning;
    clang::ASTContext context = clang::ASTContext();
    CUDAKernelCallExpr cudakce;
    BoundNodes bn;
    MatchFinder::MatchResult res(bn, &context);
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    system("mkdir -p /tmp/ascc/auto_gen/host_files/");
    EXPECT_NO_THROW(aac.HandleTranslationUnit(context));
    system("rm -rf /tmp/ascc/");
    EXPECT_NO_THROW(aac.HandleTranslationUnit(context));
}

TEST_F(TEST_ASCC_AST_INFO_COLLECTOR, ascc_CreateASTConsumer)
{
    Ascc::AsccFrontendAction afa;
    Diagnostic diag;
    SourceManager sm;
    SourceLocation sl;
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Level::Warning;
    clang::ASTContext context = clang::ASTContext();
    CUDAKernelCallExpr cudakce;
    BoundNodes bn;
    clang::CompilerInstance compiler = clang::CompilerInstance();
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    auto res = afa.CreateASTConsumer(compiler, llvm::StringRef("test"));
    EXPECT_TRUE(res != nullptr);
}