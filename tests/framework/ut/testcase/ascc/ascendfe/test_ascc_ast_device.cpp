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
#include "ascc_ast_utils.h"
#include "ascc_ast_device_analyzer.h"
#include "ascc_ast_device_consumer.h"
#include "ascc_argument_manager.h"
#include "ascc_global_env_manager.h"
#include <clang/AST/Decl.h>

class TEST_ASCC_AST_DEVICE : public testing::Test {
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

namespace Ascc {
    extern void SetFuncFileKernelHasKfc(const clang::FunctionDecl *funcDecl, bool hasKfc);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitFunctionDecl)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    static clang::AnnotateAttr attrA;
    static clang::AnnotateAttr attrB;
    MOCKER(&clang::FunctionDecl::attrs).stubs().will(returnValue(
        std::vector<clang::AnnotateAttr*>{&attrA, &attrB}
    ));
    static clang::AnnotateAttr annotateA;
    static clang::AnnotateAttr annotateB;
    clang::AnnotateAttr *annotatePtrA = &annotateA;
    MOCKER(&clang::AnnotateAttr::getAnnotation).stubs()
        .will(returnValue(llvm::StringRef("global")))
        .then(returnValue(llvm::StringRef("device")));
    MOCKER(&llvm::dyn_cast<clang::AnnotateAttr>, clang::AnnotateAttr *(const clang::AnnotateAttr*))
        .stubs()
        .will(returnValue(annotatePtrA));
    MOCKER(&clang::FunctionDecl::getName).stubs().will(returnValue(llvm::StringRef("not empty")));
    EXPECT_EQ(visitor.VisitFunctionDecl(funcDecl), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitFunctionDecl_empty)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    static clang::AnnotateAttr attrA;
    static clang::AnnotateAttr attrB;
    MOCKER(&clang::FunctionDecl::attrs).stubs().will(returnValue(
        std::vector<clang::AnnotateAttr*>{&attrA, &attrB}
    ));
    static clang::AnnotateAttr annotateA;
    static clang::AnnotateAttr annotateB;
    clang::AnnotateAttr *annotatePtrA = &annotateA;
    MOCKER(&clang::AnnotateAttr::getAnnotation).stubs()
        .will(returnValue(llvm::StringRef("global")))
        .then(returnValue(llvm::StringRef("device")));
    MOCKER(&llvm::dyn_cast<clang::AnnotateAttr>, clang::AnnotateAttr *(const clang::AnnotateAttr*))
        .stubs()
        .will(returnValue(annotatePtrA));
    EXPECT_EQ(visitor.VisitFunctionDecl(funcDecl), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitCallExpr_nullptr)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::CallExpr *callExpr = nullptr;
    EXPECT_EQ(visitor.VisitCallExpr(callExpr), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitCallExpr_FunctionDecl_printf)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::CallExpr CE;
    clang::CallExpr *callExpr = &CE;
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::CallExpr::getDirectCallee).stubs().will(returnValue(funcDecl));
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.ascendCannPackagePath = "llt_cann_stub_path";
    MOCKER(&clang::FunctionDecl::getQualifiedNameAsString).stubs()
        .will(returnValue(std::string("AscendC::printf")));
    EXPECT_EQ(visitor.VisitCallExpr(callExpr), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitCallExpr_FunctionDecl_assert)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::CallExpr CE;
    clang::CallExpr *callExpr = &CE;
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::CallExpr::getDirectCallee).stubs().will(returnValue(funcDecl));
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.ascendCannPackagePath = "llt_cann_stub_path";
    MOCKER(&clang::FunctionDecl::getQualifiedNameAsString).stubs()
        .will(returnValue(std::string("AscendC::AssertImpl")));
    EXPECT_EQ(visitor.VisitCallExpr(callExpr), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitCallExpr_UnresolvedLookupExpr_printf)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::CallExpr CE;
    clang::CallExpr *callExpr = &CE;
    clang::UnresolvedLookupExpr ULE;
    clang::UnresolvedLookupExpr *ule = &ULE;
    MOCKER(&clang::dyn_cast<clang::UnresolvedLookupExpr>, clang::UnresolvedLookupExpr *(clang::Expr*))
        .stubs()
        .will(returnValue(ule));
    MOCKER(&clang::DeclarationName::getAsString).stubs().will(returnValue(std::string("printf")));
    clang::NamespaceDecl ND;
    clang::NamespaceDecl *decl = &ND;
    MOCKER(&clang::dyn_cast<clang::NamespaceDecl>, clang::NamespaceDecl *(clang::NamespaceDecl*))
        .stubs()
        .will(returnValue(decl));
    MOCKER(&clang::NamespaceDecl::getName).stubs().will(returnValue(std::string("AscendC")));
    EXPECT_EQ(visitor.VisitCallExpr(callExpr), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitCallExpr_UnresolvedLookupExpr_AssertImpl)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::CallExpr CE;
    clang::CallExpr *callExpr = &CE;
    clang::UnresolvedLookupExpr ULE;
    clang::UnresolvedLookupExpr *ule = &ULE;
    MOCKER(&clang::dyn_cast<clang::UnresolvedLookupExpr>, clang::UnresolvedLookupExpr *(clang::Expr*))
        .stubs()
        .will(returnValue(ule));
    MOCKER(&clang::DeclarationName::getAsString).stubs().will(returnValue(std::string("AssertImpl")));
    clang::NamespaceDecl ND;
    clang::NamespaceDecl *decl = &ND;
    MOCKER(&clang::dyn_cast<clang::NamespaceDecl>, clang::NamespaceDecl *(clang::NamespaceDecl*))
        .stubs()
        .will(returnValue(decl));
    MOCKER(&clang::NamespaceDecl::getName).stubs().will(returnValue(std::string("AscendC")));
    EXPECT_EQ(visitor.VisitCallExpr(callExpr), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitCallExpr_unknown)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::CallExpr CE;
    clang::CallExpr *callExpr = &CE;
    EXPECT_EQ(visitor.VisitCallExpr(callExpr), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_GetLineNumber)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::Stmt S;
    clang::Stmt *stmt = &S;
    EXPECT_EQ(visitor.GetLineNumber(stmt), 0u);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitVarDecl_nullptr)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl *varDecl = nullptr;
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitVarDecl_unmatch)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl VD;
    clang::VarDecl *varDecl = &VD;
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitVarDecl)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl VD;
    clang::VarDecl *varDecl = &VD;
    MOCKER(&clang::VarDecl::getName).stubs()
        .will(returnValue(llvm::StringRef("__enable_feature_for_compile_default")));
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::dyn_cast<clang::FunctionDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    visitor.globalAicoreFuncsKernelType_[funcDecl] = "not empty";
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);

    MOCKER(&clang::SourceLocation::isMacroID).stubs().will(returnValue(true));
    clang::Expr E;
    clang::Expr *expr = &E;
    MOCKER(&clang::VarDecl::getInit).stubs().will(returnValue(expr));
    MOCKER(&clang::Expr::IgnoreImpCasts).stubs().will(returnValue(expr));
    MOCKER(&clang::Expr::IgnoreParens).stubs().will(returnValue(expr));
    clang::DeclRefExpr DRE;
    clang::DeclRefExpr *declRefExpr = &DRE;
    MOCKER(&clang::dyn_cast_or_null<clang::DeclRefExpr>, clang::DeclRefExpr *(clang::Expr*))
        .stubs()
        .will(returnValue(declRefExpr));
    visitor.globalAicoreFuncsKernelType_[funcDecl] = "";
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitVarDecl_updateInfo)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl VD;
    clang::VarDecl *varDecl = &VD;
    MOCKER(&clang::VarDecl::getName).stubs().will(returnValue(llvm::StringRef("MIX_NUM")));
    MOCKER(Ascc::IsVarInNamespace).stubs().will(returnValue(true));
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::dyn_cast<clang::FunctionDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    visitor.globalAicoreFuncsKernelType_[funcDecl] = "not empty";
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitVarDecl_updateInfo_dump_workspace)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl VD;
    clang::VarDecl *varDecl = &VD;
    MOCKER(&clang::VarDecl::getName).stubs().will(returnValue(llvm::StringRef("DUMP_WORKSPACE_SIZE")));
    MOCKER(Ascc::IsVarInNamespace).stubs().will(returnValue(true));
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::dyn_cast<clang::FunctionDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    visitor.globalAicoreFuncsKernelType_[funcDecl] = "not empty";
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitVarDecl_updateInfo_dump_uintsize)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl VD;
    clang::VarDecl *varDecl = &VD;
    MOCKER(&clang::VarDecl::getName).stubs().will(returnValue(llvm::StringRef("DUMP_UINTSIZE")));
    MOCKER(Ascc::IsVarInNamespace).stubs().will(returnValue(true));
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::dyn_cast<clang::FunctionDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    visitor.globalAicoreFuncsKernelType_[funcDecl] = "not empty";
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitVarDecl_updateInfo_kfcserver)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl VD;
    clang::VarDecl *varDecl = &VD;
    std::string varName = "AscendC::KfcServer";
    MOCKER(Ascc::FindVarTypeStr).stubs().will(returnValue(varName));
    MOCKER(Ascc::IsVarInNamespace).stubs().will(returnValue(true));
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::dyn_cast<clang::FunctionDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    visitor.globalAicoreFuncsKernelType_[funcDecl] = "not empty";
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitVarDecl_updateInfo_withMangledName)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl VD;
    clang::VarDecl *varDecl = &VD;
    MOCKER(&clang::VarDecl::getName).stubs()
        .will(returnValue(llvm::StringRef("__enable_feature_for_compile_default")));
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::dyn_cast<clang::FunctionDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    visitor.globalAicoreFuncsKernelType_[funcDecl] = "not empty";
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);

    MOCKER(&clang::SourceLocation::isMacroID).stubs().will(returnValue(true));
    clang::Expr E;
    clang::Expr *expr = &E;
    MOCKER(&clang::VarDecl::getInit).stubs().will(returnValue(expr));
    MOCKER(&clang::Expr::IgnoreImpCasts).stubs().will(returnValue(expr));
    MOCKER(&clang::Expr::IgnoreParens).stubs().will(returnValue(expr));
    clang::DeclRefExpr DRE;
    clang::DeclRefExpr *declRefExpr = &DRE;
    MOCKER(&clang::dyn_cast_or_null<clang::DeclRefExpr>, clang::DeclRefExpr *(clang::Expr*))
        .stubs()
        .will(returnValue(declRefExpr));
    visitor.globalAicoreFuncsKernelType_[funcDecl] = "";
    MOCKER(&clang::CompilerInstance::getASTContext).stubs().will(returnValue(context));
    MOCKER(&clang::ASTContext::createMangleContext).stubs().will(returnValue(new MangleContext()));
    MOCKER(&MangleContext::shouldMangleDeclName).stubs().will(returnValue(true));
    MOCKER(&clang::VarDecl::getInit).stubs().will(returnValue(expr));
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_VisitVarDecl_updateInfo_withoutMangledName)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl VD;
    clang::VarDecl *varDecl = &VD;
    MOCKER(&clang::VarDecl::getName).stubs()
        .will(returnValue(llvm::StringRef("__enable_feature_for_compile_default")));
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::dyn_cast<clang::FunctionDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    visitor.globalAicoreFuncsKernelType_[funcDecl] = "not empty";
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);

    MOCKER(&clang::SourceLocation::isMacroID).stubs().will(returnValue(true));
    clang::Expr E;
    clang::Expr *expr = &E;
    MOCKER(&clang::VarDecl::getInit).stubs().will(returnValue(expr));
    MOCKER(&clang::Expr::IgnoreImpCasts).stubs().will(returnValue(expr));
    MOCKER(&clang::Expr::IgnoreParens).stubs().will(returnValue(expr));
    clang::DeclRefExpr DRE;
    clang::DeclRefExpr *declRefExpr = &DRE;
    MOCKER(&clang::dyn_cast_or_null<clang::DeclRefExpr>, clang::DeclRefExpr *(clang::Expr*))
        .stubs()
        .will(returnValue(declRefExpr));
    visitor.globalAicoreFuncsKernelType_[funcDecl] = "";
    MOCKER(&clang::CompilerInstance::getASTContext).stubs().will(returnValue(context));
    MOCKER(&clang::ASTContext::createMangleContext).stubs().will(returnValue(new MangleContext()));
    MOCKER(&clang::MangleContext::shouldMangleDeclName).stubs().will(returnValue(false));
    MOCKER(&clang::VarDecl::getInit).stubs().will(returnValue(expr));
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_HandleTranslationUnit)
{
    clang::ASTContext context;
    clang::CompilerInstance compiler;
    Ascc::ASTDeviceConsumer consumer(context, compiler);
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    envVar.ascendSlogPrintToStdout = 1;
    envVar.ascendGlobalLogLevel = 0;
    system("mkdir -p /tmp/ascc/auto_gen/host_files/");
    EXPECT_NO_THROW(consumer.HandleTranslationUnit(context));
    envVar.ascendSlogPrintToStdout = 0;
    system("rm -rf /tmp/ascc/");
    EXPECT_NO_THROW(consumer.HandleTranslationUnit(context));
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_AstDeviceAnalyzer_Process)
{
    system("touch test.cpp");
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    argManager.inputFile_ = "test.cpp";
    std::string astFile = "test.cpp";
    Ascc::AsccAstDeviceAnalyzer deviceAnalyzer(astFile);
    deviceAnalyzer.Process();
    system("rm -rf test.cpp");
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_SetFuncFileKernelType)
{
    std::string filename = "test";
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::FunctionDecl FD(context);
    Ascc::AsccInfoStorage &storage = Ascc::AsccInfoStorage::GetInstance();
    std::shared_ptr<Ascc::AsccInfoFunction> normalFuncInfo = std::make_shared<Ascc::AsccInfoFunction>();
    Ascc::AsccInfoFunction::FunctionInfo funcInfoStruct;
    funcInfoStruct.returnType = "void";
    funcInfoStruct.funcName = "Foo";
    funcInfoStruct.definitionPos = "1";
    funcInfoStruct.kernelType = Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_2;
    funcInfoStruct.nameSpace = "AsccTestSSSS";
    funcInfoStruct.lineNo = 1;
    normalFuncInfo->AddFunction(":0", funcInfoStruct);
    system("touch test.cpp");
    storage.AddInfo("test", Ascc::AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION, normalFuncInfo);
    MOCKER(&llvm::StringRef::str).stubs().will(returnValue(filename));
    EXPECT_NO_THROW(Ascc::SetFuncFileKernelType(&FD, "KERNEL_TYPE_MIX_AIC_1_1"));
    EXPECT_NO_THROW(Ascc::SetFuncFileKernelType(&FD, "KERNEL_TYPE_MIX_AIC_1_3"));
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_SetFuncFileKernelHasKfc)
{
    std::string filename = "test";
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    Ascc::ASTDeviceVisitor visitor(context, compiler);
    clang::FunctionDecl FD(context);
    Ascc::AsccInfoStorage &storage = Ascc::AsccInfoStorage::GetInstance();
    std::shared_ptr<Ascc::AsccInfoFunction> normalFuncInfo = std::make_shared<Ascc::AsccInfoFunction>();
    Ascc::AsccInfoFunction::FunctionInfo funcInfoStruct2;
    std::vector<Ascc::AsccInfoFunction::ParameterInfo> tmpParams;
    tmpParams.emplace_back("T", "typename", false, Ascc::ParamType::TEMPLATE_TYPE);
    funcInfoStruct2.returnType = "void";
    funcInfoStruct2.funcName = "Foo";
    funcInfoStruct2.definitionPos = "1";
    funcInfoStruct2.kernelType = Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_2;
    funcInfoStruct2.nameSpace = "AsccTestFFFFF";
    funcInfoStruct2.lineNo = 1;
    Ascc::AsccInfoFunction::ParameterInfo param("x", "int", true, Ascc::ParamType::NORMAL_INPUT);
    funcInfoStruct2.params.emplace_back(param);
    funcInfoStruct2.templateParams = tmpParams;

    Ascc::AsccInfoFunction::FunctionInfo funcInfoStruct;
    funcInfoStruct.returnType = "void";
    funcInfoStruct.funcName = "Foo";
    funcInfoStruct.definitionPos = "1";
    funcInfoStruct.kernelType = Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_2;
    funcInfoStruct.nameSpace = "AsccTestsssssss";
    funcInfoStruct.lineNo = 1;
    funcInfoStruct.params.emplace_back(param);
    funcInfoStruct.templateParams = tmpParams;
    funcInfoStruct.mangledToInstFuncInfo["sfsfsf"] = std::make_shared<Ascc::AsccInfoFunction::FunctionInfo>(funcInfoStruct2);
    normalFuncInfo->AddFunction(":0", funcInfoStruct);
    system("touch test.cpp");
    storage.AddInfo("test", Ascc::AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION, normalFuncInfo);
    MOCKER(&llvm::StringRef::str).stubs().will(returnValue(filename));
    EXPECT_NO_THROW(Ascc::SetFuncFileKernelHasKfc(&FD, true));
}

TEST_F(TEST_ASCC_AST_DEVICE, ascc_CreateASTConsumer)
{
    Ascc::DeviceAnalyzeAction daa;
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
    HeaderSearchOptions hso;
    MOCKER(&clang::CompilerInstance::getHeaderSearchOpts).stubs().will(returnValue(hso));
    MOCKER(&clang::CompilerInstance::getASTContext).stubs().will(returnValue(context));
    auto res = daa.CreateASTConsumer(compiler, llvm::StringRef("test"));
    EXPECT_TRUE(res != nullptr);
}