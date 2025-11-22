/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <string>
#include <iostream>
#include <gtest/gtest.h>
#include <unistd.h>
#include <mockcpp/mockcpp.hpp>
#include <nlohmann/json.hpp>
#define private public
#include "asc_utils.h"
#include "asc_log.h"
#include "asc_info_manager.h"
#include "asc_interface.h"
#include "asc_ast_device_consumer.h"
#include "asc_struct.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Hashing.h>

class TEST_ASC_DEVICE_CONSUMER : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

// Note: should be the first testcase in this .cpp
TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_VisitCallExpr_nullptr)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
    clang::CallExpr *callExpr = nullptr;
    EXPECT_EQ(visitor.VisitCallExpr(callExpr), true);
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_VisitCallExpr_FunctionDecl_printf)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
    clang::CallExpr CE;
    clang::CallExpr *callExpr = &CE;
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::CallExpr::getDirectCallee).stubs().will(returnValue(funcDecl));
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.pathInfo_.cannPath = "llt_cann_stub_path";
    EXPECT_EQ(manager.hasPrintf_, false);
    MOCKER(&clang::FunctionDecl::getQualifiedNameAsString).stubs()
        .will(returnValue(std::string("AscendC::printf")));
    EXPECT_EQ(visitor.VisitCallExpr(callExpr), true);
    EXPECT_EQ(manager.hasPrintf_, true);
    manager.hasPrintf_ = false;
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_VisitCallExpr_FunctionDecl_assert)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
    clang::CallExpr CE;
    clang::CallExpr *callExpr = &CE;
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::CallExpr::getDirectCallee).stubs().will(returnValue(funcDecl));
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.pathInfo_.cannPath = "llt_cann_stub_path";
    EXPECT_EQ(manager.hasAssert_, false);
    MOCKER(&clang::FunctionDecl::getQualifiedNameAsString).stubs()
        .will(returnValue(std::string("AscendC::AssertImpl")));
    EXPECT_EQ(visitor.VisitCallExpr(callExpr), true);
    EXPECT_EQ(manager.hasAssert_, true);
    manager.hasAssert_ = false;
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_VisitCallExpr_UnresolvedLookupExpr_printf)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
    clang::CallExpr CE;
    clang::CallExpr *callExpr = &CE;
    clang::UnresolvedLookupExpr ULE;
    clang::UnresolvedLookupExpr *ule = &ULE;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    EXPECT_EQ(manager.hasPrintf_, false);
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
    EXPECT_EQ(manager.hasPrintf_, true);
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetLineNumber)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
    clang::Stmt S;
    clang::Stmt *stmt = &S;
    EXPECT_EQ(visitor.GetLineNumber(stmt), 0u);
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_VisitCallExpr_FunctionDecl_assertPrint)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
    clang::CallExpr CE;
    clang::CallExpr *callExpr = &CE;
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::CallExpr::getDirectCallee).stubs().will(returnValue(funcDecl));
    MOCKER(&clang::FunctionDecl::getQualifiedNameAsString)
        .stubs()
        .will(returnValue(std::string("AscendC::AssertPrint")));
    EXPECT_EQ(visitor.VisitCallExpr(callExpr), true);
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_VisitFunctionDecl)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    static clang::AnnotateAttr attrA;
    static clang::AnnotateAttr attrB;
    MOCKER(&clang::FunctionDecl::attrs).stubs().will(returnValue(
        std::vector<clang::AnnotateAttr*>{&attrA, &attrB}
    ));

    static clang::ParmVarDecl paramA;
    static clang::ParmVarDecl paramB;
    MOCKER(&clang::FunctionDecl::parameters).stubs().will(returnValue(
        std::vector<clang::ParmVarDecl *>{&paramA, &paramB}
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
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test mangledName",
        "test.cpp",
        2,
        2
    };
    MOCKER(&AscPlugin::GetKernelInfo).stubs().will(returnValue(mockKernelInfo));
    EXPECT_EQ(visitor.VisitFunctionDecl(funcDecl), true);
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_VisitFunctionDecl_empty)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
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
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "",
        "test.cpp",
        2,
        2
    };
    MOCKER(&AscPlugin::GetKernelInfo).stubs().will(returnValue(mockKernelInfo));
    EXPECT_EQ(visitor.VisitFunctionDecl(funcDecl), true);
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_VisitVarDecl_nullptr)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl *varDecl = nullptr;
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_VisitVarDecl_unmatch)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl VD;
    clang::VarDecl *varDecl = &VD;
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_VisitVarDecl)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl VD;
    clang::VarDecl *varDecl = &VD;
    MOCKER(&clang::VarDecl::getName).stubs()
        .will(returnValue(llvm::StringRef("__enable_feature_for_compile_default")));
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::dyn_cast<clang::FunctionDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test mangledName",
        "test.cpp",
        2,
        2
    };
    MOCKER(&AscPlugin::GetKernelInfo).stubs().will(returnValue(mockKernelInfo));
    visitor.kernelFuncsKernelType_[mockKernelInfo] = "kernel type";
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
    visitor.kernelFuncsKernelType_[mockKernelInfo] = "";
    EXPECT_EQ(visitor.VisitVarDecl(varDecl), true);
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_StoreFuncKernelType_310P)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND310P);
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test mangledName",
        "test.cpp",
        2,
        2
    };
    std::string kernelType = "KERNEL_TYPE_AICORE";
    EXPECT_NO_THROW(StoreFuncKernelType(mockKernelInfo, kernelType));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_StoreFuncKernelType_910B)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND910B);
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test mangledName",
        "test.cpp",
        2,
        2
    };
    std::string kernelType = "KERNEL_TYPE_AIV_ONLY";
    EXPECT_NO_THROW(StoreFuncKernelType(mockKernelInfo, kernelType));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl_310P)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND310P);
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test1 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {},
        {false, 0, 0},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl_310P_kernel_type_set)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND310P);
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test1 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {},
        {false, 0, 0},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_VECTOR_CORE}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl_kfc)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND910B);
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test1 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {},
        {false, 0, 0},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Open};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl_auto_identify)
{
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test1 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {},
        {false, 0, 0},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType.clear();
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl_default_1_2)
{
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test1 mangledName",
        "test.cpp",
        3,
        3
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        3,
        3,
        {"AAA"},
        {AscPlugin::Param()},
        {},
        {false, 0, 0},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

// bisheng ktype is different with ascendc ktype
TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl_asc_bisheng_ktype)
{
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test1 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {"ggg"},
        {true, 1, 1},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_AIC_ONLY}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

// bisheng ktype(core_ratio) is different with ascendc ktype
TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl_asc_bisheng_ktype_multi_ktype)
{
    AscPlugin::Param a = {"int32_t", "fff", true, "abc", ""};
    AscPlugin::Param b = {"float", "abcd", false, "abcd", ""};
    AscPlugin::CoreRatio c = {true, 1, 2};
    AscPlugin::TemplateInstance inst = {{"A", "B", "C"}, {a, b}, "add_custom", "stub", c};
    AscPlugin::TemplateInstance inst1 = {{"A", "B", "C"}, {a, b}, "add_custom", "stub", {true, 1, 1}};
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test1 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {"aiv", "aic"},
        {true, 1, 1},
        true,
        {}, {inst, inst1}
    };
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_AIC_ONLY}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl_not_found)
{
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test1 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {"aic"},
        {false, 0, 0},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}


TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl_bisheng_attr_aiv)
{
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test1 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {"aiv"},
        {false, 0, 0},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl_bisheng_attr_with_core_ratio)
{
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test1 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {"ggg"},
        {true, 1, 1},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl_bisheng_attr_with_template_core_ratio)
{
    AscPlugin::Param a = {"int32_t", "fff", true, "abc", ""};
    AscPlugin::Param b = {"float", "abcd", false, "abcd", ""};
    AscPlugin::CoreRatio c = {true, 1, 2};
    AscPlugin::TemplateInstance inst = {{"A", "B", "C"}, {a, b}, "add_custom", "stub", c};
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test1 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {"ggg"},
        {true, 1, 1},
        true,
        {}, {inst}
    };
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

// have aic, aiv, core_ratio at same time
TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl_bisheng_attr_with_template_core_ratio_error)
{
    AscPlugin::Param a = {"int32_t", "fff", true, "abc", ""};
    AscPlugin::Param b = {"float", "abcd", false, "abcd", ""};
    AscPlugin::CoreRatio c = {true, 1, 2};
    AscPlugin::TemplateInstance inst = {{"A", "B", "C"}, {a, b}, "add_custom", "stub", c};
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test1 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {"aiv", "aic"},
        {true, 1, 1},
        true,
        {}, {inst}
    };
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImpl)
{
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test2 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {"aic"},
        {false, 0, 0},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImplWithKfc)
{
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test3 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test1 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {"aic"},
        {false, 0, 0},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Open};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImplIdentifyKtype)
{
    AscPlugin::KernelFuncInfo mockKernelInfo {
        "test4 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test4 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {"aic"},
        {false, 0, 0},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType.clear();
    AscPlugin::g_kernelFuncType[mockKernelInfo] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelTypeImplDefalutReturn)
{
    AscPlugin::KernelFuncInfo mockKernelInfo1 {
        "test5 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelFuncInfo mockKernelInfo2 {
        "test6 mangledName",
        "test.cpp",
        2,
        2
    };
    AscPlugin::KernelInfo info = {
        "test1 ",
        "test5 mangledName",
        "test1 mangledName consider prefix",
        "test.cpp",
        2,
        2,
        {"AAA"},
        {AscPlugin::Param()},
        {"aic"},
        {false, 0, 0},
        false,
        {}, {}
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    AscPlugin::g_kernelFuncType.clear();
    AscPlugin::g_kernelFuncType[mockKernelInfo1] = {
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Close};
    AscPlugin::g_kernelFuncType[mockKernelInfo2] = {
    {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX}, AscPlugin::KfcScene::Close};
    EXPECT_NO_THROW(GetKernelFuncScene(info));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelInfo_withoutMangledName)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::CompilerInstance::getASTContext).stubs().will(returnValue(context));
    MOCKER(&clang::ASTContext::createMangleContext).stubs().will(returnValue(new clang::MangleContext()));
    MOCKER(&clang::MangleContext::shouldMangleDeclName).stubs().will(returnValue(false));
    EXPECT_NO_THROW(AscPlugin::GetKernelInfo(funcDecl));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetKernelInfo_withMangledName)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::CompilerInstance::getASTContext).stubs().will(returnValue(context));
    MOCKER(&clang::ASTContext::createMangleContext).stubs().will(returnValue(new clang::MangleContext()));
    MOCKER(&clang::MangleContext::shouldMangleDeclName).stubs().will(returnValue(true));
    EXPECT_NO_THROW(AscPlugin::GetKernelInfo(funcDecl));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_HandleTranslationUnit)
{
    clang::ASTContext context;
    clang::CompilerInstance compiler;
    AscPlugin::ASTDeviceConsumer consumer(context, compiler);
    AscPlugin::InfoManager::GetInstance().tmpPath_ = "/tmp/AscPlugin";
    AscPlugin::InfoManager::GetInstance().logPath_ = "/tmp/AscPlugin";
    system("mkdir -p /tmp/AscPlugin");
    const char* outEnvValue = "1";
    const char* levelEnvValue = "0";
    MOCKER(&AscPlugin::LogManager::GetOutEnv).stubs().will(returnValue(outEnvValue));
    MOCKER(&AscPlugin::LogManager::GetLevelEnv).stubs().will(returnValue(levelEnvValue));
    EXPECT_NO_THROW(consumer.HandleTranslationUnit(context));
    system("rm -rf /tmp/AscPlugin");
    const char* levelEnvValue2 = "1";
    MOCKER(&AscPlugin::LogManager::GetLevelEnv).stubs().will(returnValue(levelEnvValue2));
    EXPECT_NO_THROW(consumer.HandleTranslationUnit(context));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_GetSourceLine)
{
    const char* cch = "test";
    AscPlugin::AscDiagnostic ad;
    clang::Diagnostic diag;
    clang::SourceManager sm;
    clang::SourceLocation sl;
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Level::Warning;
    clang::ASTContext context = clang::ASTContext();
    clang::CUDAKernelCallExpr cudakce;
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

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_HandleDiagnosticInvalid)
{
    const char* outEnvValue = "0";
    MOCKER(&AscPlugin::LogManager::GetOutEnv).stubs().will(returnValue(outEnvValue));
    AscPlugin::AscDiagnostic ad;
    clang::Diagnostic diag;
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Level::Warning;
    clang::ASTContext context = clang::ASTContext();
    clang::CUDAKernelCallExpr cudakce;
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    EXPECT_NO_THROW(ad.HandleDiagnostic(level, diag));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_HandleDiagnosticValid)
{
    const char* outEnvValue = "1";
    MOCKER(&AscPlugin::LogManager::GetOutEnv).stubs().will(returnValue(outEnvValue));
    AscPlugin::AscDiagnostic ad;
    clang::Diagnostic diag;
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Level::Warning;
    clang::ASTContext context = clang::ASTContext();
    clang::CUDAKernelCallExpr cudakce;
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    MOCKER(&clang::PresumedLoc::isInvalid).stubs().will(returnValue(false));
    EXPECT_NO_THROW(ad.HandleDiagnostic(level, diag));
    level = clang::DiagnosticsEngine::Level::Error;
    EXPECT_NO_THROW(ad.HandleDiagnostic(level, diag));
    level = clang::DiagnosticsEngine::Level::Fatal;
    EXPECT_NO_THROW(ad.HandleDiagnostic(level, diag));
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_IncludeInDiagnosticCounts)
{
    AscPlugin::AscDiagnostic ad;
    EXPECT_EQ(ad.IncludeInDiagnosticCounts(), true);
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_CreateASTConsumer)
{
    AscPlugin::DeviceAnalyzeAction afa;
    clang::Diagnostic diag;
    clang::SourceManager sm;
    clang::SourceLocation sl;
    clang::DiagnosticsEngine::Level level = clang::DiagnosticsEngine::Level::Warning;
    clang::ASTContext context = clang::ASTContext();
    clang::CUDAKernelCallExpr cudakce;
    clang::BoundNodes bn;
    clang::CompilerInstance compiler = clang::CompilerInstance();
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    clang::HeaderSearchOptions hso;
    MOCKER(&clang::CompilerInstance::getHeaderSearchOpts).stubs().will(returnValue(hso));
    MOCKER(&clang::CompilerInstance::getASTContext).stubs().will(returnValue(context));
    auto res = afa.CreateASTConsumer(compiler, llvm::StringRef("test"));
    EXPECT_TRUE(res != nullptr);
}

TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_FindMatmulObjRegister)
{
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl VD;
    clang::VarDecl *varDecl = &VD;
    std::string retStr("AscendC::KfcServer");
    clang::FunctionDecl FD(context);
    clang::FunctionDecl *funcDecl = &FD;
    AscPlugin::KernelFuncInfo kernelKey = {"", "", 0, 0};
    visitor.kernelFuncsKernelType_[kernelKey] = "";
    MOCKER(static_cast<std::string (clang::QualType::*)() const>
        (&clang::QualType::getAsString)).stubs().will(returnValue(std::string("AscendC::KfcServer")));
    MOCKER(&clang::dyn_cast<clang::FunctionDecl>, clang::FunctionDecl *(const clang::DeclContext*))
        .stubs()
        .will(returnValue(funcDecl));
    EXPECT_NO_THROW(visitor.FindMatmulObjRegister(varDecl));
}


TEST_F(TEST_ASC_DEVICE_CONSUMER, asc_FindOpSystemCfg)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    clang::ASTContext context = clang::ASTContext();
    clang::CompilerInstance compiler = clang::CompilerInstance();
    AscPlugin::ASTDeviceVisitor visitor(context, compiler);
    clang::VarDecl VD;
    clang::VarDecl *varDecl = &VD;
    MOCKER(&clang::VarDecl::getName).stubs()
        .will(returnValue(llvm::StringRef("g_opSystemRunCfg")));

    EXPECT_NO_THROW(visitor.FindOpSystemCfg(varDecl));
    manager.SetOpSystemCfg(false);
}