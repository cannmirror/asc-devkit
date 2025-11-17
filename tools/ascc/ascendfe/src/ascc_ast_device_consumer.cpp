/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file ascc_ast_device_consumer.cpp
 * \brief
 */

#include "ascc_ast_device_consumer.h"

#include <fstream>
#include <string>
#include <set>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Mangle.h>
#include <llvm/Support/raw_os_ostream.h>

#include "ascc_ast_utils.h"
#include "ascc_dump_flags.h"
#include "ascc_ast_info_collector.h"
#include "ascc_info_storage.h"
#include "ascc_common_utils.h"
#include "ascc_argument_manager.h"

namespace Ascc {

namespace {
// record line num for MIX_NUM, DUMP_WORKSPACE_SIZE and DUMP_UINTSIZE in json to replace
void UpdateInfoByKeywords(clang::VarDecl *varDecl)
{
    clang::SourceManager& sourceManager = varDecl->getASTContext().getSourceManager();
    clang::SourceLocation loc = varDecl->getLocation();
    uint32_t lineNum = sourceManager.getSpellingLineNumber(loc);
    std::string varName = varDecl->getName().str();
    bool isInNameSpaceAsc = IsVarInNamespace(varDecl, "AscendC");
    if (varName == "MIX_NUM" && isInNameSpaceAsc) {
        // Find the line number for "constexpr int32_t MIX_NUM = xx"
        Ascc::AsccGlobalEnvManager::GetInstance().mixNumLineNum = lineNum;
        ASC_LOG_ASC_DEBUG(PREPROCESS, "Find MIX_NUM in file %s at line %u",
            Ascc::AsccArgumentManager::GetInstance().GetInputFile().c_str(), lineNum);
    } else if (varName == "DUMP_WORKSPACE_SIZE" && isInNameSpaceAsc) {
        // Find the line number for "const uint32_t DUMP_WORKSPACE_SIZE = xx"
        Ascc::AsccGlobalEnvManager::GetInstance().dumpWorkspaceLineNum = lineNum;
        ASC_LOG_ASC_DEBUG(PREPROCESS, "Find DUMP_WORKSPACE_SIZE in file %s at line %u",
            Ascc::AsccArgumentManager::GetInstance().GetInputFile().c_str(), lineNum);
    } else if (varName == "DUMP_UINTSIZE" && isInNameSpaceAsc) {
        // Find the line number for "constexpr size_t DUMP_UINTSIZE = xx"
        Ascc::AsccGlobalEnvManager::GetInstance().dumpUintLineNum = lineNum;
        ASC_LOG_ASC_DEBUG(PREPROCESS, "Find DUMP_UINTSIZE in file %s at line %u",
            Ascc::AsccArgumentManager::GetInstance().GetInputFile().c_str(), lineNum);
    }
}
} // namespace

ASTDeviceVisitor::ASTDeviceVisitor(clang::ASTContext &context, clang::CompilerInstance &compiler)
    : context_(context), compiler_(compiler), srcManager_(context.getSourceManager())
{}

bool ASTDeviceVisitor::VisitFunctionDecl(const clang::FunctionDecl *funcDecl)
{
    bool hasGlobal = false;
    bool hasDevice = false;
    static constexpr llvm::StringRef globalAttr = "global";
    static constexpr llvm::StringRef deviceAttr = "device";
    for (const auto *attr : funcDecl->attrs()) {
        if (const auto *annotate = llvm::dyn_cast<clang::AnnotateAttr>(attr)) {
            llvm::StringRef annotation = annotate->getAnnotation();
            if (annotation == globalAttr) {
                hasGlobal = true;
            } else if (annotation == deviceAttr) {
                hasDevice = true;
            }
        }
    }
    if (hasGlobal && hasDevice) {
        llvm::StringRef funcName = funcDecl->getName();
        // Anonymous functions (e.g., lambda) let getNameAsString return an empty string
        if (funcName.empty()) {
            ASC_LOG_ASC_WARN(PREPROCESS, "Found global aicore function, but it is anonymous");
        } else {
            ASC_LOG_ASC_DEBUG(PREPROCESS, "Found global aicore function: %s", funcName.data());
        }
        globalAicoreFuncsKernelType_[funcDecl] = "";
    }
    return true;
}

bool ASTDeviceVisitor::VisitCallExpr(clang::CallExpr *exprCall)
{
    if (!exprCall) {
        ASC_LOG_ASC_DEBUG(PREPROCESS, "Null CallExpr: Possible AST corruption or invalid code");
        return true;
    }
    auto &dumpInfo = AsccDumpFlags::GetInstance();
    AsccGlobalEnvManager &envVar = AsccGlobalEnvManager::GetInstance();
    if (clang::FunctionDecl *funcDecl = exprCall->getDirectCallee()) {
        std::string qualifiedName = funcDecl->getQualifiedNameAsString();
        clang::SourceLocation expansionStart = exprCall->getBeginLoc();
        expansionStart = srcManager_.getExpansionLoc(expansionStart);
        clang::PresumedLoc pLoc = srcManager_.getPresumedLoc(expansionStart);
        const char* fname = pLoc.getFilename();
        std::string filename = fname ? fname : "<unknown file>";
        if (filename.find(envVar.ascendCannPackagePath) == std::string::npos) {
            if (qualifiedName.find("AscendC::printf") != std::string::npos ||
                qualifiedName.find("AscendC::PRINTF") != std::string::npos ||
                qualifiedName.find("AscendC::DumpTensor") != std::string::npos ||
                qualifiedName.find("AscendC::DumpAccChkPoint") != std::string::npos) {
                dumpInfo.SetPrintfFlag();
                ASC_LOG_ASC_DEBUG(PREPROCESS, "Found %s call at line: %u, file: %s", qualifiedName.c_str(),
                    pLoc.getLine(), fname);
            } else if (qualifiedName.find("AscendC::AssertImpl") != std::string::npos ||
                qualifiedName.find("AscendC::AssertFail") != std::string::npos ||
                qualifiedName.find("AscendC::AssertPrint") != std::string::npos) {
                dumpInfo.SetAssertFlag();
                ASC_LOG_ASC_DEBUG(PREPROCESS, "Found %s call at line: %u, file: %s", qualifiedName.c_str(),
                    pLoc.getLine(), fname);
            }
        }
    } else if (clang::UnresolvedLookupExpr *ule = clang::dyn_cast<clang::UnresolvedLookupExpr>(exprCall->getCallee())) {
        std::string funcName = ule->getName().getAsString();
        clang::SourceLocation startLoc = exprCall->getBeginLoc();
        clang::SourceLocation expansionLoc = srcManager_.getExpansionLoc(startLoc);
        clang::PresumedLoc pLoc = srcManager_.getPresumedLoc(expansionLoc);
        const char* fname = pLoc.getFilename();
        std::string filename = fname ? fname : "<unknown file>";
        if (filename.find(envVar.ascendCannPackagePath) == std::string::npos) {
            if (funcName == "printf" || funcName == "PRINTF" ||
                funcName == "DumpTensor" || funcName == "DumpAccChkPoint") {
                ASC_LOG_ASC_DEBUG(PREPROCESS, "Found unresolved %s at line: %u, file: %s", funcName.c_str(),
                    GetLineNumber(exprCall), fname);
                if (const clang::NestedNameSpecifier *nns = ule->getQualifier()) {
                    if (const clang::NamespaceDecl *ns = clang::dyn_cast<clang::NamespaceDecl>(nns->getAsNamespace())) {
                        if (ns->getName() == "AscendC") {
                            dumpInfo.SetPrintfFlag();
                            ASC_LOG_ASC_DEBUG(PREPROCESS, "Confirmed AscendC::%s at line: %u", funcName.c_str(),
                                GetLineNumber(exprCall));
                        }
                    }
                }
            } else if (funcName == "AssertImpl" || funcName == "AssertFail" || funcName == "AssertPrint")  {
                ASC_LOG_ASC_DEBUG(PREPROCESS, "Found unresolved %s at line: %u, file: %s", funcName.c_str(),
                    GetLineNumber(exprCall), fname);
                if (const clang::NestedNameSpecifier *nns = ule->getQualifier()) {
                    if (const clang::NamespaceDecl *ns = clang::dyn_cast<clang::NamespaceDecl>(nns->getAsNamespace())) {
                        if (ns->getName() == "AscendC") {
                            dumpInfo.SetAssertFlag();
                            ASC_LOG_ASC_DEBUG(PREPROCESS, "Confirmed AscendC::%s at line: %u", funcName.c_str(),
                                GetLineNumber(exprCall));
                        }
                    }
                }
            }
        }
    }
    return true;
}

unsigned ASTDeviceVisitor::GetLineNumber(const clang::Stmt *stmt) const
{
    return srcManager_.getExpansionLineNumber(stmt->getBeginLoc());
}

void ASTDeviceVisitor::UpdateKernelKfcInfo(clang::VarDecl *varDecl)
{
    const std::string varType = FindVarTypeStr(varDecl);
    if (varType.find("AscendC::KfcServer") == std::string::npos) {
        return;
    }
    clang::SourceManager& sourceManager = varDecl->getASTContext().getSourceManager();
    clang::SourceLocation loc = varDecl->getLocation();
    uint32_t lineNum = sourceManager.getSpellingLineNumber(loc);
    ASC_LOG_ASC_DEBUG(PREPROCESS, "Find AscendC::KfcServer in file %s at line %u",
        Ascc::AsccArgumentManager::GetInstance().GetInputFile().c_str(), lineNum);

    const clang::DeclContext *ctx = varDecl->getDeclContext();
    while(ctx) {
        const clang::FunctionDecl *funcDecl = clang::dyn_cast<clang::FunctionDecl>(ctx);
        if (funcDecl != nullptr) {
            auto it = globalAicoreFuncsKernelType_.find(funcDecl);
            if (it != globalAicoreFuncsKernelType_.end()) {
                SetFuncFileKernelHasKfc(funcDecl, true);
                break;
            }
        }
        ctx = ctx->getParent();
    }
}


bool ASTDeviceVisitor::VisitVarDecl(clang::VarDecl *varDecl)
{
    if (!varDecl) {
        ASC_LOG_ASC_DEBUG(PREPROCESS, "Null VarDecl: Possible AST corruption or invalid code");
        return true;
    }

    UpdateInfoByKeywords(varDecl);     // Update line num for json file
    UpdateKernelKfcInfo(varDecl);      // Update whether has kfc for funcInfo

    constexpr llvm::StringRef targetVar = "__enable_feature_for_compile_default";
    if (varDecl->getName() != targetVar) {
        return true;
    }
    const clang::DeclContext *ctx = varDecl->getDeclContext();
    while(ctx) {
        if (const clang::FunctionDecl *funcDecl = clang::dyn_cast<clang::FunctionDecl>(ctx)) {
            auto it = globalAicoreFuncsKernelType_.find(funcDecl);
            if (it != globalAicoreFuncsKernelType_.end()) {
                if (it->second != "") {
                    ASC_LOG_ASC_WARN(PREPROCESS, "Global aicore function [%s] already has kernel type: %s",
                        funcDecl->getNameAsString().c_str(), it->second.str().c_str());
                    break;
                }
                // Preprocess macro expansion position​​
                clang::SourceLocation loc = srcManager_.getExpansionLoc(varDecl->getLocation());
                if (loc.isMacroID()) {
                    loc = srcManager_.getImmediateMacroCallerLoc(loc);
                }
                clang::PresumedLoc pLoc = srcManager_.getPresumedLoc(loc);
                // Init expression anlysis
                clang::Expr *initExpr = varDecl->getInit();
                if (initExpr && !initExpr->isValueDependent()) {
                    initExpr = initExpr->IgnoreImpCasts()->IgnoreParens();
                    if (auto *declRef = clang::dyn_cast_or_null<clang::DeclRefExpr>(initExpr)) {
                        static std::set<std::string> kernelTypeStorage;
                        auto kernelTypeStr = declRef->getDecl()->getNameAsString();
                        SetFuncFileKernelType(funcDecl, kernelTypeStr);
                        auto ret = kernelTypeStorage.insert(std::move(kernelTypeStr));
                        it->second = ret.first->c_str();
                        std::string mangledName;
                        if (auto mangleContext = funcDecl->getASTContext().createMangleContext()) {
                            if (mangleContext->shouldMangleDeclName(funcDecl)) {
                                llvm::raw_string_ostream mangledStream(mangledName);
                                mangleContext->mangleName(funcDecl, mangledStream);
                                mangledStream.flush();
                            } else {
                                mangledName = funcDecl->getNameAsString();
                            }
                            delete mangleContext;
                        }
                        ASC_LOG_ASC_DEBUG(PREPROCESS, "[AST]: Kernel type: %s at file: %s, line: %u, func name: %s, "
                            "mangling name: %s", it->second.str().c_str(), pLoc.getFilename(), pLoc.getLine(),
                            funcDecl->getNameAsString().c_str(), mangledName.c_str());
                    }
                }
                break;
            }
        }
        ctx = ctx->getParent();
    }
    return true;
}

ASTDeviceConsumer::ASTDeviceConsumer(clang::ASTContext &context, clang::CompilerInstance &compiler)
    : visitor_(context, compiler), compiler_(compiler)
{}

void ASTDeviceConsumer::HandleTranslationUnit(clang::ASTContext &context)
{
    Ascc::AsccGlobalEnvManager &envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    std::string dumpPath = envVar.asccTmpHostGenPath + "/ASTDumpDevice.txt";
    // Creating File Streams and Adapters
    std::ofstream dumpFile(dumpPath);
    if (dumpFile) {
        llvm::raw_os_ostream dumpStream(dumpFile);
        // Invoke the dump method and redirect the output
        if (Ascc::AsccGlobalEnvManager::ascendSlogPrintToStdout == 1 &&
            Ascc::AsccGlobalEnvManager::ascendGlobalLogLevel <= 0) {
            context.getTranslationUnitDecl()->dump(dumpStream);
            ASC_LOG_ASC_DEBUG(PREPROCESS, "AST dump device saved to: %s", dumpPath.c_str());
        }
    } else {
        ASC_LOG_ASC_DEBUG(PREPROCESS, "Failed to create AST dump device file");
    }
    visitor_.TraverseDecl(context.getTranslationUnitDecl());
}

std::unique_ptr<clang::ASTConsumer> DeviceAnalyzeAction::CreateASTConsumer(
    clang::CompilerInstance &compiler, llvm::StringRef inputFile)
{
    (void)inputFile;
    for (const auto &entry : compiler.getHeaderSearchOpts().UserEntries) {
        ASC_LOG_ASC_DEBUG(PREPROCESS, "ASCC Device analyzer: Header Search Path: %s", entry.Path.c_str());
    }
    clang::DiagnosticsEngine &diagEngine = compiler.getDiagnostics();
    diagEngine.setClient(new AsccDiagnostic(), /*ShouldOwnClient=*/true);
    diagEngine.setErrorLimit(0);
    return std::make_unique<ASTDeviceConsumer>(compiler.getASTContext(), compiler);
}
}  // namespace Ascc