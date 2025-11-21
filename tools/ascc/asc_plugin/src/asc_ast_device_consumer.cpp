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
 * \file asc_ast_device_consumer.cpp
 * \brief
 */

#include "asc_ast_device_consumer.h"

#include <fstream>
#include <string>
#include <utility>
#include <set>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ASTContext.h>
#include <llvm/Support/raw_os_ostream.h>

#include "asc_log.h"

namespace AscPlugin {

namespace {
bool IsPrintfRelated(const std::string &funcName)
{
    return (funcName == "printf" || funcName == "PRINTF" || funcName == "DumpTensor" || funcName == "DumpAccChkPoint" ||
            funcName == "AssertImpl");
}
}  // namespace

ASTDeviceVisitor::ASTDeviceVisitor(clang::ASTContext &context, clang::CompilerInstance &compiler)
    : context_(context), compiler_(compiler), srcManager_(context.getSourceManager())
{}

uint32_t ASTDeviceVisitor::GetLineNumber(const clang::Stmt *stmt) const
{
    return static_cast<uint32_t>(srcManager_.getExpansionLineNumber(stmt->getBeginLoc()));
}

llvm::DenseMap<KernelFuncInfo, std::pair<std::unordered_set<KernelMetaType>, KfcScene>> g_kernelFuncType;

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
        AscPlugin::KernelFuncInfo kernelKey = GetKernelInfo(funcDecl);
        llvm::StringRef funcName = funcDecl->getName();
        if (kernelKey.mangledName.empty()) {
            ASC_LOGW("Found global aicore function at %s:%u, but it is anonymous", kernelKey.fileName.c_str(),
                kernelKey.lineNum);
        } else {
            ASC_LOGD("Found global aicore function: %s, mangledName: %s at %s:%u, col:%u", funcName.data(),
                kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(), kernelKey.lineNum, kernelKey.colNum);
        }

        std::vector<std::string> kernelVarInfo = {funcName.data()};
        for (clang::ParmVarDecl *param : funcDecl->parameters()) {
            std::string typeStr = param->getType().getAsString(); // variable type string
            std::string nameStr = param->getNameAsString();       // variable name string
            kernelVarInfo.insert(kernelVarInfo.end(), {typeStr, nameStr});
            ASC_LOGD("Global aicore function: %s, mangledName: %s, has variable type %s, variable name %s.",
                funcName.data(), kernelKey.mangledName.c_str(), typeStr.c_str(), nameStr.c_str());
        }
        AscPlugin::g_kernelVarMap.insert({kernelKey.mangledName, kernelVarInfo});

        kernelFuncsKernelType_[kernelKey] = "";
        g_kernelFuncType[kernelKey] = {{KernelMetaType::KERNEL_TYPE_MAX}, KfcScene::Close};
    }
    return true;
}

bool ASTDeviceVisitor::VisitCallExpr(clang::CallExpr *exprCall)
{
    if (!exprCall) {
        ASC_LOGD("Null CallExpr: Possible AST corruption or invalid code");
        return true;
    }

    clang::FunctionDecl *funcDecl = exprCall->getDirectCallee();
    clang::UnresolvedLookupExpr *ule = clang::dyn_cast<clang::UnresolvedLookupExpr>(exprCall->getCallee());
    if ((!funcDecl) && (!ule)) {
        return true;
    }

    auto& manager = InfoManager::GetInstance();
    clang::SourceLocation startLoc = exprCall->getBeginLoc();
    // get source code location from macro expansion
    clang::SourceLocation expansionLoc = srcManager_.getExpansionLoc(startLoc);
    clang::PresumedLoc pLoc = srcManager_.getPresumedLoc(expansionLoc);
    const char* fname = pLoc.getFilename();
    std::string filename = fname ? fname : "<unknown file>";
    // Need to find printf / assert from users files instead of cann package files
    if (filename.find(manager.GetPathInfo().cannPath) != std::string::npos) {
        return true;
    }

    if (funcDecl) {
        std::string qualifiedName = funcDecl->getQualifiedNameAsString();
        if (qualifiedName.find("AscendC::printf") != std::string::npos ||
            qualifiedName.find("AscendC::PRINTF") != std::string::npos ||
            qualifiedName.find("AscendC::DumpTensor") != std::string::npos ||
            qualifiedName.find("AscendC::DumpAccChkPoint") != std::string::npos) {
            manager.SetHasPrintf(true);
            ASC_LOGD("Found %s call at line: %u, file: %s", qualifiedName.c_str(), pLoc.getLine(), fname);
        } else if (qualifiedName.find("AscendC::AssertImpl") != std::string::npos ||
                   qualifiedName.find("AscendC::AssertFail") != std::string::npos ||
                   qualifiedName.find("AscendC::AssertPrint") != std::string::npos) {
            manager.SetHasAssert(true);
            ASC_LOGD("Found %s call at line: %u, file: %s", qualifiedName.c_str(), pLoc.getLine(), fname);
        }
    } else if (ule) {
        std::string funcName = ule->getName().getAsString();
        if (IsPrintfRelated(funcName)) {
            ASC_LOGD("Found unresolved %s at line: %u, file: %s", funcName.c_str(), GetLineNumber(exprCall), fname);
            if (const clang::NestedNameSpecifier *nns = ule->getQualifier()) {
                if (const clang::NamespaceDecl *ns = clang::dyn_cast<clang::NamespaceDecl>(nns->getAsNamespace())) {
                    if (ns->getName() == "AscendC") {
                        ASC_LOGD("Confirmed AscendC::%s at line: %u, file: %s", funcName.c_str(),
                            GetLineNumber(exprCall), fname);
                        (funcName == "AssertImpl" || funcName == "AssertFail" || funcName == "AssertPrint")
                            ? manager.SetHasAssert(true)
                            : manager.SetHasPrintf(true);
                    }
                }
            }
        }
    }
    return true;
}

void StoreFuncKernelType(const AscPlugin::KernelFuncInfo& kernelKey, const std::string& kernelTypeStr)
{
    bool kernelTypeValid = false;
    ShortSocVersion shortSoc = InfoManager::GetInstance().GetShortSocVersion();
    // For each kernel, there should be only 1 Ascendc Kernel type
    if (shortSoc == ShortSocVersion::ASCEND910B) {
        auto iter = KERNEL_TYPE_MAP_V220.find(kernelTypeStr);
        if (iter != KERNEL_TYPE_MAP_V220.end()) {
            g_kernelFuncType[kernelKey].first = {iter->second};
            kernelTypeValid = true;
        }
    } else if (shortSoc == ShortSocVersion::ASCEND310P) {
        auto iter = KERNEL_TYPE_MAP_V200.find(kernelTypeStr);
        // 310P only supports AICORE for now
        if (iter != KERNEL_TYPE_MAP_V200.end() && kernelTypeStr == "KERNEL_TYPE_AICORE") {
            g_kernelFuncType[kernelKey].first = {iter->second};
            kernelTypeValid = true;
        }
    } else if (shortSoc == ShortSocVersion::ASCEND910_95) {
        auto iter = KERNEL_TYPE_MAP_C310.find(kernelTypeStr);
        if (iter != KERNEL_TYPE_MAP_C310.end()) {
            g_kernelFuncType[kernelKey].first = {iter->second};
            kernelTypeValid = true;
        }
    }

    if (kernelTypeValid) {
        ASC_LOGD("kernel type: [%s] store successfully, kernel func mangled name: %s at %s:%u, col:%u",
            kernelTypeStr.c_str(), kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(), kernelKey.lineNum,
            kernelKey.colNum);
    } else {
        ASC_LOGE("kernel type: [%s] is not supported on current soc, store failed, kernel func mangled name: %s "
            "at %s:%u, col:%u", kernelTypeStr.c_str(), kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(),
            kernelKey.lineNum, kernelKey.colNum);
    }
}

std::unordered_set<KernelMetaType> GetBishengKType(const KernelInfo& kernelInfo)
{
    std::unordered_set<KernelMetaType> res;
    // aiv only / aic only
    for (const auto& kernelAttr : kernelInfo.kernelAttributes) {
        if (kernelAttr == "aiv") {
            res.insert(KernelMetaType::KERNEL_TYPE_AIV_ONLY);
        } else if (kernelAttr == "aic") {
            res.insert(KernelMetaType::KERNEL_TYPE_AIC_ONLY);
        }
    }
    // core ratio for mix
    if (!kernelInfo.isTemplate) {
        if (kernelInfo.ratio.isCoreRatio) {
            auto kType = GetBishengKTypeByCoreRatio(kernelInfo.ratio);
            if (kType != KernelMetaType::KERNEL_TYPE_MAX) {
                res.insert(kType);
            }
        }
    } else {
        for (const auto& instance : kernelInfo.templateInstances) {
            if (instance.ratio.isCoreRatio) {
                auto kType = GetBishengKTypeByCoreRatio(instance.ratio);
                if (kType != KernelMetaType::KERNEL_TYPE_MAX) {
                    res.insert(kType);
                }
            }
        }
    }
    std::string kernelTypeStr;
    for (const auto& ktype : res) {
        kernelTypeStr += KTYPE_STR_MAP.at(ktype) + ", ";
    }
    ASC_LOGD("kernel func mangled name: %s at %s:%u, col:%u has following bisheng kernel type: %s.",
        kernelInfo.kernelMangledName.c_str(), kernelInfo.fileName.c_str(), kernelInfo.lineNum, kernelInfo.colNum,
        kernelTypeStr.c_str());

    if (res.size() > 1) {
        // case only for core_ratio(x, y), thus not have aic / aiv only
        if (std::find(res.begin(), res.end(), KernelMetaType::KERNEL_TYPE_AIC_ONLY) != res.end()) {
            ASC_LOGE("kernel func mangled name: %s at %s:%u, col:%u should not have KERNEL_TYPE_AIC_ONLY while have MIX"
                " kernel type",kernelInfo.kernelMangledName.c_str(), kernelInfo.fileName.c_str(), kernelInfo.lineNum,
                kernelInfo.colNum);
        }
        if (std::find(res.begin(), res.end(), KernelMetaType::KERNEL_TYPE_AIV_ONLY) != res.end()) {
            ASC_LOGE("kernel func mangled name: %s at %s:%u, col:%u should not have KERNEL_TYPE_AIV_ONLY while have MIX"
                " kernel type",kernelInfo.kernelMangledName.c_str(), kernelInfo.fileName.c_str(), kernelInfo.lineNum,
                kernelInfo.colNum);
        }
    }

    return res;
}

std::pair<std::unordered_set<KernelMetaType>, KfcScene> GetKernelFuncScene(const AscPlugin::KernelInfo& kernelInfo)
{
    KernelFuncInfo kernelKey = {kernelInfo.kernelMangledName, kernelInfo.fileName, kernelInfo.lineNum,
        kernelInfo.colNum};
    std::unordered_set<KernelMetaType> defaultKType = {AscPlugin::KernelMetaType::KERNEL_TYPE_MAX};
    auto it = g_kernelFuncType.find(kernelKey);
    ShortSocVersion shortSoc = InfoManager::GetInstance().GetShortSocVersion();
    KernelMetaType socDefaultKtype = DEFAULT_KERNEL_TYPE_MAP.at(shortSoc);
    if (it != g_kernelFuncType.end()) {
        const auto& [ktypeSet, kfcFlag] = it->second;
        std::unordered_set<KernelMetaType> bishengKtype = GetBishengKType(kernelInfo);
        if (ktypeSet == defaultKType) { // user does not set Ascendc Kernel type. May need ascendc auto identification
            // 310P does not need automatic kernel type identification
            if (shortSoc == ShortSocVersion::ASCEND310P) {
                ASC_LOGD("Can not find Kernel type, kernel func mangled name: %s at %s:%u, col:%u, using default "
                    "KERNEL_TYPE_AICORE", kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(), kernelKey.lineNum,
                    kernelKey.colNum);
                return {{socDefaultKtype}, KfcScene::Close};
            }
            // if user set bisheng Kernel type, use bisheng kernel type instead
            if (!bishengKtype.empty()) {
                return {bishengKtype, kfcFlag};
            }

            // 910B needs automatic kernel type identification
            if (kfcFlag == AscPlugin::KfcScene::Open) {
                ASC_LOGD("kernel func mangled name: %s at %s:%u, col:%u using kfcserver and user don't set kernel type "
                    "manually, using default KERNEL_TYPE_MIX_AIC_1_2", kernelKey.mangledName.c_str(),
                    kernelKey.fileName.c_str(), kernelKey.lineNum, kernelKey.colNum);
                return {{socDefaultKtype}, kfcFlag};   // Kfc using mix 1:2
            }
            if (g_kernelFuncType.size() == 1) {
                ASC_LOGD("Can not find Kernel type, kernel func mangled name: %s at %s:%u, col:%u, automatic kernel type "
                    "identification is now enabled", kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(),
                    kernelKey.lineNum, kernelKey.colNum);
                return {{IdentifyKtypeImpl(kernelKey, kernelInfo.templateInstances)}, kfcFlag};
            }
            ASC_LOGD("Can not find Kernel type, kernel func mangled name: %s at %s:%u, col:%u, using default "
                "KERNEL_TYPE_MIX_AIC_1_2", kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(),
                kernelKey.lineNum, kernelKey.colNum);
            return {{socDefaultKtype}, KfcScene::Close};
        } else {
            KernelMetaType ascKType = ExtractKernelType(ktypeSet); // ascend only has 1 kernel type for each kernel
            // 310P only supports KERNEL_TYPE_AICORE for now
            if (shortSoc == ShortSocVersion::ASCEND310P && ascKType != AscPlugin::KernelMetaType::KERNEL_TYPE_AICORE) {
                ASC_LOGE("Unsupported kernel type %s for kernel func mangled name: %s at %s:%u, col:%u. Current soc "
                    "only supports KERNEL_TYPE_AICORE for now.", KERNEL_TYPE_STR_MAP.at(ascKType).c_str(),
                    kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(), kernelKey.lineNum, kernelKey.colNum);
                return {{socDefaultKtype}, KfcScene::Close};
            }
            // If has core ratio, check whether bisheng kernel type is consistent with ascendc kernel type
            KernelMetaType curBishengKType = ExtractKernelType(bishengKtype);
            if (bishengKtype.size() > 1) {
                ASC_LOGE("Has more than 1 kernel type for kernel func mangled name: %s at %s:%u, col:%u, contradicts "
                    "with Ascendc kernel type %s", kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(),
                    kernelKey.lineNum, kernelKey.colNum, KTYPE_STR_MAP.at(ascKType).c_str());
            } else if (bishengKtype.size() == 1 && curBishengKType != ascKType) {
                ASC_LOGE("Bisheng kernel type %s contradicts with Ascendc kernel type %s for kernel func mangled name:"
                    " %s at %s:%u, col:%u", KTYPE_STR_MAP.at(curBishengKType).c_str(),
                    KTYPE_STR_MAP.at(ascKType).c_str(), kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(),
                    kernelKey.lineNum, kernelKey.colNum);
            }
            return it->second;
        }
    } else {
        ASC_LOGE("Unknown kernelInfo, mangledName: %s at %s:%u, col:%u, please check log.",
            kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(), kernelKey.lineNum, kernelKey.colNum);
        return {{socDefaultKtype}, KfcScene::Close};
    }
}

AscPlugin::KernelFuncInfo GetKernelInfo(const clang::FunctionDecl *funcDecl)
{
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

    clang::SourceLocation loc = funcDecl->getLocation();
    clang::SourceManager &srcMgr = funcDecl->getASTContext().getSourceManager();
    std::string fileName = CheckAndGetFullPath(srcMgr.getFilename(loc).str());
    uint32_t lineNum = static_cast<uint32_t>(srcMgr.getSpellingLineNumber(loc));
    uint32_t colNum = static_cast<uint32_t>(srcMgr.getSpellingColumnNumber(loc));
    return {mangledName, fileName, lineNum, colNum};
}

void ASTDeviceVisitor::FindKernelType(clang::VarDecl *varDecl)
{
    constexpr llvm::StringRef targetVar = "__enable_feature_for_compile_default";
    if (varDecl->getName() != targetVar) {
        return;
    }
    const clang::DeclContext *ctx = varDecl->getDeclContext();
    while(ctx) {
        if (const clang::FunctionDecl *funcDecl = clang::dyn_cast<clang::FunctionDecl>(ctx)) {
            AscPlugin::KernelFuncInfo kernelKey = GetKernelInfo(funcDecl);
            auto it = kernelFuncsKernelType_.find(kernelKey);
            if (it != kernelFuncsKernelType_.end()) {
                if (it->second != "") {
                    ASC_LOGW("Global aicore function [%s], mangledName: %s, already has kernel type: %s",
                        funcDecl->getNameAsString().c_str(), kernelKey.mangledName.c_str(), it->second.str().c_str());
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
                        auto ret = kernelTypeStorage.insert(std::move(kernelTypeStr));
                        it->second = ret.first->c_str();
                        ASC_LOGD("Found Kernel type: [%s] at file: %s, line: %u, kernelFunc name: %s, "
                            "mangled name: %s", it->second.str().c_str(), pLoc.getFilename(), pLoc.getLine(),
                            funcDecl->getNameAsString().c_str(), kernelKey.mangledName.c_str());
                        StoreFuncKernelType(kernelKey, it->second.str());
                    }
                }
                break;
            }
        }
        ctx = ctx->getParent();
    }
}

void ASTDeviceVisitor::FindOpSystemCfg(const clang::VarDecl *varDecl) const
{
    constexpr llvm::StringRef targetVar = "g_opSystemRunCfg";
    if (varDecl->getName() != targetVar) {
        return;
    } else {
        auto& manager = InfoManager::GetInstance();
        manager.SetOpSystemCfg(true);
        clang::SourceLocation loc = varDecl->getLocation();
        clang::SourceManager &srcMgr = varDecl->getASTContext().getSourceManager();
        std::string fileName = srcMgr.getFilename(loc).str();
        uint32_t lineNum = static_cast<uint32_t>(srcMgr.getSpellingLineNumber(loc));
        ASC_LOGW("Found g_opSystemRunCfg definition at line: %u, in file: %s. It is an inner system global var, please "
                 "do not use it!",
            lineNum,
            fileName.c_str());
    }
}

void ASTDeviceVisitor::FindMatmulObjRegister(clang::VarDecl *varDecl)
{
    const std::string varType = varDecl->getType().getAsString();
    if (varType.find("AscendC::KfcServer") == std::string::npos) {
        return;
    }
    const clang::DeclContext *ctx = varDecl->getDeclContext();
    while(ctx) {
        if (const clang::FunctionDecl *funcDecl = clang::dyn_cast<clang::FunctionDecl>(ctx)) {
            KernelFuncInfo kernelKey = GetKernelInfo(funcDecl);
            if (kernelFuncsKernelType_.find(kernelKey) != kernelFuncsKernelType_.end()) {
                g_kernelFuncType[kernelKey].second = KfcScene::Open;
                ASC_LOGD("Matmul Scene: [Open] store successfully, Find AscendC::KfcServer: %s at %s:%u, col:%u",
                    kernelKey.mangledName.c_str(),
                    kernelKey.fileName.c_str(),
                    kernelKey.lineNum,
                    kernelKey.colNum);
            }
        }
        ctx = ctx->getParent();
    }
}

bool ASTDeviceVisitor::VisitVarDecl(clang::VarDecl *varDecl)
{
    if (!varDecl) {
        ASC_LOGD("Null VarDecl: Possible AST corruption or invalid code");
        return true;
    }
    FindMatmulObjRegister(varDecl);
    FindKernelType(varDecl);
    FindOpSystemCfg(varDecl);
    return true;
}

ASTDeviceConsumer::ASTDeviceConsumer(clang::ASTContext &context, clang::CompilerInstance &compiler)
    : visitor_(context, compiler), compiler_(compiler)
{}

void ASTDeviceConsumer::HandleTranslationUnit(clang::ASTContext &context)
{
    std::string dumpPath = AscPlugin::InfoManager::GetInstance().GetTempPath() + "/ASTDumpDevice.txt";
    // Creating File Streams and Adapters
    std::ofstream dumpFile(dumpPath);
    if (dumpFile) {
        llvm::raw_os_ostream dumpStream(dumpFile);
        // Invoke the dump method and redirect the output
        const char * const outEnv = AscPlugin::LogManager::GetOutEnv();
        const char * const levelEnv = AscPlugin::LogManager::GetLevelEnv();
        if (outEnv != nullptr && outEnv[0] == '1' && levelEnv != nullptr && levelEnv[0] <= '0') {
            context.getTranslationUnitDecl()->dump(dumpStream);
            ASC_LOGD("AST dump device saved to: %s", dumpPath.c_str());
        }
    } else {
        ASC_LOGD("Failed to create AST dump device file");
    }
    visitor_.TraverseDecl(context.getTranslationUnitDecl());
}


std::string AscDiagnostic::GetSourceLine(
    const clang::SourceManager &srcManager, const clang::SourceLocation &srcLoc) const
{
    if (!srcLoc.isValid()) {
        return "<<<INVALID LOCATION>>>";
    }
    clang::FileID fileID = srcManager.getFileID(srcLoc);
    if (fileID.isInvalid() || !srcManager.getSLocEntry(fileID).isFile()) {
        return "<<<INVALID FILE>>>";
    }
    bool invalid = false;
    llvm::StringRef bufferData = srcManager.getBufferData(fileID, &invalid);
    if (invalid || bufferData.empty()) {
        return "<<<INVALID BUFFER>>>";
    }
    unsigned lineStartOffset = srcManager.getDecomposedLoc(srcLoc).second - (srcManager.getSpellingColumnNumber(srcLoc) - 1);
    const char *bufferStart = bufferData.data();
    const char *lineStart = bufferStart + lineStartOffset;
    const char *lineEnd = lineStart;
    while (*lineEnd != '\n' && *lineEnd != '\r' && lineEnd < bufferStart + bufferData.size()) {
        ++lineEnd;
    }
    return std::string(lineStart, lineEnd - lineStart);
}

constexpr uint32_t AST_MESSAGE_BUFFER = 100;
static const std::unordered_map<clang::DiagnosticsEngine::Level, const char*> DIAG_LEVEL_TO_STR = {
    {clang::DiagnosticsEngine::Level::Ignored, "Ignored"},
    {clang::DiagnosticsEngine::Level::Note, "Note"},
    {clang::DiagnosticsEngine::Level::Remark, "Remark"},
    {clang::DiagnosticsEngine::Level::Warning, "Warning"},
    {clang::DiagnosticsEngine::Level::Error, "Error"},
    {clang::DiagnosticsEngine::Level::Fatal, "Fatal"}
};

void AscDiagnostic::HandleDiagnostic(clang::DiagnosticsEngine::Level diagLevel, const clang::Diagnostic &info)
{
    const char * const outEnv = AscPlugin::LogManager::GetOutEnv();
    if (outEnv == nullptr || outEnv[0] != '1') {
        return;
    }
    const clang::SourceManager &srcManager = info.getSourceManager();
    clang::SourceLocation srcLoc = info.getLocation(); // 诊断发生位置

    clang::PresumedLoc pLoc = srcManager.getPresumedLoc(srcLoc);
    if (pLoc.isInvalid()) {
        return; // 位置无效时退出
    }
    const char *fileName = pLoc.getFilename(); // 文件名（完整路径）
    uint32_t line = static_cast<uint32_t>(pLoc.getLine());            // 行号
    uint32_t column = static_cast<uint32_t>(pLoc.getColumn());        // 列号
    std::string CodeLine = GetSourceLine(srcManager, srcManager.getSpellingLoc(srcLoc)); // 源码内容
    std::string indicator = std::string(column - 1, ' ') + "^"; // 指示符
    llvm::SmallString<AST_MESSAGE_BUFFER> message;
    info.FormatDiagnostic(message);
    if (diagLevel >= clang::DiagnosticsEngine::Level::Warning) {
        ASC_LOGD(
            "Location[%s:%u:%u]: AST %s : %s.\n%s\n%s",
            fileName,
            line,
            column,
            DIAG_LEVEL_TO_STR.at(diagLevel),
            message.c_str(),
            CodeLine.c_str(),
            indicator.c_str());
    }
}

std::unique_ptr<clang::ASTConsumer> DeviceAnalyzeAction::CreateASTConsumer(
    clang::CompilerInstance &compiler, llvm::StringRef inputFile)
{
    (void)inputFile;
    for (const auto &entry : compiler.getHeaderSearchOpts().UserEntries) {
        ASC_LOGD("Asc Device analyzer: Header Search Path: %s", entry.Path.c_str());
    }
    clang::DiagnosticsEngine &diagEngine = compiler.getDiagnostics();
    diagEngine.setClient(new AscDiagnostic(), /*ShouldOwnClient=*/true);
    diagEngine.setErrorLimit(0);
    return std::make_unique<ASTDeviceConsumer>(compiler.getASTContext(), compiler);
}
}  // namespace AscPlugin