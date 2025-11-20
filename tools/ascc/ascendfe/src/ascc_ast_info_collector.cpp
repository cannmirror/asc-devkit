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
 * \file ascc_ast_info_collector.cpp
 * \brief
 */


#include "ascc_ast_info_collector.h"

#include <memory>
#include <fstream>
#include <sstream>
#include <clang/AST/TemplateBase.h>
#include <clang/AST/PrettyPrinter.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/AST/TemplateName.h>
#include <llvm/Support/raw_os_ostream.h>

#include "ascc_log.h"
#include "ascc_types.h"
#include "ascc_info_aicore_function.h"
#include "ascc_info_callexpr.h"
#include "ascc_match_global_info.h"

namespace Ascc {
using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
constexpr uint32_t AST_MESSAGE_BUFFER = 100;
static const std::unordered_map<clang::DiagnosticsEngine::Level, const char*> DIAG_LEVEL_TO_STR = {
    {clang::DiagnosticsEngine::Level::Ignored, "Ignored"},
    {clang::DiagnosticsEngine::Level::Note, "Note"},
    {clang::DiagnosticsEngine::Level::Remark, "Remark"},
    {clang::DiagnosticsEngine::Level::Warning, "Warning"},
    {clang::DiagnosticsEngine::Level::Error, "Error"},
    {clang::DiagnosticsEngine::Level::Fatal, "Fatal"}
};

std::optional<std::string> KerenelInfoCollector::GetEnumNameForValue(const FunctionDecl* funcDecl, unsigned index,
    int64_t val, const PrintingPolicy& policy) const
{
    auto funcTemplate = funcDecl->getPrimaryTemplate();
    if (!funcTemplate) {
        return std::nullopt;
    }

    const auto* tplParams = funcTemplate->getTemplateParameters();
    if (!tplParams || index >= tplParams->size()) {
        return std::nullopt;
    }

    const auto* paramDecl = tplParams->getParam(index);
    const auto* nonTypeParam = dyn_cast<NonTypeTemplateParmDecl>(paramDecl);
    if (!nonTypeParam) {
        return std::nullopt;
    }

    QualType paramType = nonTypeParam->getType()
        .getNonReferenceType()
        .getUnqualifiedType();

    const auto* enumType = paramType->getAs<EnumType>();
    if (!enumType) {
        return std::nullopt;
    }

    const EnumDecl* enumDecl = enumType->getDecl();

    for (auto it = enumDecl->enumerator_begin();
         it != enumDecl->enumerator_end(); ++it) {
        const EnumConstantDecl* enumConst = *it;

        if (enumConst->getInitVal().getExtValue() != val) {
            continue;
        }

        PrintingPolicy pp(policy);
        pp.SuppressTagKeyword = true;
        std::string enumTypeName = paramType.getAsString(pp);

        return enumTypeName + "::" + enumConst->getNameAsString();
    }

    return std::nullopt;
}

void KerenelInfoCollector::KernelDeclHandle(const FunctionDecl *funcDecl, const MatchFinder::MatchResult &result) const
{
    ASTContext *context = result.Context;
    auto& srcMgr = context->getSourceManager();
    std::unique_ptr<MangleContext> mangleCtx(context->createMangleContext());
    const PrintingPolicy& policy = context->getPrintingPolicy();
    SourceLocation srcLoc = funcDecl->getLocation();
    FullSourceLoc fullLoc(srcLoc, *result.SourceManager);
    std::string file = fullLoc.getFileEntry()->getName().str();
    std::string funcName = funcDecl->getNameAsString();
    std::string nameScope = GetQualifiedScope(funcDecl);
    std::string returnType = funcDecl->getReturnType().getAsString();
    returnType = returnType == "_Bool" ? "bool" : returnType; // 将 _Bool 转换为 bool
    uint32_t line = static_cast<uint32_t>(fullLoc.getSpellingLineNumber());
    uint32_t endLine = static_cast<uint32_t>(srcMgr.getSpellingLineNumber(funcDecl->getEndLoc()));
    // 获取mangling名（mangled name）
    std::string mangledName;
    if (mangleCtx->shouldMangleDeclName(funcDecl)) {
        llvm::raw_string_ostream mangledStream(mangledName);
        mangleCtx->mangleName(funcDecl, mangledStream);
        mangledStream.flush();
    } else {
        mangledName = funcDecl->getNameAsString();
    }
    std::shared_ptr<AsccInfoFunction> allFuncInfo = std::dynamic_pointer_cast<AsccInfoFunction>(
        GetStorageInfo(file, AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION));
    std::string key = funcName + ":" + std::to_string(line);
    if (allFuncInfo->GetFunctionInfo(key) == nullptr) {
        allFuncInfo->AddFunction(key, AsccInfoFunction::FunctionInfo());
    }
    auto funcInfoPtr = allFuncInfo->GetFunctionInfo(key);
    if (funcInfoPtr == nullptr) {
        ASC_LOG_ASC_ERROR(AST, "funcInfoPtr is nullptr");
        return;
    }
    if (funcDecl->isTemplateInstantiation()) {
        funcInfoPtr->mangledToInstFuncInfo[mangledName] = std::make_shared<AsccInfoFunction::FunctionInfo>();
    }
    AsccInfoFunction::FunctionInfo &funcInfo =
        funcDecl->isTemplateInstantiation() ? (*(funcInfoPtr->mangledToInstFuncInfo[mangledName])) : *funcInfoPtr;
    // 通用获取
    funcInfo.funcName = funcName;
    funcInfo.definitionPos = file;
    funcInfo.lineNo = line;
    funcInfo.startLineNo = line;
    funcInfo.isTemplate = IsTemplate(funcDecl);
    funcInfo.isTempInst = funcDecl->isTemplateInstantiation();
    funcInfo.isTempExpSpec = funcDecl->isFunctionTemplateSpecialization();
    funcInfo.nameSpace = nameScope;
    funcInfo.returnType = returnType;
    funcInfo.endLineNo = endLine;
    funcInfo.manglingName = mangledName;

    // 获取参数列表
    std::stringstream paramsLogStr;
    for (const ParmVarDecl *parm : funcDecl->parameters()) {
        QualType paramType = parm->getType();
        std::string name = parm->getNameAsString();
        std::string type = (funcDecl->isTemplateInstantiation() || funcDecl->isFunctionTemplateSpecialization())
                                ? paramType.getCanonicalType().getAsString(policy) // 实例化层，需要带namespace
                                : paramType.getAsString(policy);
        bool isPointer = paramType->isPointerType();
        type = type == "_Bool" ? "bool" : type; // 将 _Bool 转换为 bool
        funcInfo.params.emplace_back(name, type, isPointer, ParamType::NORMAL_INPUT);

        // 维测log拼接
        std::string isPtr = isPointer ? "pointer" : "no pointer";
        paramsLogStr << type << " " << name << " (" << isPtr << ") ";
    }

    std::stringstream templateParamsLogStr;
    if (funcDecl->isTemplateInstantiation() || funcDecl->isFunctionTemplateSpecialization()) { // 在实例化层获取模板参数
        if (auto *specInfo = funcDecl->getTemplateSpecializationInfo()) {
            if (auto *tplDecl = specInfo->getTemplate()) {
                SourceLocation tplStartLoc = tplDecl->getSourceRange().getBegin();
                FullSourceLoc fullTplLoc(tplStartLoc, *result.SourceManager);
                SourceLocation specStartLoc = funcDecl->getSourceRange().getBegin();
                FullSourceLoc fullSpecLoc(specStartLoc, *result.SourceManager);
                if (funcDecl->isTemplateInstantiation()) {
                    funcInfo.startLineNo = static_cast<uint32_t>(fullTplLoc.getSpellingLineNumber());
                } else {
                    funcInfo.startLineNo = static_cast<uint32_t>(fullSpecLoc.getSpellingLineNumber());
                }
            }
            const TemplateArgumentList &args = *(specInfo->TemplateArguments);
            for (unsigned i = 0; i < args.size(); ++i) {
                auto kind = args[i].getKind();
                if (kind == TemplateArgument::Type) { // 模板类型参数
                    std::string type = args[i].getAsType().getAsString(policy);
                    type = type == "_Bool" ? "bool" : type; // 将 _Bool 转换为 bool
                    funcInfo.templateParams.emplace_back("", type, false, ParamType::TEMPLATE_TYPE);
                    templateParamsLogStr << type << " ";
                } else if (kind == TemplateArgument::Integral) {
                    auto val = args[i].getAsIntegral().getExtValue();
                    std::string vatStr;
                    bool isEnum = false;
                    auto enumName = GetEnumNameForValue(funcDecl, i, val, policy);
                    if (enumName) {
                        vatStr = *enumName;
                        isEnum = true;
                    } else {
                        vatStr = std::to_string(val);
                    }
                    funcInfo.templateParams.emplace_back("", vatStr, false, isEnum ? ParamType::TEMPLATE_ENUM :
                        ParamType::TEMPLATE_INT);
                    templateParamsLogStr << vatStr << " ";
                } else if (kind == TemplateArgument::Declaration) {
                    if (ValueDecl *decl = args[i].getAsDecl()) {
                        // 获取声明的完全限定名
                        std::string declName;
                        llvm::raw_string_ostream nameOS(declName);
                        decl->printQualifiedName(nameOS, policy);
                        funcInfo.templateParams.emplace_back("", declName, false, ParamType::TEMPLATE_DECL);
                        templateParamsLogStr << declName << " ";
                    }
                } else if (kind == TemplateArgument::Template) {
                    TemplateName tplName = args[i].getAsTemplate();
                    std::string tplStr;
                    llvm::raw_string_ostream tplOS(tplStr);
                    tplName.print(tplOS, policy);
                    funcInfo.templateParams.emplace_back("", tplStr, false, ParamType::TEMPLATE_TEMPLATE);
                    templateParamsLogStr << tplStr << " ";
                }
            }
        }
    } else if (auto *tplDecl = funcDecl->getDescribedFunctionTemplate()) { // 模板声明
        SourceLocation tplStartLoc = tplDecl->getSourceRange().getBegin();
        FullSourceLoc fullTplLoc(tplStartLoc, *result.SourceManager);
        funcInfo.startLineNo = static_cast<uint32_t>(fullTplLoc.getSpellingLineNumber());
        if (const auto *tplParams = tplDecl->getTemplateParameters()) {
            for (unsigned i = 0; i < tplParams->size(); ++i) {
                if (auto *typeParam = dyn_cast<TemplateTypeParmDecl>(tplParams->getParam(i))) {
                    // 类型参数（如 typename T）[2,5](@ref)
                    std::string name = typeParam->getNameAsString();
                    const std::string& typeStr = "typename";
                    funcInfo.templateParams.emplace_back(name, typeStr, false, ParamType::TEMPLATE_TYPE);
                    templateParamsLogStr << typeStr << " " << name << " ";
                } else if (auto *nonTypeParam = dyn_cast<NonTypeTemplateParmDecl>(tplParams->getParam(i))) {
                    // 非类型参数（如 int N）[2,8](@ref)
                    std::string name = nonTypeParam->getNameAsString();
                    std::string typeStr = nonTypeParam->getType().getAsString(policy);
                    typeStr = typeStr == "_Bool" ? "bool" : typeStr; // 将 _Bool 转换为 bool
                    ParamType paramType = (typeStr == "const auto &" || typeStr == "auto &") ? ParamType::TEMPLATE_DECL
                                                                                             : ParamType::TEMPLATE_INT;
                    funcInfo.templateParams.emplace_back(name, typeStr, false, paramType);
                    templateParamsLogStr << typeStr << " " << name << " ";
                } else if (auto *tplTplParam = dyn_cast<TemplateTemplateParmDecl>(tplParams->getParam(i))) {
                    // 模板模板参数（如 template<typename> class Container）[2](@ref)
                    std::string name = tplTplParam->getNameAsString();
                    std::string params;
                    if (auto *innerParams = tplTplParam->getTemplateParameters()) {
                        for (unsigned j = 0; j < innerParams->size(); ++j) {
                            params += j > 0 ? ", " : "";
                            params += "typename " + innerParams->getParam(j)->getNameAsString();
                        }
                    }
                    std::string typeStr = "template<" + params + "> class";
                    funcInfo.templateParams.emplace_back(name, typeStr, false, ParamType::TEMPLATE_TEMPLATE);
                    templateParamsLogStr << typeStr << " " << name << " ";
                }
            }
        }
    }
    if (!funcDecl->isTemplateInstantiation()) {
        std::vector<std::pair<uint32_t, uint32_t>> lineRanges;
        lineRanges.emplace_back(funcInfo.startLineNo, endLine);
        AsccInfoAicoreFunc::GetInstance().StoreKernelDefScope(file, lineRanges);
    }
    std::string isTempLogStr = funcInfo.isTemplate ? "Yes" : "No";
    std::string isInstantLogStr = funcInfo.isTempInst ? "Yes" : "No";
    std::string isExpSpecLogStr = funcInfo.isTempExpSpec ? "Yes" : "No";
    // 输出信息
    ASC_LOG_ASC_INFO(AST, "FunctionDecl Find :\n"
        "Location: [%s:%u]\n"
        "Function Name: [%s]\n"
        "Mangled Name: [%s]\n"
        "isTemplate: [%s]\n"
        "isInstant: [%s]\n"
        "isExpSpec: [%s]\n"
        "namespace: [%s]\n"
        "Return: [%s]\n"
        "Params: [%s]\n"
        "Template Params: [%s]\n",
        file.c_str(), static_cast<uint32_t>(line), funcName.c_str(), mangledName.c_str(), isTempLogStr.c_str(),
        isInstantLogStr.c_str(), isExpSpecLogStr.c_str(), nameScope.c_str(), returnType.c_str(),
        paramsLogStr.str().c_str(), templateParamsLogStr.str().c_str());
}

void KerenelInfoCollector::KernelCallHandle(
    const CUDAKernelCallExpr *kernelCall, const MatchFinder::MatchResult &result) const
{
    ASTContext *context = result.Context;
    std::unique_ptr<MangleContext> mangleCtx(context->createMangleContext());
    if (const FunctionDecl *callee = kernelCall->getDirectCallee()) {
        std::string calleeName = callee->getNameAsString();
        std::string calleeMangled;
        if (mangleCtx->shouldMangleDeclName(callee)) {
            llvm::raw_string_ostream mangledStream(calleeMangled);
            mangleCtx->mangleName(callee, mangledStream);
            mangledStream.flush();
        } else {
            calleeMangled = calleeName;
        }
        // 获取调用位置
        SourceLocation callLoc = kernelCall->getBeginLoc();
        FullSourceLoc fullCallLoc(callLoc, *result.SourceManager);
        uint32_t callLine = static_cast<uint32_t>(fullCallLoc.getSpellingLineNumber());
        uint32_t callCol = static_cast<uint32_t>(fullCallLoc.getSpellingColumnNumber());
        std::string callFile = fullCallLoc.getFileEntry()->getName().str();
        AsccInfoCallExpr callInfo = {calleeName, calleeMangled, callFile, callLine, callCol};
        AsccMatchGlobalInfo::GetInstance().AddGlobalKernelCallExpr(
            callInfo.file + ":" + std::to_string(callInfo.line) + ":" + std::to_string(callInfo.column), callInfo);
        ASC_LOG_ASC_INFO(AST, "FunctionCall Find :\n"
            "Location: [%s:%u:%u]\n"
            "Function Name: [%s]\n"
            "Mangled Name: [%s]\n",
            callFile.c_str(), callLine, callCol, calleeName.c_str(), calleeMangled.c_str());
    }
}

void KerenelInfoCollector::run(const MatchFinder::MatchResult &result)
{
    ASC_LOG_ASC_DEBUG(PREPROCESS, "AST match in");
    // 处理函数定义
    if (const FunctionDecl *funcDecl = result.Nodes.getNodeAs<FunctionDecl>("kernelDecl")) {
        ASC_LOG_ASC_DEBUG(PREPROCESS, "AST match Kernel Declaration");
        KernelDeclHandle(funcDecl, result);
    }
    // 处理内核调用
    if (const CUDAKernelCallExpr *kernelCall = result.Nodes.getNodeAs<CUDAKernelCallExpr>("kernelCall")) {
        ASC_LOG_ASC_DEBUG(PREPROCESS, "AST match Kernel Call");
        KernelCallHandle(kernelCall, result);
    }
}

std::string KerenelInfoCollector::GetQualifiedScope(const FunctionDecl *funcDecl) const
{
    std::string qualifiedName;
    const DeclContext *ctx = funcDecl->getDeclContext();

    // 向上遍历上下文获取命名空间链
    while (ctx && !isa<TranslationUnitDecl>(ctx)) {
        if (const auto *NS = dyn_cast<NamespaceDecl>(ctx)) {
            if (!NS->isAnonymousNamespace()) {
                qualifiedName = NS->getNameAsString() + "::" + qualifiedName;
            }
        }
        ctx = ctx->getParent();
    }
    return qualifiedName.empty() ? "" : qualifiedName;
}

bool KerenelInfoCollector::IsTemplate(const FunctionDecl* funcDecl) const
{
    return funcDecl->isTemplateInstantiation() || funcDecl->isFunctionTemplateSpecialization() ||
           funcDecl->getDescribedFunctionTemplate() != nullptr;
}

std::shared_ptr<AsccInfoBase> KerenelInfoCollector::GetStorageInfo(
    const std::string &file, const AscCursorTypes& infoType) const
{
    const auto& allInfo = AsccInfoStorage::GetInstance().GetAllInfos();
    if (allInfo.find(file) == allInfo.end() ||
        allInfo.at(file).find(infoType) == allInfo.at(file).end()) {
        auto functions = std::make_shared<AsccInfoFunction>();
        AsccInfoStorage::GetInstance().AddInfo(file, infoType, functions);
    }
    return allInfo.at(file).at(infoType);
}

void AsccDiagnostic::HandleDiagnostic(clang::DiagnosticsEngine::Level diagLevel, const clang::Diagnostic &info)
{
    if (AsccGlobalEnvManager::ascendSlogPrintToStdout == 0) {
        return;
    }
    const clang::SourceManager &srcManager = info.getSourceManager();
    clang::SourceLocation srcLoc = info.getLocation(); // 诊断发生位置

    clang::PresumedLoc pLoc = srcManager.getPresumedLoc(srcLoc);
    if (pLoc.isInvalid()) {
        return; // 位置无效时退出
    }
    const char *fileName = pLoc.getFilename(); // 文件名（完整路径）
    unsigned line = pLoc.getLine();            // 行号
    unsigned column = pLoc.getColumn();        // 列号
    std::string CodeLine = GetSourceLine(srcManager, srcManager.getSpellingLoc(srcLoc)); // 源码内容
    std::string indicator = std::string(column - 1, ' ') + "^"; // 指示符
    llvm::SmallString<AST_MESSAGE_BUFFER> message;
    info.FormatDiagnostic(message);
    if (diagLevel >= clang::DiagnosticsEngine::Level::Warning) {
        ASC_LOG_ASC_WARN(AST,
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

std::string AsccDiagnostic::GetSourceLine(
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

AsccASTConsumer::AsccASTConsumer()
{
    finder_.addMatcher(
        functionDecl(hasAttr(clang::attr::CUDAGlobal)).bind("kernelDecl"),
        &callback_
    );
    finder_.addMatcher(
        cudaKernelCallExpr().bind("kernelCall"),
        &callback_
    );
}
void AsccASTConsumer::HandleTranslationUnit(clang::ASTContext &context)
{
    PrintingPolicy policy = context.getPrintingPolicy();
    policy.adjustForCPlusPlus();
    policy.FullyQualifiedName = true;
    policy.SuppressTagKeyword = true;
    policy.SuppressScope = false;
    context.setPrintingPolicy(policy);
    Ascc::AsccGlobalEnvManager &envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    std::string dumpPath = envVar.asccTmpHostGenPath + "/ASTDump.txt";
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
    finder_.matchAST(context);
}

std::unique_ptr<clang::ASTConsumer> AsccFrontendAction::CreateASTConsumer(
    clang::CompilerInstance &compileInst, llvm::StringRef inputFile)
{
    (void)inputFile;
    clang::DiagnosticsEngine &diagEngine = compileInst.getDiagnostics();
    diagEngine.setClient(new AsccDiagnostic(), /*ShouldOwnClient=*/true);
    diagEngine.setErrorLimit(0);
    return std::make_unique<AsccASTConsumer>();
}

} // namespace Ascc