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
 * \file ascc_device_stub.cpp
 * \brief
 */

#include "ascc_device_stub.h"

#include <iostream>
#include <string>
#include <fstream>
#include <cctype>
#include <unordered_set>
#include <stack>
#include <llvm/Support/JSON.h>
#include "llvm/Support/FileSystem.h"

#include "ascc_log.h"
#include "ascc_types.h"
#include "ascc_compile_v220.h"
#include "ascc_host_stub.h"
#include "ascc_mangle.h"
#include "ascc_dump_flags.h"
#include "ascc_match_global_info.h"
#include "ascc_global_env_manager.h"
#include "ascc_argument_manager.h"
#include "ascc_compile_factory.h"
#include "ascc_utils.h"
#include "ascc_ast_utils.h"

namespace Ascc {
static constexpr uint32_t MIX_KERNEL_SECTION_MASK =
    (static_cast<uint32_t>(1) << static_cast<uint32_t>(CodeMode::KERNEL_TYPE_MIX_AIC_1_1)) |
    (static_cast<uint32_t>(1) << static_cast<uint32_t>(CodeMode::KERNEL_TYPE_MIX_AIC_1_2));
static constexpr uint32_t MIX_TYPE_MASK =
    (static_cast<uint32_t>(1) << static_cast<uint32_t>(CodeMode::KERNEL_TYPE_MIX_AIV_1_0)) |
    (static_cast<uint32_t>(1) << static_cast<uint32_t>(CodeMode::KERNEL_TYPE_MIX_AIC_1_0)) | MIX_KERNEL_SECTION_MASK;
static constexpr uint32_t AIV_COMPILE_MASK =
    (static_cast<uint32_t>(1) << static_cast<uint32_t>(CodeMode::KERNEL_TYPE_AIV_ONLY)) |
    (static_cast<uint32_t>(1) << static_cast<uint32_t>(CodeMode::KERNEL_TYPE_MIX_AIV_1_0)) | MIX_KERNEL_SECTION_MASK;
static constexpr uint32_t AIC_COMPILE_MASK =
    (static_cast<uint32_t>(1) << static_cast<uint32_t>(CodeMode::KERNEL_TYPE_AIC_ONLY)) |
    (static_cast<uint32_t>(1) << static_cast<uint32_t>(CodeMode::KERNEL_TYPE_MIX_AIC_1_0)) | MIX_KERNEL_SECTION_MASK;

static constexpr const char GLOBAL_ATTR[] = "__attribute__((cce_global)) ";

static const std::unordered_map<CodeMode, std::pair<std::string, std::string>> KERNEL_TYPE_TO_SECTION_TYPE = {
    {CodeMode::KERNEL_TYPE_AIC_ONLY, {"FunLevelKType", "K_TYPE_AIC"}},
    {CodeMode::KERNEL_TYPE_AIV_ONLY, {"FunLevelKType", "K_TYPE_AIV"}},
    {CodeMode::KERNEL_TYPE_MIX_AIV_1_0, {"FunLevelMixCoreType", "K_TYPE_MIX_AIV_MAIN"}},
    {CodeMode::KERNEL_TYPE_MIX_AIC_1_0, {"FunLevelMixCoreType", "K_TYPE_MIX_AIC_MAIN"}},
    {CodeMode::KERNEL_TYPE_MIX_AIC_1_1, {"FunLevelMixCoreType", "K_TYPE_MIX_AIC_MAIN"}},
    {CodeMode::KERNEL_TYPE_MIX_AIC_1_2, {"FunLevelMixCoreType", "K_TYPE_MIX_AIC_MAIN"}}
};

static const std::unordered_map<FuncMetaType, std::string> FUNC_METATYPE_TO_STR = {
    {FuncMetaType::F_TYPE_KTYPE, "F_TYPE_KTYPE"},
    {FuncMetaType::F_TYPE_CROSS_CORE_SYNC, "F_TYPE_CROSS_CORE_SYNC"},
    {FuncMetaType::F_TYPE_MAX, "F_TYPE_MAX"}
};

static const std::unordered_map<CodeMode, std::string> KERNEL_TYPE_TO_TASK_RATION = {
    {CodeMode::KERNEL_TYPE_MIX_AIV_1_0, "{{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 0, 1}"},
    {CodeMode::KERNEL_TYPE_MIX_AIC_1_0, "{{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 0}"},
    {CodeMode::KERNEL_TYPE_MIX_AIC_1_1, "{{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 1}"},
    {CodeMode::KERNEL_TYPE_MIX_AIC_1_2, "{{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2}"}
};

AsccDeviceStub::AsccDeviceStub(const ShortSoCVersion &coreType, const std::string &dstFilePath)
    : coreType_(coreType), dstFilePath_(dstFilePath)
{
    if (coreType_ != Ascc::ShortSoCVersion::ASCEND910B) {
        return;
    }
    auto& [aicArgs, aivArgs, mixAicArgs, mixAivArgs, mixAicArgsOneToOne, mixAivArgsOneToOne] = compileArgsList_;
    this->dumpSize_ = AsccDumpFlags::GetInstance().GetDumpSize();
    dstFilePath_ = CheckAndGetFullPath(dstFilePath);
    if (Ascc::AsccArgumentManager::GetInstance().GetPreTaskType() != PreTaskType::NONE) {
        dstFilePath_ = Ascc::AsccArgumentManager::GetInstance().GetModulePath();
    }
    this->deviceStubHeader_ = this->dstFilePath_ + "/device_stub_decl.h";
    aivArgs.customOption = "aiv";
    mixAivArgs.customOption = "aiv";
    mixAivArgsOneToOne.customOption = "aiv";
    aicArgs.customOption = "aic";
    mixAicArgs.customOption = "aic";
    mixAicArgsOneToOne.customOption = "aic";
    aivArgs.file = dstFilePath_ + "/" + KERNEL_TYPE_TO_FILE_NAME.at(CodeMode::KERNEL_TYPE_AIV_ONLY).at(0);
    aicArgs.file = dstFilePath_ + "/" + KERNEL_TYPE_TO_FILE_NAME.at(CodeMode::KERNEL_TYPE_AIC_ONLY).at(0);
    mixAicArgs.file = dstFilePath_ + "/" + KERNEL_TYPE_TO_FILE_NAME.at(CodeMode::KERNEL_TYPE_MIX_AIC_1_0).at(0);
    mixAivArgs.file = dstFilePath_ + "/" + KERNEL_TYPE_TO_FILE_NAME.at(CodeMode::KERNEL_TYPE_MIX_AIV_1_0).at(0);
    mixAicArgsOneToOne.file =
        dstFilePath_ + "/" + KERNEL_TYPE_TO_FILE_NAME.at(CodeMode::KERNEL_TYPE_MIX_AIC_1_1).at(0);
    mixAivArgsOneToOne.file =
        dstFilePath_ + "/" + KERNEL_TYPE_TO_FILE_NAME.at(CodeMode::KERNEL_TYPE_MIX_AIC_1_1).at(1);
    aivArgs.outputPath = dstFilePath_ + "/device_stub_aiv.o";
    aicArgs.outputPath = dstFilePath_ + "/device_stub_aic.o";
    mixAicArgs.outputPath = dstFilePath_ + "/device_stub_mix_aic.o";
    mixAivArgs.outputPath = dstFilePath_ + "/device_stub_mix_aiv.o";
    mixAicArgsOneToOne.outputPath = dstFilePath_ + "/device_stub_mix_aic_1_1.o";
    mixAivArgsOneToOne.outputPath = dstFilePath_ + "/device_stub_mix_aiv_1_1.o";
    mixAicArgs.definitions.emplace_back("__MIX_CORE_MACRO__=1");
    mixAivArgs.definitions.emplace_back("__MIX_CORE_MACRO__=1");
    mixAicArgsOneToOne.definitions = {"__MIX_CORE_AIC_RATION__=1", "__MIX_CORE_MACRO__=1"};
    mixAivArgsOneToOne.definitions = {"__MIX_CORE_AIC_RATION__=1", "__MIX_CORE_MACRO__=1"};
    UpdateDumpCompileOption();
};

void AsccDeviceStub::UpdateDumpCompileOption()
{
    for (auto& args : compileArgsList_) {
        args.definitions.emplace_back(std::string("ONE_CORE_DUMP_SIZE=" + std::to_string(dumpSize_)));
        if (Ascc::AsccDumpFlags::GetInstance().IsDumpOn()) {
            args.definitions.emplace_back("ASCENDC_DUMP=1");
        }
    }
}

void AsccDeviceStub::UpdateWorkFlowCompileOption()
{
    for (auto& args : compileArgsList_) {
        args.options.emplace_back("--cce-disable-kernel-global-attr-check");
    }
}

AsccStatus AsccDeviceStub::Process()
{
    if (!AsccMatchGlobalInfo::GetInstance().HasKernelCall()) {
        ASC_LOG_ASC_WARN(DEVICE_STUB, "Files have no kernel info!");
        return AsccStatus::SUCCESS;
    }
    stubHeaderCode_.open(deviceStubHeader_, std::ios::app);
    AsccInfoStorage &storage = AsccInfoStorage::GetInstance();
    const auto &allInfos = storage.GetAllInfos();
    UpdateNewWorkflowFlag();
    std::string incFile = isNewWorkflow_ ? newWorkflowFile_ : AsccArgumentManager::GetInstance().GetInputFile();
    GenHeadCode(stubHeaderCode_, incFile);
    for (const auto &[filePath, fileInfo] : allInfos) {
        if (GenDeviceStubCode(filePath, fileInfo) == AsccStatus::FAILURE) {
            ASC_LOG_ASC_ERROR(DEVICE_STUB, "Code generate failed!");
            return AsccStatus::FAILURE;
        }
    }
    stubHeaderCode_.close();

    // if for sub_module path task, do not need to compile device stub files
    if (Ascc::AsccArgumentManager::GetInstance().GetPreTaskType() != PreTaskType::NONE) {
        return AsccStatus::SUCCESS;
    }

    ASCC_CHECK((CompileDeviceStub() == AsccStatus::SUCCESS),
        { ASC_LOG_ASC_ERROR(DEVICE_STUB, "Device compile failed!"); });
    return AsccStatus::SUCCESS;
}

AsccStatus AsccDeviceStub::GenerateJsonFiles() const
{
    bool enableDFX = Ascc::AsccDumpFlags::GetInstance().IsDumpOn();
    uint32_t dumpSize = Ascc::AsccDumpFlags::GetInstance().GetDumpSize();
    AsccInfoStorage &storage = AsccInfoStorage::GetInstance();
    const auto &allInfos = storage.GetAllInfos();
    if (allInfos.empty()) {
        ASC_LOG_ASC_WARN(DEVICE_STUB, "In GenerateJsonFiles, files have no kernel info!");
        return AsccStatus::SUCCESS;
    }
    for (const auto &[filePath, fileInfo] : allInfos) {
        auto funcsInfo = GetFileInfo<Funcs>(fileInfo);
        std::string jsonName = RemoveSuffix(filePath) + ".json";
        uint32_t funcCallCount = 0;          // records how many <<<>>> functions are called, thus need to generate stub
        llvm::json::Object jsonObj;
        for (auto it = funcsInfo->Begin(); it != funcsInfo->End(); ++it) {
            const auto& funcInfo = it->second;
            if (funcInfo.IsTempDecl()) {
                if (funcInfo.mangledToInstFuncInfo.size() == 0) {
                    WarnNoFuncCalls(funcInfo);
                    continue;
                }
                for (const auto &[manglingName, mangledFuncInfo] : funcInfo.mangledToInstFuncInfo) {
                    (void)manglingName;
                    if (GenerateJsonObj(jsonObj, *mangledFuncInfo, this->dstFilePath_, filePath, enableDFX, dumpSize)) {
                        funcCallCount += 1;
                    }
                }
            } else {
                if (GenerateJsonObj(jsonObj, funcInfo, this->dstFilePath_, filePath, enableDFX, dumpSize)) {
                    funcCallCount += 1;
                }
            }
        }

        // has <<<>>> function called, needs to generate stub files
        if (funcCallCount > 0) {
            ASCC_CHECK((MergeJsonObjs(jsonObj, jsonName) == AsccStatus::SUCCESS),
                {HandleError("Merge json objects failed.");});
        }
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccDeviceStub::GenTmpDeviceCode(const std::string &inputFile, const std::string &outputFile,
    const std::unordered_map<std::string, std::unordered_set<std::string>>& kernelCallLinesColumns)
{
    if (!Ascc::IsPathLegal(inputFile) || !Ascc::IsParentDirValid(inputFile)) {
        ASC_LOG_ASC_ERROR(PREPROCESS, "inputFile path [%s] does not exist!", inputFile.c_str());
        return AsccStatus::FAILURE;
    }
    std::ifstream infile(inputFile);
    if (!infile.is_open()) {
        ASC_LOG_ASC_ERROR(PREPROCESS, "Failed to open input file: [%s]!", inputFile.c_str());
        return AsccStatus::FAILURE;
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(infile, line)) {
        lines.emplace_back(line);
    }
    infile.close();
    const auto& lineColumnSet = kernelCallLinesColumns.at(inputFile);
    for (const auto& iter : lineColumnSet) {
        size_t colonPos = iter.find(":");   // iter format: "row:col"
        uint32_t row = std::stoul(iter.substr(0, colonPos));
        uint32_t col = std::stoul(iter.substr(colonPos + 1));
        uint32_t tmpRow = row - 1;
        while (lines[tmpRow].find(";", col - 1) == std::string::npos) {
            lines[tmpRow] = std::string("/*device_stub*/");
            ++tmpRow;
        }
        size_t semicolonPos;
        if (tmpRow == row - 1) {
            semicolonPos = lines[tmpRow].find(";", col - 1);
        } else {
            semicolonPos = lines[tmpRow].find(";");
        }
        lines[tmpRow].replace(col - 1, semicolonPos - (col - 1) + 1, std::string("/*device_stub*/"));
    }
    if (!Ascc::IsPathLegal(outputFile) || !Ascc::IsParentDirValid(outputFile)) {
        ASC_LOG_ASC_ERROR(PREPROCESS, "outputFile path [%s] does not exist!", outputFile.c_str());
        return AsccStatus::FAILURE;
    }
    std::ofstream outfile(outputFile);
    if (!outfile.is_open()) {
        ASC_LOG_ASC_ERROR(PREPROCESS, "Failed to open output file: [%s]!", outputFile.c_str());
        return AsccStatus::FAILURE;
    }
    for (const auto &content : lines) {
        outfile << content << "\n";
    }
    outfile.close();
    this->newWorkflowFile_ = outputFile;
    return AsccStatus::SUCCESS;
}

AsccStatus AsccDeviceStub::UpdateNewWorkflowFlag()
{
    const auto& inputFile = AsccArgumentManager::GetInstance().GetInputFile();
    const auto& kernelCallMap = AsccMatchGlobalInfo::GetInstance().GetGlobalKernelCallLineColumn();
    if (kernelCallMap.size() > 1 || kernelCallMap.find(inputFile) == kernelCallMap.end()) {
        this->isNewWorkflow_ = false;
        UpdateWorkFlowCompileOption();
        return AsccStatus::SUCCESS;
    }
    this->isNewWorkflow_ = true;
    std::string outputFile = dstFilePath_ + "/stub_" + Ascc::GetFileName(inputFile);
    return GenTmpDeviceCode(inputFile, outputFile, kernelCallMap);
}

AsccStatus AsccDeviceStub::GenDeviceStubCode(const std::string &filePath, const AsccInfoStorage::FileInfos &fileInfo)
{
    auto funcsInfo = Ascc::GetFileInfo<Funcs>(fileInfo);
    ASCC_CHECK((DeviceStubCodeImpl(filePath, funcsInfo) == AsccStatus::SUCCESS),
        {ASC_LOG_ASC_ERROR(DEVICE_STUB, "Device stub generate fail! Please check log!");});
    return AsccStatus::SUCCESS;
}

AsccStatus AsccDeviceStub::ParamInit(const FuncInfo& funcInfo)
{
    this->kernelType_ = funcInfo.kernelType;
    this->codeModeMask_ = static_cast<uint32_t>(1) << static_cast<uint32_t>(this->kernelType_);
    this->beginExtraCounter_ = 0;
    this->endExtraCounter_ = 0;
    this->kernelCounter_ = 0;
    this->stubFiles_.clear();
    this->haveWorkspace_ = false;
    return AsccStatus::SUCCESS;
}

AsccStatus AsccDeviceStub::DeviceStubCodeImpl(const std::string &filePath, const std::shared_ptr<Funcs> &funcsInfo)
{
    if (funcsInfo == nullptr) {
        ASC_LOG_ASC_WARN(DEVICE_STUB, "File %s has no kernel function info!", filePath.c_str());
        return AsccStatus::SUCCESS;
    }

    std::string incFile = isNewWorkflow_ ? newWorkflowFile_ : AsccArgumentManager::GetInstance().GetInputFile();
    for (auto it = funcsInfo->Begin(); it != funcsInfo->End(); ++it) {
        const auto& funcInfo = it->second;
        ParamInit(funcInfo);
        this->stubFiles_ = {dstFilePath_ + "/" + KERNEL_TYPE_TO_FILE_NAME.at(kernelType_).at(0)};
        if (KERNEL_TYPE_TO_FILE_NAME.at(kernelType_).size() > 1) {
            this->stubFiles_.emplace_back(dstFilePath_ + "/" + KERNEL_TYPE_TO_FILE_NAME.at(kernelType_).at(1));
        }
        for (const auto& stubFile : stubFiles_) {
            std::ofstream stubCode(stubFile, std::ios::app);
            if (stubIsGen_.count(stubFile) == 0) {
                stubCode << "#include \"" << this->deviceStubHeader_ << "\"\n\n";
            }
            ASCC_CHECK((GenStubFunc(stubCode, funcInfo) == AsccStatus::SUCCESS),
                { ASC_LOG_ASC_ERROR(DEVICE_STUB, "Device stub stub function generate fail!"); });
            stubCode.close();
            ++kernelCounter_;
        }
        for (const auto& stubFile : stubFiles_) {
            stubIsGen_.emplace(stubFile);
        }
    }

    return AsccStatus::SUCCESS;
}

void AsccDeviceStub::GenHeadCode(std::ofstream &stubCode, const std::string &filePath) const
{
    stubCode << "#undef __global__\n";
    stubCode << "#define __global__ inline\n";
    stubCode << "#include \"" << filePath << "\"\n";
    stubCode << "#undef __global__\n";
    stubCode << "#if ASCENDC_CPU_DEBUG\n";
    stubCode << "#define __global__\n";
    stubCode << "#else\n";
    stubCode << "#define __global__ __attribute__((cce_kernel))\n";
    stubCode << "#endif\n\n";
    stubCode << "#ifndef ONE_CORE_DUMP_SIZE\n";
    stubCode << "#define ONE_CORE_DUMP_SIZE " << std::to_string(this->dumpSize_) << " * 1\n";
    stubCode << "#endif\n\n";
}

AsccDeviceStub::FuncInfo AsccDeviceStub::GetNewFunctionInfo(
    const FuncInfo &funcInfo, const bool dumpTypeIsNotNone, const bool dumpAscendCStamp)
{
    this->beginExtraCounter_ = 0;
    this->endExtraCounter_ = 0;
    FuncInfo newFuncInfo = funcInfo;
    newFuncInfo.returnType = std::string("void");
    newFuncInfo.funcName = std::string("__device_stub__" + newFuncInfo.funcName);
    if (IsMix()) {
        ++this->beginExtraCounter_;
        newFuncInfo.params.insert(
            newFuncInfo.params.begin(), ParamInfo("ffts_addr", "uint8_t *", true, ParamType::NORMAL_INPUT));
    }
    if (dumpTypeIsNotNone || dumpAscendCStamp) {
        ++this->beginExtraCounter_;
        newFuncInfo.params.insert(
            newFuncInfo.params.begin(), ParamInfo("dumpAddr", "uint8_t *", true, ParamType::NORMAL_INPUT));
    }
    ++this->endExtraCounter_;
    newFuncInfo.params.insert(
        newFuncInfo.params.end(), ParamInfo("overflow_status", "uint8_t *", true, ParamType::NORMAL_INPUT));
    return newFuncInfo;
}

std::vector<std::string> AsccDeviceStub::GetAllNestedNameSpace(const std::string &nameSpacePrefix) const
{
    if (nameSpacePrefix.empty()) {
        ASC_LOG_ASC_INFO(DEVICE_STUB, "Has no nameSpace");
        return {};
    }
    std::vector<std::string> nameSpaceStack;
    std::string nameSpace = std::string();
    for (const auto &c : nameSpacePrefix) {
        if (c == ':') {
            if (!nameSpace.empty()) {
                nameSpaceStack.emplace_back(nameSpace);
                nameSpace.clear();
            }
            continue;
        }
        nameSpace += c;
    }
    if (!nameSpace.empty()) {
        nameSpaceStack.emplace_back(nameSpace);
    }
    return nameSpaceStack;
}

void AsccDeviceStub::GenStubFuncDefinition(std::ofstream &stubCode, const FuncInfo &newFuncInfo)
{
    std::string paramsStr = GetStubFuncDefParamsAndWorksapceFlag(newFuncInfo);
    std::string paramsStrNoFfts = GetStubFuncDefParamsAndWorksapceFlag(newFuncInfo, /*skipFfts*/true);
    stubCode << "inline __aicore__ " << newFuncInfo.returnType << " " << newFuncInfo.funcName;
    stubCode << "(" << paramsStr << ")" << std::endl;
    if (kernelCounter_ > 0) { // mix gen code only once
        return;
    }
    stubHeaderCode_ << "inline __aicore__ " << newFuncInfo.returnType << " " << newFuncInfo.funcName;
    stubHeaderCode_ << "(" << paramsStr << ");" << std::endl;
    if (paramsStrNoFfts != paramsStr) {
        stubHeaderCode_ << "inline __aicore__ " << newFuncInfo.returnType << " " << newFuncInfo.funcName;
        stubHeaderCode_ << "(" << paramsStrNoFfts << ");" << std::endl;
    }
}

void AsccDeviceStub::GenTemplateStubFuncDefinition(
    std::ofstream &stubCode, const FuncInfo &newFuncInfo, std::string &originCallTempParams)
{
    std::string paramsStr = GetStubFuncDefParamsAndWorksapceFlag(newFuncInfo);
    std::string paramsStrNoFfts = GetStubFuncDefParamsAndWorksapceFlag(newFuncInfo, /*skipFfts*/true);
    size_t counter = 0;
    std::string templateParamsStr = std::string();
    for (auto &param : newFuncInfo.templateParams) {
        ++counter;
        templateParamsStr.append(param.type + " " + param.paraName);
        originCallTempParams.append(param.paraName);
        if (counter < newFuncInfo.templateParams.size()) {
            templateParamsStr.append(", ");
            originCallTempParams.append(", ");
        }
    }
    stubCode << "template<" << templateParamsStr << ">" << std::endl;
    stubCode << "inline __aicore__ " << newFuncInfo.returnType << " " << newFuncInfo.funcName;
    stubCode << "(" << paramsStr << ")" << std::endl;
    if (kernelCounter_ > 0) { // mix gen code only once
        return;
    }
    stubHeaderCode_ << "template<" << templateParamsStr << ">" << std::endl;
    stubHeaderCode_ << "inline __aicore__ " << newFuncInfo.returnType << " " << newFuncInfo.funcName;
    stubHeaderCode_ << "(" << paramsStr << ");" << std::endl;
    if (paramsStrNoFfts != paramsStr) {
        stubHeaderCode_ << "template<" << templateParamsStr << ">" << std::endl;
        stubHeaderCode_ << "inline __aicore__ " << newFuncInfo.returnType << " " << newFuncInfo.funcName;
        stubHeaderCode_ << "(" << paramsStrNoFfts << ");" << std::endl;
    }
}

void AsccDeviceStub::GenTemplateExpSpecStubFuncDefinition(
    std::ofstream &stubCode, const FuncInfo &newFuncInfo, std::string &originCallTempParams)
{
    std::string paramsStr = GetStubFuncDefParamsAndWorksapceFlag(newFuncInfo);
    size_t counter = 0;
    std::string templateParamsStr = std::string();
    for (auto &param : newFuncInfo.templateParams) {
        ++counter;
        templateParamsStr.append(param.type + " " + param.paraName);
        originCallTempParams.append(param.paraName);
        if (counter < newFuncInfo.templateParams.size()) {
            templateParamsStr.append(", ");
            originCallTempParams.append(", ");
        }
    }
    stubCode << "template<>" << std::endl;
    stubCode << "inline __aicore__ " << newFuncInfo.returnType << " " << newFuncInfo.funcName;
    stubCode << "<" << templateParamsStr << ">(" << paramsStr << ")" << std::endl;
}

std::string AsccDeviceStub::GetStubFuncDefParamsAndWorksapceFlag(const FuncInfo& newFuncInfo, bool skipFfts)
{
    std::string paramsStr = std::string();
    size_t counter = 0;
    for (const auto &param : newFuncInfo.params) {
        counter++;
        if (skipFfts && param.paraName == std::string("ffts_addr")) {
            continue;
        }
        if (param.isPointer) {
            paramsStr.append(GLOBAL_ATTR);
        }
        paramsStr.append(param.type + " " + param.paraName);
        if (counter < newFuncInfo.params.size()) {
            paramsStr.append(", ");
        }
        if (param.paraName == "workspace") {
            haveWorkspace_ = true;
        }
    }
    return paramsStr;
}

void AsccDeviceStub::StubFuncDumpAndHardSyncImpl(std::ofstream &stubCode,
    const bool dumpTypeIsNotNone, const bool dumpAscendCStamp, const bool dumpTypeIsPrintf) const
{
    if (dumpTypeIsNotNone || dumpAscendCStamp) {
        if (IsMix()) {
            stubCode << "    AscendC::InitDump(true, dumpAddr, ONE_CORE_DUMP_SIZE);\n";
        } else {
            stubCode << "    AscendC::InitDump(false, dumpAddr, ONE_CORE_DUMP_SIZE);\n";
        }
        if (dumpAscendCStamp) {
            stubCode << "    "
                        "AscendC::AscendCTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_INIT_"
                        "DUMP));\n";
        }
    }
    if (IsMix()) {
        stubCode << "    icache_preload(1);\n";
        stubCode << "    if (ffts_addr != nullptr) {\n";
        stubCode << "        set_ffts_base_addr((uint64_t)ffts_addr);\n";
        stubCode << "    }\n";
        if (dumpAscendCStamp) {
            stubCode <<
                "    "
                "AscendC::AscendCTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_FFTS_ADDR));\n";
        }
    }
    if (dumpTypeIsPrintf) {
        stubCode << "    uint64_t __ascendc_tStamp = 0;\n";
        stubCode << "    uint64_t __ascendc_version = 0;\n";
        stubCode << "     __gm__ char* __ascendc_versionStr = nullptr;\n";
        stubCode << "    GetCannVersion(__ascendc_versionStr, __ascendc_version, __ascendc_tStamp);\n";
        stubCode << "    if (__ascendc_tStamp == 0) {\n";
        stubCode << "        AscendC::printf(\"[WARNING]: CANN TimeStamp is invalid, ";
        stubCode << "CANN TimeStamp is %u\\n\", __ascendc_tStamp);\n";
        stubCode << "    } else {\n";
        stubCode << "        AscendC::printf(\"CANN Version: %s, TimeStamp: %u\\n\", ";
        stubCode << "(__gm__ const char*)(__ascendc_versionStr), __ascendc_tStamp);\n";
        stubCode << "    }\n";
    }
}

void AsccDeviceStub::StubFuncWorkSpaceImpl(std::ofstream &stubCode, const bool dumpAscendCStamp,
    const bool hasKfcServer) const
{
    if (!haveWorkspace_) {
        return;
    }

    stubCode << "    GM_ADDR workspace_param;\n";
    stubCode << "    GM_ADDR workspace_usr;\n";
    stubCode << "    workspace_param = workspace;\n";
    if (IsMix()) {
        stubCode << "    if (workspace_param == nullptr) {\n";
        stubCode << "        return;\n";
        stubCode << "    }\n";
    }
    stubCode << "    AscendC::SetSysWorkspaceForce(workspace_param);\n";
    stubCode << "    workspace_usr = AscendC::GetUserWorkspace(workspace_param);\n";
    if (IsMix() && hasKfcServer) {
        stubCode << "    if constexpr (g_coreType == AscendC::AIC) {\n";
        stubCode << "        matmul::clearWorkspace(workspace_param);\n";
        if (dumpAscendCStamp) {
            stubCode << "        AscendC::AscendCTimeStamp(static_cast<uint32_t>";
            stubCode << "(AscendC::TimeStampId::TIME_STAMP_WRAP_CLEAR_WK_SPAC));\n";
        }
        stubCode << "    }\n";
    }
    stubCode << "    workspace = workspace_usr;\n";
}

void AsccDeviceStub::GetStubFuncDumpInfo(bool &dumpTypeIsNotNone, bool &dumpTypeIsPrintf, bool &dumpAscendCStamp) const
{
    bool isDumpCloseManual = AsccDumpFlags::GetInstance().GetIsDumpCloseManual();
    bool printfFlag = AsccDumpFlags::GetInstance().GetPrintfFlag();
    dumpTypeIsNotNone = !isDumpCloseManual && (printfFlag || AsccDumpFlags::GetInstance().GetAssertFlag());
    dumpTypeIsPrintf = !isDumpCloseManual && printfFlag;
    dumpAscendCStamp = false;
}

void AsccDeviceStub::StubFuncCallImpl(std::ofstream &stubCode,
    const FuncInfo &funcInfo, const std::string &originCallTempParams) const
{
    std::string paramsImplStr = std::string();
    size_t counter = 0;
    for (const auto &param : funcInfo.params) {
        counter++;
        paramsImplStr.append(param.paraName);
        if (counter < funcInfo.params.size()) {
            paramsImplStr.append(", ");
        }
    }
    if (!funcInfo.nameSpace.empty()) {
        stubCode << "    " << funcInfo.nameSpace << funcInfo.funcName;
    } else {
        stubCode << "    " << funcInfo.funcName;
    }
    if (!originCallTempParams.empty()) {
        stubCode << "<" << originCallTempParams << ">";
    }
    stubCode << "(" << paramsImplStr << ");\n";
}

AsccStatus AsccDeviceStub::GenNormalFuncSymbolImpl(std::ofstream &stubCode, const FuncInfo &newFuncInfo)
{
    if (!AsccMatchGlobalInfo::GetInstance().IsCalled(newFuncInfo.manglingName)) {
        return AsccStatus::SUCCESS;
    }
    std::string cmpStr = "__device_stub__";
    std::string newFuncName = (cmpStr + newFuncInfo.manglingName) == newFuncInfo.funcName
                                  ? newFuncInfo.manglingName + "_call"
                                  : newFuncInfo.manglingName;
    auto& ascFixedMangleMap = AsccMangle::GetInstance().GetOriginToFixedMangledNames();
    ascFixedMangleMap[newFuncInfo.manglingName] = newFuncName;
    std::string paramDecl = std::string();
    std::string paramCall = std::string();
    std::string tempParamCall = std::string();
    for (size_t i = 0; i < newFuncInfo.params.size(); ++i) {
        if (newFuncInfo.params[i].isPointer) {
            paramDecl.append(GLOBAL_ATTR);
        }
        paramDecl.append(newFuncInfo.params[i].type + " " + newFuncInfo.params[i].paraName);
        paramCall.append(newFuncInfo.params[i].paraName);
        if (i < newFuncInfo.params.size() - 1) {
            paramDecl.append(", ");
            paramCall.append(", ");
        }
    }
    for (size_t i = 0; i < newFuncInfo.templateParams.size(); ++i) {
        tempParamCall.append(newFuncInfo.templateParams[i].type);
        if (i < newFuncInfo.templateParams.size() - 1) {
            tempParamCall.append(", ");
        }
    }
    stubCode << "extern \"C\" __global__ __aicore__ " << newFuncInfo.returnType << " " << newFuncName;
    if (kernelType_ == CodeMode::KERNEL_TYPE_MIX_AIC_1_0 ||
        (stubFiles_.size() > 1 && kernelCounter_ == 0)) {
        stubCode << "_mix_aic";
    }
    if (kernelType_ == CodeMode::KERNEL_TYPE_MIX_AIV_1_0 ||
        (stubFiles_.size() > 1 && kernelCounter_ == 1)) {
        stubCode << "_mix_aiv";
    }
    stubCode << "(" << paramDecl << ")\n";
    stubCode << "{\n" << newFuncInfo.nameSpace << newFuncInfo.funcName;
    if (newFuncInfo.IsTempSpec()) {
        stubCode << "<" << tempParamCall << ">";
    }
    stubCode << "(" << paramCall << ");\n}\n\n";
    return AsccStatus::SUCCESS;
}

AsccStatus AsccDeviceStub::GenTempFuncSymbolImpl(std::ofstream &stubCode, const FuncInfo &newFuncInfo)
{
    for (const auto& [manglingName, instFuncInfo] : newFuncInfo.mangledToInstFuncInfo) {
        if (!AsccMatchGlobalInfo::GetInstance().IsCalled(manglingName)) {
            continue;
        }
        auto& ascFixedMangleMap = AsccMangle::GetInstance().GetOriginToFixedMangledNames();
        ascFixedMangleMap[instFuncInfo->manglingName] = instFuncInfo->manglingName;
        std::string paramDecl = std::string();
        std::string paramCall = std::string();
        std::string tempParamCall = std::string();
        for (size_t i = 0; i < this->beginExtraCounter_; ++i) {
            paramDecl.append(GLOBAL_ATTR + newFuncInfo.params[i].type + " " + newFuncInfo.params[i].paraName + ", ");
            paramCall.append(newFuncInfo.params[i].paraName + ", ");
        }
        for (const ParamInfo& paramInfo : instFuncInfo->params) {
            if (paramInfo.isPointer) {
                paramDecl.append(GLOBAL_ATTR);
            }
            paramDecl.append(paramInfo.type + " " + paramInfo.paraName + ", ");
            paramCall.append(paramInfo.paraName + ", ");
        }
        for (int i = static_cast<int>(this->endExtraCounter_); i > 0; --i) {
            size_t index = newFuncInfo.params.size() - i;
            paramDecl.append(GLOBAL_ATTR + newFuncInfo.params[index].type + " " + newFuncInfo.params[index].paraName);
            paramCall.append(newFuncInfo.params[index].paraName);
            if (index < newFuncInfo.params.size() - 1) {
                paramDecl.append(", ");
                paramCall.append(", ");
            }
        }
        for (size_t i = 0; i < instFuncInfo->templateParams.size(); ++i) {
            tempParamCall.append(instFuncInfo->templateParams[i].type);
            if (i < instFuncInfo->templateParams.size() - 1) {
                tempParamCall.append(", ");
            }
        }
        stubCode << "extern \"C\" __global__ __aicore__ " << instFuncInfo->returnType << " " << manglingName;
        if (kernelType_ == CodeMode::KERNEL_TYPE_MIX_AIC_1_0 || (stubFiles_.size() > 1 && kernelCounter_ == 0)) {
            stubCode << "_mix_aic";
        }
        if (kernelType_ == CodeMode::KERNEL_TYPE_MIX_AIV_1_0 || (stubFiles_.size() > 1 && kernelCounter_ == 1)) {
            stubCode << "_mix_aiv";
        }
        stubCode << "(" << paramDecl << ")\n";
        stubCode << "{\n" << newFuncInfo.nameSpace << newFuncInfo.funcName << "<" << tempParamCall << ">";
        stubCode << "(" << paramCall << ");\n}\n\n";
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccDeviceStub::StubFuncInstImpl(std::ofstream &stubCode, const FuncInfo &newFuncInfo)
{
    if (newFuncInfo.IsTempDecl() && newFuncInfo.mangledToInstFuncInfo.size() > 0) {
        return GenTempFuncSymbolImpl(stubCode, newFuncInfo);
    } else if (newFuncInfo.IsTempSpec() || !newFuncInfo.isTemplate) {
        return GenNormalFuncSymbolImpl(stubCode, newFuncInfo);
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccDeviceStub::GetManglingList(std::vector<std::string> &manglingNameList, const FuncInfo& funcInfo) const
{
    const auto& ascFixedMangleMap = AsccMangle::GetInstance().GetOriginToFixedMangledNames();
    if (!funcInfo.isTemplate || funcInfo.IsTempSpec()) {
        if (ascFixedMangleMap.find(funcInfo.manglingName) == ascFixedMangleMap.end()) {
            return AsccStatus::SUCCESS;
        }
        manglingNameList.emplace_back(ascFixedMangleMap.at(funcInfo.manglingName));
        return AsccStatus::SUCCESS;
    }
    for (const auto& instFuncInfo : funcInfo.mangledToInstFuncInfo) {
        const auto& manglingName = instFuncInfo.first;
        if (ascFixedMangleMap.find(manglingName) == ascFixedMangleMap.end()) {
            continue;
        }
        manglingNameList.emplace_back(ascFixedMangleMap.at(manglingName));
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccDeviceStub::StubFuncKtypeSectionImpl(std::ofstream &stubCode, const FuncInfo& funcInfo)
{
    const bool isGenKtypeSection = coreType_ == ShortSoCVersion::ASCEND910B;
    if (!isGenKtypeSection) {
        ASC_LOG_ASC_INFO(DEVICE_STUB, "No need to generate kernel type section.");
        return AsccStatus::SUCCESS;
    }
    std::vector<std::string> manglingNameList;
    if (GetManglingList(manglingNameList, funcInfo) == AsccStatus::FAILURE) {
        ASC_LOG_ASC_ERROR(DEVICE_STUB, "Get %s Mangling Name fail!", funcInfo.funcName.c_str());
        return AsccStatus::FAILURE;
    }
    for (const auto& name : manglingNameList) {
        if (GenKtypeSection(stubCode, funcInfo, name) == AsccStatus::FAILURE) {
            ASC_LOG_ASC_ERROR(DEVICE_STUB, "%s, GenKtypeSection fail!", funcInfo.funcName.c_str());
            return AsccStatus::FAILURE;
        }
        ++templateKey_;
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccDeviceStub::GenStubFunc(std::ofstream &stubCode, const FuncInfo &funcInfo)
{
    bool dumpTypeIsNotNone;
    bool dumpTypeIsPrintf;
    bool dumpAscendCStamp;
    GetStubFuncDumpInfo(dumpTypeIsNotNone, dumpTypeIsPrintf, dumpAscendCStamp);
    FuncInfo newFuncInfo = GetNewFunctionInfo(funcInfo, dumpTypeIsNotNone, dumpAscendCStamp);
    std::vector<std::string> nameSpaceStack = GetAllNestedNameSpace(newFuncInfo.nameSpace);
    for (const auto &nameSpace : nameSpaceStack) {
        stubCode << "namespace " << nameSpace << " {" << std::endl;
        stubHeaderCode_ << "namespace " << nameSpace << " {" << std::endl;
    }
    std::string originCallTempParams = std::string();
    if (funcInfo.IsTempSpec()) {
        GenTemplateExpSpecStubFuncDefinition(stubCode, newFuncInfo, originCallTempParams);
    } else if (funcInfo.IsTempDecl()) {
        GenTemplateStubFuncDefinition(stubCode, newFuncInfo, originCallTempParams);
    } else {
        GenStubFuncDefinition(stubCode, newFuncInfo);
    }
    stubCode << "{" << std::endl;
    StubFuncDumpAndHardSyncImpl(stubCode, dumpTypeIsNotNone, dumpAscendCStamp, dumpTypeIsPrintf);
    StubFuncWorkSpaceImpl(stubCode, dumpAscendCStamp, funcInfo.hasKfcServer);
    StubFuncCallImpl(stubCode, funcInfo, originCallTempParams);
    stubCode << "}" << std::endl;
    stubCode << std::endl;
    for (const auto &nameSpace : nameSpaceStack) {
        stubCode << "} // " << nameSpace << std::endl;
        stubHeaderCode_ << "} // " << nameSpace << std::endl;
    }
    if (StubFuncInstImpl(stubCode, newFuncInfo) == AsccStatus::FAILURE) {
        ASC_LOG_ASC_ERROR(DEVICE_STUB, "Func %s generate template instantiation failed!", funcInfo.funcName.c_str());
        return AsccStatus::FAILURE;
    }
    if (StubFuncKtypeSectionImpl(stubCode, funcInfo) == AsccStatus::FAILURE) {
        ASC_LOG_ASC_ERROR(DEVICE_STUB, "Func %s generate kernel type section failed!", funcInfo.funcName.c_str());
        return AsccStatus::FAILURE;
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccDeviceStub::GenKtypeSection(
    std::ofstream &stubCode, const FuncInfo &funcInfo, const std::string &mangName)
{
    std::pair<std::string, std::string> typeInfo;
    if (this->GetKernelType(typeInfo) == AsccStatus::FAILURE) {
        ASC_LOG_ASC_ERROR(DEVICE_STUB, "invalid codemode, GetKernelType fail!");
        return AsccStatus::FAILURE;
    }
    const auto &[typeStrucName, kType] = typeInfo;
    KtypesSectionParam params = {funcInfo.funcName, typeStrucName, kType};
    if ((this->codeModeMask_ & MIX_KERNEL_SECTION_MASK) > 0) {
        ASC_LOG_ASC_INFO(DEVICE_STUB, "Section KernelSectionMode: MIX.");
        if (kernelCounter_ == 0) {
            stubCode << this->GetKtypeSectionVariable(
                params, FuncMetaType::F_TYPE_KTYPE, KernelSectionMode::MIX_AIC, mangName);
            return AsccStatus::SUCCESS;
        } else if (kernelCounter_ == 1) {
            stubCode << this->GetKtypeSectionVariable(
                params, FuncMetaType::F_TYPE_KTYPE, KernelSectionMode::MIX_AIV, mangName);
            return AsccStatus::SUCCESS;
        }
    }
    if (this->kernelType_ == CodeMode::KERNEL_TYPE_MIX_AIV_1_0) {
        ASC_LOG_ASC_INFO(DEVICE_STUB, "Section KernelSectionMode: AIV.");
        stubCode << this->GetKtypeSectionVariable(
            params, FuncMetaType::F_TYPE_KTYPE, KernelSectionMode::MIX_AIV, mangName);
        return AsccStatus::SUCCESS;
    }
    if (this->kernelType_ == CodeMode::KERNEL_TYPE_MIX_AIC_1_0) {
        ASC_LOG_ASC_INFO(DEVICE_STUB, "Section KernelSectionMode: AIC.");
        stubCode << this->GetKtypeSectionVariable(
            params, FuncMetaType::F_TYPE_KTYPE, KernelSectionMode::MIX_AIC, mangName);
        return AsccStatus::SUCCESS;
    }
    ASC_LOG_ASC_INFO(DEVICE_STUB, "Section KernelSectionMode: NO_MIX.");
    stubCode << this->GetKtypeSectionVariable(
        params, FuncMetaType::F_TYPE_KTYPE, KernelSectionMode::NO_MIX, mangName);
    return AsccStatus::SUCCESS;
}

std::string AsccDeviceStub::GetKtypeSectionVariable(const KtypesSectionParam &params,
    const FuncMetaType &funcMetaType, const KernelSectionMode &genMode, const std::string &mangName)
{
    std::string sectionVar = std::string();
    std::string newVarName = params.variableName;
    std::string sectionMangName;
    if (genMode == KernelSectionMode::MIX_AIC) {
        newVarName.append("_mix_aic_section");
        sectionMangName = mangName + "_mix_aic";
    }
    if (genMode == KernelSectionMode::MIX_AIV) {
        newVarName.append("_mix_aiv_section");
        sectionMangName = mangName + "_mix_aiv";
    }
    if (genMode == KernelSectionMode::NO_MIX) {
        newVarName.append("_section");
        sectionMangName = mangName;
    }
    newVarName.append("_" + std::to_string(templateKey_));
    sectionVar.append("static const struct " + params.typeStrucName + " " + newVarName + " __attribute__ ");
    sectionVar.append("((used, section (\".ascend.meta." + sectionMangName + "\"))) = ");
    sectionVar.append(
        "{ { {" + FUNC_METATYPE_TO_STR.at(funcMetaType) + ", sizeof(unsigned int)}, " + params.kType + "}");
    if (KERNEL_TYPE_TO_TASK_RATION.find(this->kernelType_) != KERNEL_TYPE_TO_TASK_RATION.end()) {
        sectionVar.append(", " + KERNEL_TYPE_TO_TASK_RATION.at(this->kernelType_));
    }
    sectionVar.append(" };\n");
    return sectionVar;
}

AsccStatus AsccDeviceStub::CompileDeviceStub()
{
    if (coreType_ != ShortSoCVersion::ASCEND910B) {
        return AsccStatus::SUCCESS;
    }
    for (const auto &args : compileArgsList_) {
        if (stubIsGen_.find(args.file) == stubIsGen_.end()) {
            continue;
        }
        ASCC_CHECK(CompileTask<AsccCompileV220>(args) == AsccStatus::SUCCESS,
            { ASC_LOG_ASC_ERROR(DEVICE_STUB, "Device stub file %s compile fail!", args.file.c_str()); });
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccDeviceStub::GetKernelType(std::pair<std::string, std::string> &typeInfo) const
{
    if (KERNEL_TYPE_TO_SECTION_TYPE.find(this->kernelType_) == KERNEL_TYPE_TO_SECTION_TYPE.end()) {
        ASC_LOG_ASC_ERROR(DEVICE_STUB, "Invalid kernelType: [%u]!", static_cast<uint32_t>(this->kernelType_));
        return AsccStatus::FAILURE;
    }
    typeInfo = KERNEL_TYPE_TO_SECTION_TYPE.at(this->kernelType_);
    return AsccStatus::SUCCESS;
}

bool AsccDeviceStub::IsMix() const
{
    return (this->codeModeMask_ & MIX_TYPE_MASK) > 0;
}
}  // namespace Ascc