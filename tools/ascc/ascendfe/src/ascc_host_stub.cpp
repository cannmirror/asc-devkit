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
 * \file ascc_host_stub.cpp
 * \brief
 */

#include "ascc_host_stub.h"

#include <cstring>
#include <sstream>
#include <vector>
#include <stack>
#include <unordered_map>
#include <algorithm>

#include "ascc_log.h"
#include "ascc_types.h"
#include "ascc_utils.h"
#include "ascc_info_aicore_function.h"
#include "ascc_compile_host.h"
#include "ascc_argument_manager.h"
#include "ascc_mangle.h"
#include "ascc_dump_flags.h"
#include "ascc_match_global_info.h"

namespace Ascc {
static const std::unordered_map<Ascc::CodeMode, Ascc::KernelMode> KERNEL_MODE_MAP = {
    {Ascc::CodeMode::KERNEL_TYPE_AIV_ONLY, Ascc::KernelMode::AIV},
    {Ascc::CodeMode::KERNEL_TYPE_AIC_ONLY, Ascc::KernelMode::AIC},
    {Ascc::CodeMode::KERNEL_TYPE_MIX_AIV_1_0, Ascc::KernelMode::MIX},
    {Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_0, Ascc::KernelMode::MIX},
    {Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_1, Ascc::KernelMode::MIX},
    {Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_2, Ascc::KernelMode::MIX},
};

inline bool KernelFuncIsCall(const AsccInfoFunction::FunctionInfo& info)
{
    if ((!info.isTemplate || info.IsTempSpec()) &&
        !AsccMatchGlobalInfo::GetInstance().IsCalled(info.manglingName)) {
        return false;
    }
    if (info.IsTempDecl() && info.mangledToInstFuncInfo.size() == 0) {
        return false;
    }
    return true;
}


static const std::unordered_map<CodeMode, const char*> KTYPE_TO_LAUNCH_PARAMS = {
    {CodeMode::KERNEL_TYPE_AIV_ONLY, "g_kernel_handle_aiv, 5"},
    {CodeMode::KERNEL_TYPE_AIC_ONLY, "g_kernel_handle_aic, 6"},
    {CodeMode::KERNEL_TYPE_MIX_AIV_1_0, "g_kernel_handle_mix, 7"},
    {CodeMode::KERNEL_TYPE_MIX_AIC_1_0, "g_kernel_handle_mix, 8"},
    {CodeMode::KERNEL_TYPE_MIX_AIC_1_1, "g_kernel_handle_mix, 9"},
    {CodeMode::KERNEL_TYPE_MIX_AIC_1_2, "g_kernel_handle_mix, 10"}
};
bool AsccHostStub::IsKernelFuncFound() const
{
    return AsccMatchGlobalInfo::GetInstance().HasKernelCall();
}

void AsccHostStub::UpdateKernelTypeStatus()
{
    Ascc::PreTaskType preTaskType = Ascc::AsccArgumentManager::GetInstance().GetPreTaskType();
    auto &storage = AsccInfoStorage::GetInstance().GetAllInfos();
    for (const auto &storageInfo : storage) {
        auto functions = std::dynamic_pointer_cast<AsccInfoFunction>(AsccInfoStorage::GetInstance().GetInfo(
            std::string(storageInfo.first), AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION));
        if (functions == nullptr) {
            continue;
        }
        for (auto it = functions->Begin(); it != functions->End(); ++it) {
            Ascc::AsccInfoFunction::FunctionInfo &info = it->second;
            Ascc::KernelMode curKernelMode = Ascc::KERNEL_MODE_MAP.at(info.kernelType);
            if (preTaskType != PreTaskType::NONE && !KernelFuncIsCall(info)) {
                continue;
            }
            if (curKernelMode == KernelMode::MIX) {
                hasMix_ = true;
            }
            if (curKernelMode == KernelMode::AIC) {
                hasAic_ = true;
            }
            if (curKernelMode == KernelMode::AIV) {
                hasAiv_ = true;
            }
        }
    }
    typeNums_ = (hasMix_ ? 1u : 0u) + (hasAic_ ? 1u : 0u) + (hasAiv_ ? 1u : 0u);
}

void AsccHostStub::UpdateDumpStatus()
{
    auto &dumpInfo = Ascc::AsccDumpFlags::GetInstance();
    hasPrintf_ = dumpInfo.IsDumpOn() && dumpInfo.GetPrintfFlag();
    hasAssert_ = dumpInfo.IsDumpOn() && dumpInfo.GetAssertFlag();
}

AsccHostStub::AsccHostStub(PreTaskType preTaskType)
{
    hostStubFilePath_ = (preTaskType == Ascc::PreTaskType::HOST) ?
        Ascc::AsccArgumentManager::GetInstance().GetModulePath() + "/host_stub.h" :
        Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath + "/host_stub.h";
}

void AsccHostStub::GenKernelHandleCheck(const std::string& suffix)
{
    std::ostringstream codeSource;
    codeSource << "    if (g_kernel_handle_" << suffix << " == nullptr) {\n"
               << "        printf(\"[ERROR] %s\\n\", ascendcErrMsg);\n"
               << "        return 0;\n"
               << "    }\n";
    hostStubSource_ << codeSource.str();
}

void AsccHostStub::GenKernelHandleRegisterCode(const std::string &suffix)
{
    std::ostringstream codeSource;
    codeSource << "    ret = AscendDevBinaryRegister(__replaced_ascend_kernel." << suffix << "_buf,\n"
               << "                                  __replaced_ascend_kernel." << suffix << "_file_len,\n"
               << "                                  &g_kernel_handle_" << suffix << ");\n"
               << "    if (ret != 0) {\n"
               << "        printf(\"AscendDevBinaryRegister " << suffix << " ret %d \\n\", ret);\n"
               << "    }\n";
    hostStubSource_ << codeSource.str();
}

void AsccHostStub::GenBinaryRegisterCode()
{
    std::ostringstream codeSource;
    codeSource << "static void __register_kernels(void) __attribute__((constructor));\n"
               << "void __register_kernels(void)\n"
               << "{\n"
               << "    const char* compileSocVersion = \"__replaced_ascend_compile_soc_version\";\n"
               << "    int32_t ret;\n"
               << "    bool checkSocVersion = AscendCheckSoCVersion(compileSocVersion, ascendcErrMsg);\n"
               << "    if (!checkSocVersion) {\n"
               << "        return;\n"
               << "    }\n";
    hostStubSource_ << codeSource.str();
    codeSource.str("");
    codeSource.clear();
    if (hasMix_) {
        GenKernelHandleRegisterCode("mix");
    }
    if (hasAiv_) {
        GenKernelHandleRegisterCode("aiv");
    }
    if (hasAic_) {
        GenKernelHandleRegisterCode("aic");
    }
    codeSource << "    AscendProfRegister();\n"
               << "}\n\n";
    hostStubSource_ << codeSource.str();
}

void AsccHostStub::GenKernelHandleUnregisterCode(const std::string& handleSuffix)
{
    std::ostringstream codeSource;
    codeSource << "        if (g_kernel_handle_" << handleSuffix << ") {\n"
               << "            UnregisterAscendBinary(g_kernel_handle_" << handleSuffix << ");\n"
               << "            g_kernel_handle_" << handleSuffix << " = nullptr;\n"
               << "        }\n";
    hostStubSource_ << codeSource.str();
}

void AsccHostStub::GenHostStubUsedIntfDecl()
{
    std::ostringstream codeSource;
    codeSource << "extern \"C\" {\n"
               << "int32_t AscendDevBinaryRegister(const void *fileBuf, size_t fileSize, void **handle);\n"
               << "int32_t AscendFunctionRegister(void *handle, const char *stubFunc);\n"
               << "int32_t AscendKernelLaunchWithFlagV2(const char *stubFunc, const uint32_t blockDim, void **args,\n"
               << "    uint32_t size, const void *stream);\n"
               << "uint32_t GetAscendCoreSyncAddr(void **addr);\n"
               << "int UnregisterAscendBinary(void *hdl);\n"
               << "void StartAscendProf(const char *name, uint64_t *startTime);\n"
               << "void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);\n"
               << "bool GetAscendProfStatus();\n"
               << "uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);\n"
               << "uint32_t FreeAscendMemDevice(void *devMem);\n"
               << "bool AscendCheckSoCVersion(const char *socVersion, char* errMsg);\n"
               << "void AscendProfRegister();\n"
               << "uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);\n"
               << "uint32_t LaunchAscendKernelForVectorCore(const char *opType, void *handle, const uint64_t key, void **args,\n"
               << "    uint32_t size, const void *stream, bool enbaleProf, uint32_t aicBlockDim, uint32_t aivBlockDim,\n"
               << "    uint32_t aivBlockDimOffset);\n";
    if (hasAssert_) {
        codeSource << "int32_t rtSetExceptionExtInfo(const rtArgsSizeInfoAscc_t * const sizeInfo);\n"
                   << "namespace Adx {\n"
                   << "    void *AdumpGetSizeInfoAddr(uint32_t space, uint32_t &atomicIndex);\n"
                   << "}\n";
    }
    codeSource << "}\n\n"
               << "namespace Adx {\n"
               << "    void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,\n"
               << "        void *stream, const char *opType);\n"
               << "}\n\n";
    hostStubSource_ << codeSource.str();
}

std::string AsccHostStub::GetHostStubFilePath() const
{
    return hostStubFilePath_;
}

AsccStatus AsccHostStub::GenHostStubFile()
{
    bool isKernelFuncFound = IsKernelFuncFound();
    if (!isKernelFuncFound) {
        ASC_LOG_ASC_WARN(HOST_STUB, "No kernel function found!");
        return AsccStatus::SUCCESS;
    }
    if (!Ascc::IsPathLegal(hostStubFilePath_) || !Ascc::IsParentDirValid(hostStubFilePath_)) {
        ASC_LOG_ASC_ERROR(HOST_STUB, "host_stub file path [%s] does not exist!", hostStubFilePath_.c_str());
        return AsccStatus::FAILURE;
    }
    hostStubSource_.open(hostStubFilePath_, std::ios::out);
    if (!hostStubSource_) {
        ASC_LOG_ASC_ERROR(HOST_STUB, "Failed to create host stub file!");
        return AsccStatus::FAILURE;
    }
    UpdateKernelTypeStatus();
    UpdateDumpStatus();

    GenHostStubHeadCode();
    auto& env = AsccGlobalEnvManager::GetInstance();
    if (Ascc::AsccArgumentManager::GetInstance().GetPreTaskType() == PreTaskType::HOST) {
        env.kernelCallStubFile = Ascc::AsccArgumentManager::GetInstance().GetModulePath() + "/kernel_call_stub.cpp";
    } else {
        env.kernelCallStubFile = env.asccTmpHostGenPath + "/kernel_call_stub.cpp";
    }
    kernelCallStubFile_.open(env.kernelCallStubFile, std::ios::app);
    GenLaunchProfilingCode();
    auto &storage = AsccInfoStorage::GetInstance().GetAllInfos();
    for (const auto &storageInfo : storage) {
        auto functions = std::dynamic_pointer_cast<AsccInfoFunction>(AsccInfoStorage::GetInstance().GetInfo(
            std::string(storageInfo.first), AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION));
        if (functions == nullptr) {
            continue;
        }
        GenLaunchProfilingBody(functions);
    }
    hostStubSource_ << "#endif // HOST_STUB_H\n";
    hostStubSource_.close();
    kernelCallStubFile_.close();
    return AsccStatus::SUCCESS;
}

void AsccHostStub::GenSetExceptionDumpCode()
{
    std::ostringstream codeSource;
    codeSource << "static void ascendc_set_exception_dump_info(uint32_t dumpSize)\n"
               << "{\n"
               << "    uint32_t atomicIndex = 0U;\n"
               << "    uint32_t addrNum = 1U;\n\n"
               << "    void *exceptionDumpAddr = Adx::AdumpGetSizeInfoAddr(addrNum + ascendcExceptionDumpHead, atomicIndex);\n"
               << "    if (exceptionDumpAddr == nullptr) {\n"
               << "        printf(\"Get exceptionDumpAddr is nullptr.\\n\");\n"
               << "        return;\n"
               << "    }\n\n"
               << "    uint64_t *sizeInfoAddr = reinterpret_cast<uint64_t *>(exceptionDumpAddr);\n"
               << "    *sizeInfoAddr = static_cast<uint64_t>(atomicIndex);\n"
               << "    sizeInfoAddr++;\n\n"
               << "    *sizeInfoAddr = static_cast<uint64_t>(1);\n"
               << "    sizeInfoAddr++;\n\n"
               << "    *sizeInfoAddr = dumpSize * 75;\n"
               << "    constexpr uint64_t workspaceOffset = (4ULL << 56ULL);\n"
               << "    *sizeInfoAddr |= workspaceOffset;\n\n"
               << "    const rtArgsSizeInfoAscc sizeInfo = {exceptionDumpAddr, atomicIndex};\n"
               << "    int32_t ret = rtSetExceptionExtInfo(&sizeInfo);\n"
               << "    if (ret != 0) {\n"
               << "        printf(\"rtSetExceptionExtInfo failed, ret = %d.\\n\", ret);\n"
               << "    }\n"
               << "}\n\n";
    hostStubSource_ << codeSource.str();
}

void AsccHostStub::GenerateCodeForMode(KernelMode mode, std::vector<std::string> &ascendKernelStruct)
{
    static const char* prefixList[] = {"mix", "aiv", "aic"};
    std::string prefix = prefixList[static_cast<uint32_t>(mode)];
    std::ostringstream codeSource;
    codeSource << "    uint32_t " << prefix << "_type;\n"
               << "    uint32_t " << prefix << "_len;\n"
               << "    uint32_t " << prefix << "_file_len;\n"
               << "    uint8_t " << prefix << "_buf[__replaced_" << prefix << "_len];\n";
    hostStubSource_ << codeSource.str();
    ascendKernelStruct.emplace_back(std::to_string(static_cast<uint32_t>(mode)));
    ascendKernelStruct.emplace_back("__replaced_" + prefix + "_len");
    ascendKernelStruct.emplace_back("__replaced_" + prefix + "_file_len");
    ascendKernelStruct.emplace_back("{0}");
}

void AsccHostStub::GenUnregisterCode()
{
    std::ostringstream codeSource;
    codeSource << "class KernelHandleGradUnregister {\n"
               << "private:\n"
               << "    KernelHandleGradUnregister() {}\n"
               << "    ~KernelHandleGradUnregister() {\n";
    hostStubSource_ << codeSource.str();
    codeSource.str("");
    codeSource.clear();
    if (hasMix_) {
        GenKernelHandleUnregisterCode("mix");
    }
    if (hasAiv_) {
        GenKernelHandleUnregisterCode("aiv");
    }
    if (hasAic_) {
        GenKernelHandleUnregisterCode("aic");
    }
    codeSource << "    }\n"
               << "    KernelHandleGradUnregister(const KernelHandleGradUnregister&) = delete;\n"
               << "    KernelHandleGradUnregister& operator=(const KernelHandleGradUnregister&) = delete;\n"
               << "public:\n"
               << "    static KernelHandleGradUnregister& GetInstance() {\n"
               << "        static KernelHandleGradUnregister instance;\n"
               << "        return instance;\n"
               << "    }\n"
               << "};\n\n";
    hostStubSource_ << codeSource.str();
}

void AsccHostStub::GenManglingRegisterCode(const std::string &manglingName, const KernelMode &curKernelMode)
{
    std::ostringstream codeSource;
    codeSource << "        stubFunc_ = \"" << manglingName << "\";\n"
               << "        g_kernel_handle_register_ = g_kernel_handle_";
    if (curKernelMode == Ascc::KernelMode::MIX) {
        codeSource << "mix";
    } else if (curKernelMode == Ascc::KernelMode::AIC) {
        codeSource << "aic";
    } else if (curKernelMode == Ascc::KernelMode::AIV) {
        codeSource << "aiv";
    }
    codeSource << ";\n"
               << "        retRegister_ = AscendFunctionRegister(g_kernel_handle_register_, stubFunc_);\n"
               << "        if (retRegister_ != 0) {\n"
               << "            printf(\"AscendFunctionRegister ret %d \\n\", retRegister_);\n"
               << "        }\n";
    hostStubSource_ << codeSource.str();
}

void AsccHostStub::GenManglingRegisterBody(const std::shared_ptr<AsccInfoFunction> &funcsInfo)
{
    std::unordered_map<std::string, std::string> &ascFixedMangleMap =
        Ascc::AsccMangle::GetInstance().GetOriginToFixedMangledNames();
    for (auto it = funcsInfo->Begin(); it != funcsInfo->End(); ++it) {
        auto &funcInfo = it->second;
        Ascc::CodeMode curKernelType = funcInfo.kernelType;
        Ascc::KernelMode curKernelMode = Ascc::KERNEL_MODE_MAP.at(curKernelType);
        if (!funcInfo.isTemplate || funcInfo.IsTempSpec()) {
            if (ascFixedMangleMap.find(funcInfo.manglingName) == ascFixedMangleMap.end()) {
                continue;
            }
            GenManglingRegisterCode(ascFixedMangleMap[funcInfo.manglingName], curKernelMode);
            continue;
        }
        for (auto &pair : funcInfo.mangledToInstFuncInfo) {
            if (ascFixedMangleMap.find(pair.first) == ascFixedMangleMap.end()) {
                continue;
            }
            GenManglingRegisterCode(ascFixedMangleMap[pair.first], curKernelMode);
        }
    }
}

void AsccHostStub::GenFunctionRegisterCode()
{
    std::ostringstream codeSource;
    codeSource << "namespace {\n"
               << "class KernelHandleGradRegister {\n"
               << "private:\n"
               << "    KernelHandleGradRegister() {}\n"
               << "    ~KernelHandleGradRegister() {}\n"
               << "    KernelHandleGradRegister(const KernelHandleGradRegister&) = delete;\n"
               << "    KernelHandleGradRegister& operator=(const KernelHandleGradRegister&) = delete;\n"
               << "public:\n"
               << "    static KernelHandleGradRegister& GetInstance() {\n"
               << "        static KernelHandleGradRegister instance;\n"
               << "        return instance;\n"
               << "    }\n"
               << "    void AsccFunctionRegister() {\n";
    hostStubSource_ << codeSource.str();
    codeSource.str("");
    codeSource.clear();
    AsccInfoStorage &storage = AsccInfoStorage::GetInstance();
    const auto &allInfos = storage.GetAllInfos();
    for (const auto &allInfo : allInfos) {
        auto fileInfo = allInfo.second;
        std::shared_ptr<AsccInfoFunction> functions = Ascc::GetFileInfo<AsccInfoFunction>(fileInfo);
        if (functions != nullptr) {
            GenManglingRegisterBody(functions);
        }
    }
    codeSource << "    }\n"
               << "private:\n"
               << "    int32_t retRegister_;\n"
               << "    const char* stubFunc_;\n"
               << "    void *g_kernel_handle_register_;\n"
               << "};\n\n"

               << "static struct AutoRegistrar {\n"
               << "    AutoRegistrar() {\n"
               << "        KernelHandleGradRegister::GetInstance().AsccFunctionRegister();\n"
               << "    }\n"
               << "} autoRegistrar;\n"
               << "}\n\n";
    hostStubSource_ << codeSource.str();
}

void AsccHostStub::GenAscendKernelStruct()
{
    std::ostringstream codeSource;
    codeSource << "static struct ascend_kernels {\n"
               << "    uint32_t version;\n"
               << "    uint32_t type_cnt;\n";
    std::vector<std::string> ascendKernelStruct;
    ascendKernelStruct.emplace_back(std::to_string(1));
    ascendKernelStruct.emplace_back(std::to_string(typeNums_));
    hostStubSource_ << codeSource.str();
    codeSource.str("");
    codeSource.clear();

    if (hasMix_) {
        GenerateCodeForMode(KernelMode::MIX, ascendKernelStruct);
    }
    if (hasAiv_) {
        GenerateCodeForMode(KernelMode::AIV, ascendKernelStruct);
    }
    if (hasAic_) {
        GenerateCodeForMode(KernelMode::AIC, ascendKernelStruct);
    }

    codeSource << "} __replaced_ascend_kernel __attribute__ ((section (\"__replaced_ascend_section\"))) = {";
    for (size_t i = 0; i < ascendKernelStruct.size(); ++i) {
        if (i > 0) {
            codeSource << ",";
        }
        codeSource << ascendKernelStruct[i];
    }
    codeSource << "};\n\n";
    hostStubSource_ << codeSource.str();
}

void AsccHostStub::GenHostStubHeadCode()
{
    bool isPreTask = AsccArgumentManager::GetInstance().GetPreTaskType() != PreTaskType::NONE;
    std::string incFile = isPreTask ? "" : "#include <cstdio>\n";
    std::ostringstream codeSource;
    codeSource << "#ifndef HOST_STUB_H\n"
               << "#define HOST_STUB_H\n\n"
               << incFile << "\n";
    hostStubSource_ << codeSource.str();
    codeSource.str("");
    codeSource.clear();

    if (hasAssert_) {
        codeSource << "constexpr unsigned int ascendcExceptionDumpHead = 2U;\n"
                   << "typedef struct rtArgsSizeInfoAscc {\n"
                   << "    void *infoAddr;\n"
                   << "    uint32_t atomicIndex;\n"
                   << "} rtArgsSizeInfoAscc_t;\n";
    }
    codeSource << "static char ascendcErrMsg[4096] = {0};\n\n";
    if (hasMix_) {
        codeSource << "static void *g_kernel_handle_mix = nullptr;\n";
    }
    if (hasAiv_) {
        codeSource << "static void *g_kernel_handle_aiv = nullptr;\n";
    }
    if (hasAic_) {
        codeSource << "static void *g_kernel_handle_aic = nullptr;\n";
    }
    codeSource << "\n";
    hostStubSource_ << codeSource.str();
    codeSource.str("");
    codeSource.clear();

    GenAscendKernelStruct();
    GenHostStubUsedIntfDecl();
    GenUnregisterCode();
    GenBinaryRegisterCode();
    GenFunctionRegisterCode();

    hostStubSource_ << codeSource.str();
    if (hasAssert_) {
        GenSetExceptionDumpCode();
    }
}

void AsccHostStub::GenLaunchProfilingCode()
{
    std::ostringstream codeSource;
    codeSource << "inline uint32_t launch_and_profiling"
               << "(const char* stubFunc, uint32_t blockDim, void* stream, void **args, uint32_t size, void *handle, "
                  "uint32_t ktype)\n"
               << "{\n"
               << "    uint64_t startTime;\n"
               << "    const char *name = stubFunc;\n"
               << "    bool profStatus = GetAscendProfStatus();\n"
               << "    if (profStatus) {\n"
               << "        StartAscendProf(name, &startTime);\n"
               << "    }\n"
               << "    if (handle == nullptr) {\n"
               << "        printf(\"[ERROR] %s\\n\", ascendcErrMsg);\n"
               << "        return 0;\n"
               << "    }\n"
               << "    int32_t retLaunch = AscendKernelLaunchWithFlagV2(stubFunc, blockDim, args, size, stream);\n"
               << "    if (retLaunch != 0) {\n"
               << "        printf(\"AscendKernelLaunchWithFlagV2 ret %u\\n\", retLaunch);\n"
               << "    }\n"
               << "    if (profStatus) {\n"
               << "        ReportAscendProf(name, blockDim, ktype, startTime);\n"
               << "    }\n"
               << "    return retLaunch;\n"
               << "}\n";
    hostStubSource_ << codeSource.str();
}

static std::string MapParamTypeToVoid(std::string paramType)
{
    return (paramType == "uint8_t *" || paramType == "unsigned char *") ? "void*" : paramType;
}

void AsccHostStub::GenStubFuncImpl(const AsccInfoFunction::FunctionInfo& info)
{
    Ascc::PreTaskType preTaskType = Ascc::AsccArgumentManager::GetInstance().GetPreTaskType();
    auto &dumpInfo = AsccDumpFlags::GetInstance();
    uint32_t dumpSize = dumpInfo.GetDumpSize();
    CodeMode curKernelType = info.kernelType;
    KernelMode curKernelMode = KERNEL_MODE_MAP.at(info.kernelType);
    bool hasFfts = curKernelMode == Ascc::KernelMode::MIX || curKernelType == CodeMode::KERNEL_TYPE_MIX_AIV_1_0 ||
                 curKernelType == CodeMode::KERNEL_TYPE_MIX_AIC_1_0;
    std::ostringstream funcImplCode;
    if (Ascc::AsccArgumentManager::GetInstance().GetPreTaskType() == PreTaskType::HOST) {
        std::string hostStubFilePath = Ascc::AsccArgumentManager::GetInstance().GetModulePath() + "/host_stub.h";
        funcImplCode << "#include \"" << hostStubFilePath << "\"\n";
    }
    funcImplCode << GenStubFuncDecl(info, /* hasNameSpace = */true) << "\n{\n";
    if (preTaskType != PreTaskType::NONE && !KernelFuncIsCall(info)) {
        funcImplCode << "    return 0;\n}\n";
        kernelCallStubFile_ << funcImplCode.str();
        return;
    }
    funcImplCode << "    (void)__ascc_hold__;\n";
    funcImplCode << "    struct {\n";
    if (dumpInfo.IsDumpOn()) {
        funcImplCode << "        void* __ascendc_dump;\n";
    }
    if (hasFfts) {
        funcImplCode << "        alignas(((alignof(void*) + 3) >> 2) << 2) void* ffts_addr;\n";
    }
    for (auto &param : info.params) {
        funcImplCode << "        alignas(((alignof(" << MapParamTypeToVoid(param.type) << ") + 3) >> 2) << 2) "
                   << MapParamTypeToVoid(param.type) << " " << param.paraName << ";\n";
    }
    funcImplCode << "        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;\n";
    funcImplCode << "    } __ascendc_args;\n";

    // args_declare_code
    funcImplCode << "    uint32_t __ascendc_ret;\n";
    if (dumpInfo.IsDumpOn()) {
        funcImplCode <<
            "    constexpr uint32_t __ascendc_one_core_dump_size = " << std::to_string(dumpSize) << ";\n";
        funcImplCode <<
            "    AllocAscendMemDevice(&(__ascendc_args.__ascendc_dump), __ascendc_one_core_dump_size * 75);\n";
    }
    funcImplCode << "    constexpr uint32_t __ascendc_overflow_status_size = 8;\n";
    funcImplCode <<
        "    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);\n";
    if (hasFfts) {
        funcImplCode << "    void *ffts_addr;\n";
        funcImplCode << "    __ascendc_ret = GetAscendCoreSyncAddr(&ffts_addr);\n";
        funcImplCode << "    if (__ascendc_ret != 0) {\n";
        funcImplCode << "        ::printf(\"GetAscendCoreSyncAddr ret %u\\n\", __ascendc_ret);\n";
        funcImplCode << "        return __ascendc_ret;\n";
        funcImplCode << "    }\n";
        funcImplCode << "    __ascendc_args.ffts_addr = ffts_addr;\n";
    }
    for (auto &param : info.params) {
        funcImplCode << "    __ascendc_args." << param.paraName << " = " << param.paraName << ";\n";
    }
    funcImplCode << "    const char* __ascendc_name = \"" << info.funcName << "\";\n";
    funcImplCode << "    const char* __ascc_manglingName__ = nullptr;\n";
    funcImplCode << ManglingNameJudgeCode(info); // choose mangling
    funcImplCode << "    if (__ascc_manglingName__ == nullptr) {\n";
    funcImplCode << "        ::printf(\"Kernel Function %s call failed!\\n\", __ascendc_name);\n";
    funcImplCode << "        return 1U;\n";
    funcImplCode << "    }\n";
    if (hasAssert_) {
        funcImplCode << "    ascendc_set_exception_dump_info(__ascendc_one_core_dump_size);\n";
    }
    funcImplCode << "    __ascendc_ret = launch_and_profiling(__ascc_manglingName__, __ascc_blockDim__, "
                    "__ascc_stream__, (void **)&__ascendc_args, sizeof(__ascendc_args), ";
    funcImplCode << KTYPE_TO_LAUNCH_PARAMS.at(info.kernelType);
    funcImplCode << ");\n";
    funcImplCode << "    KernelHandleGradUnregister::GetInstance();\n";
    if (hasPrintf_) {
        funcImplCode << "    Adx::AdumpPrintWorkSpace(__ascendc_args.__ascendc_dump, "
                               "__ascendc_one_core_dump_size * 75, __ascc_stream__, __ascendc_name);\n";
    }
    if (dumpInfo.IsDumpOn()) {
        funcImplCode << "    FreeAscendMemDevice(__ascendc_args.__ascendc_dump);\n";
    }
    funcImplCode << "    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);\n";
    funcImplCode << "    return __ascendc_ret;\n";
    funcImplCode << "}\n";
    kernelCallStubFile_ << funcImplCode.str();
}

std::string AsccHostStub::ManglingNameJudgeCode(const AsccInfoFunction::FunctionInfo& info)
{
    std::unordered_map<std::string, std::string> &ascFixedMangleMap =
        Ascc::AsccMangle::GetInstance().GetOriginToFixedMangledNames();
    std::string judgeCode = std::string();
    if (!info.IsTempDecl() && ascFixedMangleMap.find(info.manglingName) != ascFixedMangleMap.end()) {
        judgeCode = "    __ascc_manglingName__ = \"" + ascFixedMangleMap[info.manglingName] + "\";\n";
        return judgeCode;
    }

    for (const auto& [manglingName, instFuncInfo] : info.mangledToInstFuncInfo) {
        if (ascFixedMangleMap.find(manglingName) == ascFixedMangleMap.end()) {
            ASC_LOG_ASC_WARN(HOST_STUB, "Kernel func [%s] has no call.", instFuncInfo->manglingName.c_str());
            continue;
        }
        judgeCode += "    if constexpr (";
        for (size_t i = 0; i < instFuncInfo->templateParams.size(); ++i) {
            if (i > 0) {
                judgeCode += " && ";
            }
            const auto& declParamInfo = info.templateParams[i];
            const auto& instParamInfo = instFuncInfo->templateParams[i];
            if (instParamInfo.paramType == ParamType::TEMPLATE_TYPE) {
                judgeCode +=
                    "AscendC::Std::is_same<" + declParamInfo.paraName + ", " + instParamInfo.type + ">::value";
            }
            if (instParamInfo.paramType == ParamType::TEMPLATE_INT ||
                instParamInfo.paramType == ParamType::TEMPLATE_ENUM) {
                judgeCode += declParamInfo.paraName + " == " + instParamInfo.type;
            }
            if (instParamInfo.paramType == ParamType::TEMPLATE_DECL) {
                judgeCode += "&" + declParamInfo.paraName + " == &" + instParamInfo.type;
            }
            if (instParamInfo.paramType == ParamType::TEMPLATE_TEMPLATE) {
                std::string typeFuncName = "__AsccIsMyType" + std::to_string(isTypeCounter_);
                ++isTypeCounter_;
                judgeCode += typeFuncName + "<" + declParamInfo.paraName + ">::value";
                kernelCallStubFile_ << "template <" + declParamInfo.type + " " + declParamInfo.paraName + ">\n";
                kernelCallStubFile_ << "struct " + typeFuncName + " : AscendC::Std::false_type {};\n";
                kernelCallStubFile_ << "template <>\n";
                kernelCallStubFile_ << "struct " + typeFuncName + "<" + instParamInfo.type +
                                           "> : AscendC::Std::true_type {};\n";
            }
        }
        judgeCode += ") {\n";
        judgeCode += "        __ascc_manglingName__ = \"" + ascFixedMangleMap[manglingName] + "\";\n";
        judgeCode += "    }\n";
    }
    return judgeCode;
}

std::string AsccHostStub::GenStubFuncDecl(const AsccInfoFunction::FunctionInfo& info, bool hasNameSpace) const
{
    std::string functionEntryReplace;
    std::string paramsList = "(uint32_t __ascc_blockDim__, void* __ascc_hold__, void* __ascc_stream__";
    for (auto &param : info.params) {
        paramsList += ", " + param.type + " " + param.paraName;
    }
    paramsList += ")";
    std::string funcName = hasNameSpace ? info.nameSpace + info.funcName : info.funcName;
    std::string tempParamDecl;
    std::string tempParamCall;
    if (info.isTemplate) {
        tempParamDecl = "template<";
        tempParamCall = "<";
        for (size_t i = 0; i < info.templateParams.size(); i++) {
            if (i > 0) {
                tempParamDecl += ", ";
                tempParamCall += ", ";
            }
            tempParamDecl += info.templateParams[i].type + " " + info.templateParams[i].paraName;
            tempParamCall += info.templateParams[i].type;
        }
        tempParamDecl += ">";
        tempParamCall += ">";
    }
    if (!info.isTemplate) {
        functionEntryReplace += "uint32_t " + funcName + paramsList;
    } else if (info.IsTempDecl()) {
        functionEntryReplace += tempParamDecl + " uint32_t " + funcName + paramsList;
    } else if (info.IsTempSpec()) {
        functionEntryReplace += "template<> uint32_t " + funcName + tempParamCall + paramsList;
    }
    return functionEntryReplace;
}

void AsccHostStub::GenLaunchProfilingBody(const std::shared_ptr<AsccInfoFunction> &functions)
{
    if (functions == nullptr) {
        ASC_LOG_ASC_WARN(HOST_STUB,
            "The content of function is null, the process of parsing the function parameters is terminated!");
        return;
    }
    AsccInfoAicoreFunc &aicoreFuncInfo = AsccInfoAicoreFunc::GetInstance();
    for (auto it = functions->Begin(); it != functions->End(); ++it) {
        AsccInfoFunction::FunctionInfo &info = it->second;
        GenStubFuncImpl(info);
        aicoreFuncInfo.StorekernelDefLineCode(info.definitionPos, info.startLineNo - 1, GenStubFuncDecl(info) + ";");
    }
}

static AsccStatus FindAndReplace(std::string &content, const std::string& target, const std::string& replacement)
{
    size_t pos = 0;
    while ((pos = content.find(target, pos)) != std::string::npos) {
        content.replace(pos, target.length(), replacement);
        pos += replacement.length();
    }
    return AsccStatus::SUCCESS;
}

static int64_t GetDeviceFileSize(const std::string& basePath, const std::string& suffix)
{
    const std::string filename = "/device_" + suffix + ".o";
    std::string fullPath = basePath + "/link_files/merge_obj_final";
    std::string fullPathCheck = CheckAndGetFullPath(fullPath);
    if (fullPathCheck.empty()) {
        return -1LL;
    }
    fullPath = fullPathCheck + filename;
    if (!Ascc::IsPathLegal(fullPath) || !Ascc::IsParentDirValid(fullPath)) {
        ASC_LOG_ASC_ERROR(HOST_STUB, "device ofile path [%s] does not exist!", fullPath.c_str());
        return -1LL;
    }
    std::ifstream deviceFile(fullPath, std::ios::binary | std::ios::ate);
    if (!deviceFile) {
        ASC_LOG_ASC_WARN(HOST_STUB, "Failed to open file: %s", fullPath.c_str());
        return -1LL;  // returns - 1 for error
    }
    return static_cast<int64_t>(deviceFile.tellg());
}

AsccStatus AsccHostStub::UpdateHostStubByDevice()
{
    // get file size from device_xxx.o
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    int64_t mixFileSize  = GetDeviceFileSize(envVar.asccTmpPath, "mix");
    int64_t aicFileSize  = GetDeviceFileSize(envVar.asccTmpPath, "aic");
    int64_t aivFileSize  = GetDeviceFileSize(envVar.asccTmpPath, "aiv");
    if (mixFileSize < 0 || aicFileSize < 0 || aivFileSize < 0) {
        ASC_LOG_ASC_WARN(HOST_STUB, "Some device object files are missing or empty");
    }
    if (!Ascc::IsPathLegal(hostStubFilePath_) || !Ascc::IsParentDirValid(hostStubFilePath_)) {
        ASC_LOG_ASC_ERROR(HOST_STUB, "hostStubFilePath_ [%s] does not exist!", hostStubFilePath_.c_str());
        return AsccStatus::FAILURE;
    }
    std::ifstream fileHost(hostStubFilePath_);
    ASCC_CHECK((fileHost.is_open()), {ASC_LOG_ASC_ERROR(HOST_STUB, "Can not open host_stub.h. Please check it!");});

    std::string content((std::istreambuf_iterator<char>(fileHost)), std::istreambuf_iterator<char>());
    fileHost.close();

    std::string socVersion = Ascc::ToLower(Ascc::AsccArgumentManager::GetInstance().GetNpuArchStr());
    FindAndReplace(content, "__replaced_ascend_compile_soc_version", socVersion);
    FindAndReplace(content, "__replaced_ascend_kernel", "__ascend_kernel_" + socVersion + "_kernels");
    FindAndReplace(content, "__replaced_ascend_section", ".ascend.kernel." + socVersion + ".kernels");
    if (AsccArgumentManager::GetInstance().GetPreTaskType() == Ascc::PreTaskType::NONE) {
        // mix
        FindAndReplace(content, "__replaced_mix_len", std::to_string(mixFileSize));
        FindAndReplace(content, "__replaced_mix_file_len", std::to_string(mixFileSize));
        // aic
        FindAndReplace(content, "__replaced_aic_len", std::to_string(aicFileSize));
        FindAndReplace(content, "__replaced_aic_file_len", std::to_string(aicFileSize));
        // aiv
        FindAndReplace(content, "__replaced_aiv_len", std::to_string(aivFileSize));
        FindAndReplace(content, "__replaced_aiv_file_len", std::to_string(aivFileSize));
    }

    std::ofstream outFile(hostStubFilePath_);
    if (!outFile.is_open()) {
        outFile.close();
        ASC_LOG_ASC_ERROR(HOST_STUB, "Can not write replace content to host_stub.h");
        return AsccStatus::FAILURE;
    }
    outFile << content;
    outFile.close();

    return AsccStatus::SUCCESS;
}
}  // namespace Ascc