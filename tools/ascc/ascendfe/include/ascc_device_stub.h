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
 * \file ascc_device_stub.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_DEVICE_STUB_H__
#define __INCLUDE_ASCC_DEVICE_STUB_H__
#include <string>
#include <utility>
#include <unordered_set>
#include <memory>
#include <fstream>

#include "ascc_types.h"
#include "ascc_info_function.h"
#include "ascc_info_storage.h"

namespace Ascc {
constexpr uint32_t ARGS_SIZE = 6;
class AsccDeviceStub {
public:
    AsccDeviceStub(const ShortSoCVersion &coreType, const std::string &dstFilePath);
    AsccStatus Process();
    AsccStatus GenerateJsonFiles() const;

private:
    enum class KernelSectionMode : uint32_t {
        NO_MIX = 0,
        MIX_AIV = 1,
        MIX_AIC = 2
    };
    struct KtypesSectionParam {
        std::string variableName;
        std::string typeStrucName;
        std::string kType;
    };
    using Funcs = AsccInfoFunction;
    using FuncInfo = AsccInfoFunction::FunctionInfo;
    using ParamInfo = AsccInfoFunction::ParameterInfo;

    void UpdateDumpCompileOption();
    void UpdateWorkFlowCompileOption();
    AsccStatus GenTmpDeviceCode(const std::string &inputFile, const std::string &outputFile,
        const std::unordered_map<std::string, std::unordered_set<std::string>>& kernelCallLinesColumns);
    AsccStatus UpdateNewWorkflowFlag();
    AsccStatus GenDeviceStubCode(const std::string &filePath, const AsccInfoStorage::FileInfos &fileInfo);
    AsccStatus ParamInit(const FuncInfo& funcInfo);
    AsccStatus DeviceStubCodeImpl(const std::string &filePath, const std::shared_ptr<Funcs> &funcsInfo);
    void GenHeadCode( std::ofstream &stubCode, const std::string &filePath) const;
    FuncInfo GetNewFunctionInfo(
        const FuncInfo &funcInfo, const bool dumpTypeIsNotNone, const bool dumpAscendCStamp);
    std::vector<std::string> GetAllNestedNameSpace(const std::string &nameSpacePrefix) const;
    void GenStubFuncDefinition(std::ofstream &stubCode, const FuncInfo &newFuncInfo);
    void GenTemplateStubFuncDefinition(
        std::ofstream &stubCode, const FuncInfo &newFuncInfo, std::string &originCallTempParams);
    void GenTemplateExpSpecStubFuncDefinition(
        std::ofstream &stubCode, const FuncInfo &newFuncInfo, std::string &originCallTempParams);
    std::string GetStubFuncDefParamsAndWorksapceFlag(const FuncInfo& newFuncInfo, bool skipFfts = false);
    void StubFuncDumpAndHardSyncImpl(std::ofstream &stubCode,
        const bool dumpTypeIsNotNone, const bool dumpAscendCStamp, const bool dumpTypeIsPrintf) const;
    void StubFuncWorkSpaceImpl(std::ofstream &stubCode, const bool dumpAscendCStamp, const bool hasKfcServer) const;
    void GetStubFuncDumpInfo(bool &dumpTypeIsNotNone, bool &dumpTypeIsPrintf, bool &dumpAscendCStamp) const;
    void StubFuncCallImpl(
        std::ofstream &stubCode, const FuncInfo &funcInfo, const std::string &originCallTempParams) const;
    AsccStatus GenNormalFuncSymbolImpl(std::ofstream &stubCode, const FuncInfo &newFuncInfo);
    AsccStatus GenTempFuncSymbolImpl(std::ofstream &stubCode, const FuncInfo &newFuncInfo);
    AsccStatus StubFuncInstImpl(std::ofstream &stubCode, const FuncInfo &newFuncInfo);
    AsccStatus GetManglingList(std::vector<std::string> &manglingNameList, const FuncInfo& funcInfo) const;
    AsccStatus StubFuncKtypeSectionImpl(std::ofstream &stubCode, const FuncInfo& funcInfo);
    AsccStatus GenStubFunc(std::ofstream &stubCode, const FuncInfo &funcInfo);
    AsccStatus GenKtypeSection(std::ofstream &stubCode, const FuncInfo &funcInfo, const std::string &mangName);
    std::string GetKtypeSectionVariable(const KtypesSectionParam &params, const FuncMetaType &funcMetaType,
        const KernelSectionMode &genMode, const std::string &mangName);
    AsccStatus CompileDeviceStub();
    AsccStatus GetKernelType(std::pair<std::string, std::string> &typeInfo) const;
    bool IsMix() const;

private:
    bool isNewWorkflow_ = false;
    bool haveWorkspace_ = false;
    uint32_t templateKey_ = 0;
    uint32_t codeModeMask_ = 0;
    uint32_t beginExtraCounter_ = 0;
    uint32_t endExtraCounter_ = 0;
    uint32_t kernelCounter_ = 0; // in mix scene: 0 -> aic, 1 -> aiv
    std::ofstream stubHeaderCode_;
    size_t dumpSize_ = 1048576;
    ShortSoCVersion coreType_ = ShortSoCVersion::INVALID_TYPE;
    std::string dstFilePath_ = std::string();
    std::string deviceStubHeader_ = std::string();
    std::string newWorkflowFile_ = std::string();
    std::vector<std::string> stubFiles_;
    CodeMode kernelType_ = CodeMode::KERNEL_TYPE_MIX_AIC_1_2;
    std::unordered_set<std::string> stubIsGen_;
    std::array<CompileArgs, ARGS_SIZE> compileArgsList_;
};
} // Ascc
#endif // __INCLUDE_ASCC_DEVICE_STUB_H__