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
 * \file asc_compile_options.cpp
 * \brief
 */

#include "asc_compile_options.h"
#include "asc_info_manager.h"
#include "asc_log.h"

namespace AscPlugin {

// pair: <whether has mix including MIX_AIV/AIC_1_0, whether has MIX_1_1 and MIX_1_2 at same time>
KernelTypeResult CheckHasMixKernelFunc()
{
    KernelTypeResult res;
    KernelFuncInfo mixOneToOneInfo;
    KernelFuncInfo mixOneToTwoInfo;
    KernelFuncInfo mixOneToOneWithKfcInfo;
    KernelFuncInfo mixOneToTwoWithKfcInfo;
    for (const auto& funcInfo : InfoManager::GetInstance().GetGlobalSymbolInfo()) {
        KernelMetaType kType = std::get<0>(funcInfo.second);
        std::string fileName = std::get<1>(funcInfo.second);
        uint32_t lineNum = std::get<2>(funcInfo.second);
        uint32_t colNum = std::get<3>(funcInfo.second);
        KfcScene kfcScene = std::get<4>(funcInfo.second);
        if (kType == KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1) {
            res.hasMixOneToOne = true;
            mixOneToOneInfo = {funcInfo.first, fileName, lineNum, colNum};
            if (kfcScene == KfcScene::Open) {
                res.hasMixOneToOneWithKfc = true;
                mixOneToOneWithKfcInfo = {funcInfo.first, fileName, lineNum, colNum};
            }
        } else if (kType == KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2) {
            res.hasMixOneToTwo = true;
            mixOneToTwoInfo = {funcInfo.first, fileName, lineNum, colNum};
            if (kfcScene == KfcScene::Open) {
                res.hasMixOneToTwoWithKfc = true;
                mixOneToTwoWithKfcInfo = {funcInfo.first, fileName, lineNum, colNum};
            }
        }
        if (InfoManager::GetInstance().GetShortSocVersion() != ShortSocVersion::ASCEND910B) {
            if (res.hasMixOneToOneWithKfc && res.hasMixOneToTwo) {
                ASC_LOGE("Having kernel function %s with KERNEL_TYPE_MIX_AIC_1_1 in file %s line %u col %u with "
                    "REGIST_MATMUL_OBJ and kernel function %s with KERNEL_TYPE_MIX_AIC_1_2 in file %s line %u col %u "
                    "is not supported.", mixOneToOneWithKfcInfo.mangledName.c_str(), mixOneToOneWithKfcInfo.fileName.c_str(),
                    mixOneToOneWithKfcInfo.lineNum, mixOneToOneWithKfcInfo.colNum, mixOneToTwoInfo.mangledName.c_str(),
                    mixOneToTwoInfo.fileName.c_str(), mixOneToTwoInfo.lineNum, mixOneToTwoInfo.colNum);
                return res;
            }
            if (res.hasMixOneToTwoWithKfc && res.hasMixOneToOne) {
                ASC_LOGE("Having kernel function %s with KERNEL_TYPE_MIX_AIC_1_1 in file %s line %u col %u and kernel "
                    "function %s with KERNEL_TYPE_MIX_AIC_1_2 in file %s line %u col %u with REGIST_MATMUL_OBJ "
                    "is not supported.", mixOneToOneInfo.mangledName.c_str(), mixOneToOneInfo.fileName.c_str(),
                    mixOneToOneInfo.lineNum, mixOneToOneInfo.colNum, mixOneToTwoWithKfcInfo.mangledName.c_str(),
                    mixOneToTwoWithKfcInfo.fileName.c_str(), mixOneToTwoWithKfcInfo.lineNum, mixOneToTwoWithKfcInfo.colNum);
                return res;
            }
        }
    }
    ASC_LOGD("KernelTypeResult result: hasMixOneToOne %d, hasMixOneToTwo %d, hasMixOneToOneWithKfc %d, "
        "hasMixOneToTwoWithKfc %d.", res.hasMixOneToOne, res.hasMixOneToTwo, res.hasMixOneToOneWithKfc,
        res.hasMixOneToTwoWithKfc);
    return res;
}

inline bool IsMixKernelType(const KernelMetaType kType)
{
    return (kType == KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0 || kType == KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0 ||
        kType == KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1 || kType == KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2);
}

// Assume mangling name is A. If AIC_ONLY / AIV_ONLY => do not need update
// If MIX_AIC_1_0, MIX_AIV_1_0, MIX_AIC_1_1, MIX_AIC_1_2,
// then update -D<manglingName>=<manglingName>_mix_aic, -D<manglingName>=<manglingName>_mix_aiv
void UpdateManglingNameSuffix(std::vector<std::string>& compileOptions, const CoreType coreType)
{
    auto& manager = InfoManager::GetInstance();
    ShortSocVersion shortSoc = manager.GetShortSocVersion();
    if (shortSoc == ShortSocVersion::ASCEND910B || shortSoc == ShortSocVersion::ASCEND910_95) {
        for (const auto& funcInfo : InfoManager::GetInstance().GetGlobalSymbolInfo()) {
            std::string manglingName = funcInfo.first;
            KernelMetaType kType = std::get<0>(funcInfo.second);
            bool isMixKernelType = IsMixKernelType(kType);
            if (coreType == CoreType::CUBE && isMixKernelType) {
                compileOptions.emplace_back(
                    "-D" + manglingName + "=" + manglingName.substr(DEVICE_STUB_PREFIX_LEN) + "_mix_aic");
            } else if (coreType == CoreType::VEC && isMixKernelType) {
                compileOptions.emplace_back(
                    "-D" + manglingName + "=" + manglingName.substr(DEVICE_STUB_PREFIX_LEN) + "_mix_aiv");
            } else {
                compileOptions.emplace_back("-D" + manglingName + "=" + manglingName.substr(DEVICE_STUB_PREFIX_LEN));
            }
        }
    } else {
        for (const auto& funcInfo : InfoManager::GetInstance().GetGlobalSymbolInfo()) {
            std::string manglingName = funcInfo.first;
            compileOptions.emplace_back("-D" + manglingName + "=" + manglingName.substr(DEVICE_STUB_PREFIX_LEN));
        }
    }
}

void CompileOptionManager::SetOldPrintOptions(std::vector<std::string>& devSocOpts) const
{
    if (userDumpStatus_ && isDumpOn_) {
        devSocOpts.emplace_back("-DONE_CORE_DUMP_SIZE=" + std::to_string(oneCoreDumpSize_));
    }
    if (!cannVersionHeader_.empty()) {
        devSocOpts.emplace_back("-include");
        devSocOpts.emplace_back(cannVersionHeader_);
    }
}

template <>
std::vector<std::string> CompileOptionManager::GetDeviceCompileOptionsWithSoc<ShortSocVersion::ASCEND910B>(
    CoreType coreType) const
{
    (void)coreType;
    std::vector<std::string> devSocOpts = {"-mllvm", "-cce-aicore-stack-size=0x8000",
                                           "-mllvm", "-cce-aicore-function-stack-size=0x8000",
                                           "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
                                           "-D__ENABLE_ASCENDC_PRINTF__"};
    if (l2CacheOn_) {
        devSocOpts.emplace_back("-DL2_CACHE_HINT");
    }
    return devSocOpts;
}

template <>
std::vector<std::string> CompileOptionManager::GetDeviceCompileOptionsWithSoc<ShortSocVersion::ASCEND910_95>(
    CoreType coreType) const
{
    KernelTypeResult kernelTypeRes = CheckHasMixKernelFunc();
    if ((kernelTypeRes.hasMixOneToOneWithKfc && kernelTypeRes.hasMixOneToTwo) ||
        (kernelTypeRes.hasMixOneToTwoWithKfc && kernelTypeRes.hasMixOneToOne)) {
        return {};
    }
    std::vector<std::string> devSocOpts = {"-mllvm", "-cce-aicore-stack-size=0x8000",
                                           "-mllvm", "-cce-aicore-function-stack-size=0x8000",
                                           "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false"};
    if (l2CacheOn_) {
        devSocOpts.emplace_back("-DL2_CACHE_HINT");
    }
    // needs to update -D__MIX_CORE_AIC_RATION__, -D__MIX_CORE_MACRO__ based on kernel type info
    if (kernelTypeRes.hasMixOneToOne || kernelTypeRes.hasMixOneToTwo) {
        // used for KFC, thus only when type is 1_1 / 1_2
        devSocOpts.emplace_back("-D__MIX_CORE_MACRO__=1");
    }
    if (kernelTypeRes.hasMixOneToOne) {
        devSocOpts.emplace_back("-D__MIX_CORE_AIC_RATION__=1");
    }
    SetOldPrintOptions(devSocOpts);
    UpdateManglingNameSuffix(devSocOpts, coreType);
    return devSocOpts;
}

template <>
std::vector<std::string> CompileOptionManager::GetDeviceCompileOptionsWithSoc<ShortSocVersion::ASCEND310P>(
    CoreType coreType) const
{
    std::vector<std::string> devSocOpts = {// bisheng will add --cce-mask-opt for 310P in default
                                           "-mllvm", "-cce-aicore-fp-ceiling=2",
                                           "-mllvm", "-cce-aicore-record-overflow=false",
                                           "-mllvm", "-cce-aicore-mask-opt=false",
                                           "-D__ENABLE_ASCENDC_PRINTF__"};

    if (coreType == CoreType::VEC) {
        devSocOpts.emplace_back("-D__ENABLE_VECTOR_CORE__");
    }
    UpdateManglingNameSuffix(devSocOpts, coreType);
    return devSocOpts;
}

template<ShortSocVersion soc>
void CompileOptionManager::RegisterOptHandler()
{
    dispatchTable_[soc] = [this](CoreType type) {
        return this->GetDeviceCompileOptionsWithSoc<soc>(type);
    };
}

void CompileOptionManager::InitDispatchTable()
{
    RegisterOptHandler<ShortSocVersion::ASCEND910B>();
    RegisterOptHandler<ShortSocVersion::ASCEND910_95>();
    RegisterOptHandler<ShortSocVersion::ASCEND310P>();
}

CompileOptionManager::CompileOptionManager() :
    socVersion_(InfoManager::GetInstance().GetShortSocVersion()),
    isAutoSyncOn_(InfoManager::GetInstance().IsAutoSyncOn()),
    userDumpStatus_(InfoManager::GetInstance().UserDumpRequested()),
    isDumpOn_(InfoManager::GetInstance().IsDumpOn()),
    l2CacheOn_(InfoManager::GetInstance().IsL2CacheEnabled()),
    oneCoreDumpSize_(InfoManager::GetInstance().GetOneCoreDumpSize()),
    optiLevel_(InfoManager::GetInstance().GetOptimizeLevel()),
    cannVersionHeader_(InfoManager::GetInstance().GetPathInfo().cannVersionHeader)
{
    InitDispatchTable();
}

std::vector<std::string> CompileOptionManager::GetDeviceCompileOptions(CoreType type) const
{
    std::vector<std::string> devSocOpts;
    auto it = dispatchTable_.find(socVersion_);
    if (it != dispatchTable_.end()) {
        devSocOpts = it->second(type); // GetDeviceCompileOptionsWithSoc<socVersion_>
    } else {
        return {};
    }
    if (devSocOpts.empty()) {
        return {};
    }

    std::vector<std::string> opts = {"-std=c++17", optiLevel_, "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0"};
    opts.emplace_back("--cce-aicore-arch=" + CCE_AICORE_MAP.at({socVersion_, type}));
    if (isAutoSyncOn_){
        opts.emplace_back("--cce-auto-sync");
    }
    if (!userDumpStatus_) {  // user passed -DASCENDC_DUMP=0 in compile args
        opts.emplace_back("-DASCENDC_DUMP=0");
    } else if (isDumpOn_) {
        opts.emplace_back("-DASCENDC_DUMP=1");
    }
    opts.insert(opts.end(), devSocOpts.begin(), devSocOpts.end());
    return opts;
}

std::vector<std::string> CompileOptionManager::GetHostCompileOptions() const
{
    return {"-std=c++17", optiLevel_, "-D__NPU_HOST__", "-DTILING_KEY_VAR=0"};
}

} // namespace AscPlugin