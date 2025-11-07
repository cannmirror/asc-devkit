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

    ASC_LOGD("KernelTypeResult result: hasMixOneToOne %d, hasMixOneToTwo %d, hasMixOneToOneWithKfc %d, "
        "hasMixOneToTwoWithKfc %d.", res.hasMixOneToOne, res.hasMixOneToTwo, res.hasMixOneToOneWithKfc,
        res.hasMixOneToTwoWithKfc);
    return res;
}

std::vector<std::string> GetHostCompileOptions()
{
    return {"-std=c++17", InfoManager::GetInstance().GetOptimizeLevel(), "-D__NPU_HOST__", "-DTILING_KEY_VAR=0"};
}

std::vector<std::string> GetDeviceCommonCompileOptions(const KernelTypeResult& kernelTypeRes)
{
    auto& manager = InfoManager::GetInstance();
    std::string cannPath = manager.GetCannPath();
    std::string optLevel = manager.GetOptimizeLevel();
    ShortSocVersion socVersion = manager.GetShortSocVersion();
    std::vector<std::string> deviceCommonOptions = {"-std=c++17", optLevel, "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0"};

    if (socVersion == ShortSocVersion::ASCEND910B) {
        // MIX_1_1 and MIX_1_2 with either one having KFC at same time is not supported
        if ((kernelTypeRes.hasMixOneToOneWithKfc && kernelTypeRes.hasMixOneToTwo) ||
            (kernelTypeRes.hasMixOneToTwoWithKfc && kernelTypeRes.hasMixOneToOne)) {
            return deviceCommonOptions;
        }

        deviceCommonOptions.insert(deviceCommonOptions.end(), {"-mllvm", "-cce-aicore-stack-size=0x8000",
            "-mllvm", "-cce-aicore-function-stack-size=0x8000", "-mllvm", "-cce-aicore-record-overflow=true",
            "-mllvm", "-cce-aicore-addr-transform", "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false"});

        if(manager.IsL2CacheEnabled()) {
            deviceCommonOptions.emplace_back("-DL2_CACHE_HINT");
        }
        // needs to update -D__MIX_CORE_AIC_RATION__, -D__MIX_CORE_MACRO__ based on kernel type info
        if (kernelTypeRes.hasMixOneToOne || kernelTypeRes.hasMixOneToTwo) {
            // used for KFC, thus only when type is 1_1 / 1_2
            deviceCommonOptions.emplace_back("-D__MIX_CORE_MACRO__=1");
        }
        if (kernelTypeRes.hasMixOneToOne) {
            deviceCommonOptions.emplace_back("-D__MIX_CORE_AIC_RATION__=1");
        }
    } else if (socVersion == ShortSocVersion::ASCEND310P) {
        deviceCommonOptions.insert(deviceCommonOptions.end(), {
            // bisheng will add --cce-mask-opt for 310P in default
            "-mllvm", "-cce-aicore-fp-ceiling=2",
            "-mllvm", "-cce-aicore-record-overflow=false", "-mllvm", "-cce-aicore-mask-opt=false"});
    }

    if (manager.IsAutoSyncOn()){
        deviceCommonOptions.emplace_back("--cce-auto-sync");
    }
    if (!manager.UserDumpRequested()) {  // user passed -DASCENDC_DUMP=0 in compile args
        deviceCommonOptions.emplace_back("-DASCENDC_DUMP=0");
    } else {
        if (manager.IsDumpOn()) {
            deviceCommonOptions.emplace_back("-DASCENDC_DUMP=1");
            deviceCommonOptions.emplace_back("-DONE_CORE_DUMP_SIZE=" + std::to_string(manager.GetOneCoreDumpSize()));
        }
    }
    deviceCommonOptions.emplace_back("-include");
    deviceCommonOptions.emplace_back(cannPath + "/include/version/cann_version.h");
    deviceCommonOptions.emplace_back(
        std::string("-D__ASC_FEATURE_META_INFO") + std::to_string(manager.GetMetaFlagCounter()));
    return deviceCommonOptions;
}

} // namespace AscPlugin