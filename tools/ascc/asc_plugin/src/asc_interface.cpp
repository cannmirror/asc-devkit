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
 * \file asc_interface.cpp
 * \brief
 */
#include "asc_interface.h"
#include "asc_dev_section_generate.h"
#include "asc_dev_funcRegistry_generate.h"
#include "asc_info_manager.h"
#include "asc_ast_utils.h"
#include "asc_ast_device_analyzer.h"
#include "asc_ast_device_consumer.h"
#include "asc_utils.h"
#include "asc_json_string.h"
#include "asc_log.h"
#include "asc_host_code_generate.h"
#include "asc_compile_options.h"
#include <dlfcn.h>
#include <limits.h>

namespace AscPlugin {

namespace {
// pluginPath Example: /cann version/x86_64-linux/ascc/lib64/libasc_plugin.so
//                     /cann version/compiler/ascc/lib64/libasc_plugin.so
//                     /cann version/tools/ascc/lib64/libasc_plugin.so
//                     /latest/x86_64-linux/ascc/lib64/libasc_plugin.so
// cannPath: directory cann version or directory latest
std::string ExtractCannPath(const std::string& pluginPath)
{
    const std::vector<std::string> potentialPath = {
        "/x86_64-linux/ascc/lib64/libasc_plugin.so",
        "/aarch64-linux/ascc/lib64/libasc_plugin.so",
        "/compiler/ascc/lib64/libasc_plugin.so",
        "/tools/ascc/lib64/libasc_plugin.so",
    };

    for (const std::string& expectedPath : potentialPath) {
        if (EndsWith(pluginPath, expectedPath)) {
            return pluginPath.substr(0, pluginPath.size() - expectedPath.size());
        }
    }
    return std::string();
}

// called in PluginGetPreCompileOpts and PluginPrologue to initialize in the first place
uint32_t InitCannPath()
{
    // dladdr must be based on visible symbols
    static uint32_t cannPathInit = []() -> uint32_t {
        auto& manager = InfoManager::GetInstance();
        Dl_info info;
        const void* funcAddr = reinterpret_cast<const void*>(&PluginGetPreCompileOpts);

        if (dladdr(funcAddr, &info) != 0) {
            char fullPath[PATH_MAX] = {0};
            if (realpath(info.dli_fname, fullPath) != nullptr) {
                std::string cannPath = ExtractCannPath(std::string(fullPath));
                if (cannPath.empty()) {
                    ASC_LOGE("Path %s for libasc_plugin.so does not belong to CANN package.", fullPath);
                    return ASC_CANNPATH_NOT_FOUND;
                }
                manager.SetCannPath(cannPath);
                return ASC_SUCCESS;
            }
        }
        ASC_LOGE("Cannot find CANN package path based on libasc_plugin.so.");
        return ASC_CANNPATH_NOT_FOUND;
    }();
    return cannPathInit;
}

const std::string& GetCceAicoreArch(const CoreType coreType)
{
    auto& manager = InfoManager::GetInstance();
    ShortSocVersion soc = manager.GetShortSocVersion();
    return CCE_AICORE_MAP.at({soc, coreType});
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
    ShortSocVersion soc = manager.GetShortSocVersion();
    if (soc == ShortSocVersion::ASCEND910B) {
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

} // namespace

int32_t PluginGetPreCompileOpts(const char** result)
{
    ASC_CHECK_NULLPTR(result, "PluginGetPreCompileOpts");

    if (InitCannPath() == ASC_CANNPATH_NOT_FOUND) {
        return ASC_CANNPATH_NOT_FOUND;
    }

    auto& manager = InfoManager::GetInstance();
    PathInfo pathInfo = manager.GetPathInfo();
    std::vector<std::string> compileOptions = {
        "-I" + pathInfo.cannIncludePath,
        "-I" + pathInfo.hostApiPath,
        "-I" + pathInfo.highLevelApiPath,
        "-I" + pathInfo.tikcfwPath,
        "-I" + pathInfo.tikcfwLibPath,
        "-I" + pathInfo.tikcfwLibMatmulPath,
        "-I" + pathInfo.tikcfwImplPath,
        "-I" + pathInfo.tikcfwInterfacePath,
        "-std=c++17"
    };

    PreCompileOptsResult res = {compileOptions};
    return DumpResultInfo(res, result);
}

int32_t PluginPrologue(const char** result, const char* config)
{
    ASC_CHECK_NULLPTR(result, "PluginPrologue");
    ASC_CHECK_NULLPTR(config, "PluginPrologue");
    if (InitCannPath() == ASC_CANNPATH_NOT_FOUND) {
        return ASC_CANNPATH_NOT_FOUND;
    }

    PrologueConfig configInfo;
    int32_t fromJsonRes = FromJson(configInfo, config);
    if (fromJsonRes != ASC_SUCCESS) {
        return fromJsonRes;
    }

    std::string timestamp = GenerateTimestamp();
    if (timestamp == "") {
        ASC_LOGE("GenerateTimestamp error.");
        return ASC_FAILURE;
    }

    auto &manager = InfoManager::GetInstance();
    manager.SetSocVersion(configInfo.npuSoc);
    manager.SetShortSocVersion(CCE_AICORE_ARCH_MAP.at(configInfo.npuArch));
    // should set tmpPath first, then set saveTemp
    std::string logFolderPath = GetTempFolder(configInfo.logPath, configInfo.source, timestamp, "log");
    std::string tmpFolderPath = GetTempFolder(configInfo.tmpPath, configInfo.source, timestamp, "temp");
    if (configInfo.saveTemp) {
        if (CreateDirectory(logFolderPath) != ASC_SUCCESS) {
            ASC_LOGE("Failed to create log folder %s.", logFolderPath.c_str());
            return ASC_FAILURE;
        }
        if (CreateDirectory(tmpFolderPath) != ASC_SUCCESS) {
            ASC_LOGE("Failed to create tmp folder %s.", tmpFolderPath.c_str());
            return ASC_FAILURE;
        }
        ASC_LOGI("Log path is set up as %s.", logFolderPath.c_str());
        ASC_LOGI("Temp path is set up as %s.", tmpFolderPath.c_str());
    }

    manager.SetTempPath(tmpFolderPath);
    manager.SetLogPath(logFolderPath);
    manager.SetSaveTempRequested(configInfo.saveTemp);
    manager.SetCompileArgs(configInfo.compileArgs);
    manager.SetSourceFile(configInfo.source);

    // do AST analyze to extract kernel type and printf/assert
    AscPlugin::AscAstDeviceAnalyzer deviceAnalyzer(configInfo.source);
    if (deviceAnalyzer.Process() != ASC_SUCCESS) {
        ASC_LOGE("AscAstAnalyzer run failed. Please check log.");
        return ASC_FAILURE;
    }
    if (manager.IsL2CacheEnabled()) {
        manager.SetAscendMetaFlag(ASC_L2CACHE_HINT_MASK);
    }
    if (manager.IsDumpOn()) {
        manager.SetAscendMetaFlag(ASC_PRINT_MASK);
    }
    manager.UpdateOneCoreDumpSize();
    ASC_LOGD("After AST analysis, hasPrintf_ is %d, hasAssert_ is %d, oneCoreDumpSize_ is %u.", manager.HasPrintf(),
        manager.HasAssert(), manager.GetOneCoreDumpSize());

    PrologueResult res = {ORIGIN_KERNEL_PREFIX, DEVICE_STUB_PREFIX};
    return DumpResultInfo(res, result);
}

int32_t PluginGenKernel(const char** result, const char* info)
{
    ASC_CHECK_NULLPTR(result, "PluginGenKernel");
    ASC_CHECK_NULLPTR(info, "PluginGenKernel");

    KernelInfo kernelInfo;
    static bool firstKernel = true;
    auto &manager = InfoManager::GetInstance();
    int32_t fromJsonRes = FromJson(kernelInfo, info);
    if (fromJsonRes != ASC_SUCCESS) {
        return fromJsonRes;
    }

    if (firstKernel) {
        manager.SetFirstKernel(true);
        firstKernel = false;
    }

    const auto& [kernelType, kfcScene] = GetKernelFuncScene(kernelInfo);
    for (const auto& ktype : kernelType) {
        if (ktype == KernelMetaType::KERNEL_TYPE_AIC_ONLY || ktype == KernelMetaType::KERNEL_TYPE_AIV_ONLY) {
            continue;
        }
        InfoManager::GetInstance().SetAscendMetaFlag(ASC_FFTS_MASK);
    }
    const auto [deviceResult, deviceStub, metaInfo] = GetDeviceCode(kernelInfo, kernelType, kfcScene);
    if (deviceResult != 0) {
        return ASC_FAILURE;
    }
    std::string hostStub  = GetHostStubCode(kernelInfo, kernelType);
    manager.SetFirstKernel(false);

    PluginKernelType pluginKtype = PluginKernelType::MIX; // if multi kernel type means core_ratio(x, y) => mix
    if (kernelType.size() == 1) {
        pluginKtype = META_KTYPE_TO_KTYPE.at(kernelType[0]);
    }
    GenKernelResult res = {hostStub, deviceStub, metaInfo, pluginKtype};
    return DumpResultInfo(res, result);
}

int32_t PluginEpilogue(const char** result)
{
    ASC_CHECK_NULLPTR(result, "PluginEpilogue");

    KernelTypeResult kernelTypeRes = CheckHasMixKernelFunc();
    // MIX_1_1 and MIX_1_2 with either one having KFC at same time is not supported
    if ((kernelTypeRes.hasMixOneToOneWithKfc && kernelTypeRes.hasMixOneToTwo) ||
        (kernelTypeRes.hasMixOneToTwoWithKfc && kernelTypeRes.hasMixOneToOne)) {
        return ASC_FAILURE;
    }

    std::vector<std::string> hostExtraCompileOptions = GetHostCompileOptions();
    std::vector<std::string> deviceCommonOptions = GetDeviceCommonCompileOptions(kernelTypeRes);

    std::vector<std::string> deviceCubeExtraCompileOptions = deviceCommonOptions;
    deviceCubeExtraCompileOptions.emplace_back("--cce-aicore-arch=" + GetCceAicoreArch(CoreType::CUBE));
    UpdateManglingNameSuffix(deviceCubeExtraCompileOptions, CoreType::CUBE);

    std::vector<std::string> deviceVecExtraCompileOptions = deviceCommonOptions;
    deviceVecExtraCompileOptions.emplace_back("--cce-aicore-arch=" + GetCceAicoreArch(CoreType::VEC));
    UpdateManglingNameSuffix(deviceVecExtraCompileOptions, CoreType::VEC);
    if (InfoManager::GetInstance().GetShortSocVersion() == ShortSocVersion::ASCEND310P) {
        deviceVecExtraCompileOptions.emplace_back("-D__ENABLE_VECTOR_CORE__");
    }

    std::string functionRegisterCode = FunctionRegistryImpl();
    EpilogueResult res = {functionRegisterCode, hostExtraCompileOptions, deviceCubeExtraCompileOptions,
        deviceVecExtraCompileOptions};
    return DumpResultInfo(res, result);
}

int32_t PluginFatbinLink(const char** result)
{
    ASC_CHECK_NULLPTR(result, "PluginFatbinLink");
    if (InitCannPath() == ASC_CANNPATH_NOT_FOUND) {
        return ASC_CANNPATH_NOT_FOUND;
    }
    auto& manager = InfoManager::GetInstance();
    std::string cannPath = manager.GetCannPath();
    std::string socVersion = manager.GetSocVersion();
    if (socVersion == "Ascend910B4-1") {
        socVersion = "Ascend910B4";      // B4-1 reads the same lib as Ascend910B4
    }
    std::vector<std::string> linkOptions = {
        // link libraies
        "-lascendc_runtime", "-lascendcl", "-lruntime", "-lerror_manager", "-lprofapi", "-lascendalog", "-lmmpa",
        "-lascend_dump", "-lc_sec", "-lstdc++",
        // link path
        "-L" + cannPath + "/lib64"
    };
    if (!socVersion.empty()) {
        linkOptions.emplace_back("-L" + cannPath + "/tools/simulator/" + socVersion + "/lib");
    }
    auto registerBinaryCode = GetBinaryRegisterCode();

    FatbinLinkResult res = {linkOptions, registerBinaryCode};
    return DumpResultInfo(res, result);
}

} // namespace AscPlugin