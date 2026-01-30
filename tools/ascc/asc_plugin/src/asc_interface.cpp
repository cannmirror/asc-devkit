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
 * \file asc_interface.cpp
 * \brief
 */
#include "asc_interface.h"
#include "asc_dev_section_generate.h"
#include "asc_dev_func_registry_generate.h"
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
#include <fstream>

namespace AscPlugin {

namespace {
// pluginPath Example: /cann version/x86_64-linux/lib64/plugin/asc/libasc_plugin.so
// cannPath: directory cann version or directory latest
std::string ExtractCannPath(const std::string& pluginPath)
{
    const std::vector<std::string> potentialPath = {
        "/x86_64-linux/lib64/plugin/asc/libasc_plugin.so",
        "/aarch64-linux/lib64/plugin/asc/libasc_plugin.so",
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

void GenerateAclrtHeader(const std::string& headerPath)
{
    if (PathCheck(headerPath.c_str(), true) == PathStatus::NOT_EXIST) {
        ASC_LOGE("GenerateAclrtHeader Path [%s]: path not exist.", headerPath.c_str());
        return;
    }

    for (const auto& pair : g_kernelVarMap) {
        std::vector<std::string> kernelVarList = pair.second;
        std::string kernelName = kernelVarList[0];   // {func name, variable1 type, variable1 name..}
        std::string fileName = headerPath + "/aclrtlaunch_" + kernelName + ".h";
        std::string upperKernelName = ToUpper(kernelName);
        ASC_LOGD("Generate ACLRT_LAUNCH_KERNEL header for %s.", fileName.c_str());

        std::string variableStr;
        uint32_t varListSize = kernelVarList.size();
        // {func name, variable1 type, variable1 name..}
        for (uint32_t i = 1; i < varListSize; i +=2) { // var list starts from index 1. 2 means in pair var type + name
            variableStr += kernelVarList[i] + " " + kernelVarList[i+1];
            if (i != varListSize - 2) {   // 2 means in pair {variable type, variable name}
                variableStr += ", ";
            }
        }
        ASC_LOGD("variableStr is %s.", variableStr.c_str());

        std::ofstream outFile;
        outFile.open(fileName, std::ios::app);
        if (!outFile) {
            ASC_LOGE("Failed to create outFile[%s]!", fileName.c_str());
            return;
        }
        outFile << "#ifndef HEADER_ACLRTLAUNCH_" << upperKernelName << "_H\n";
        outFile << "#define HEADER_ACLRTLAUNCH_" << upperKernelName << "_H\n";
        outFile << "#include \"acl/acl_base.h\"\n\n";
        outFile << "#ifndef ACLRT_LAUNCH_KERNEL\n";
        outFile << "#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func\n\n";
        outFile << "#define aclrtlaunch_" << kernelName <<"(blockdim, ...) " << kernelName <<
            "(blockdim, nullptr, __VA_ARGS__)\n";
        outFile << "#endif\n\n";
        outFile << "void " << kernelName << "(uint32_t __ascc_blockDim__, void* __ascc_hold__, void* __ascc_stream__, "
            << variableStr << ");\n";
        outFile << "#endif\n";
        outFile.close();
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
    std::vector<std::string> compileOptions = {"-std=c++17"};
    for (auto& incPath: pathInfo.cannIncludePath) {
        compileOptions.emplace_back("-I" + incPath);
    }

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

    if (manager.GetShortSocVersion() != ShortSocVersion::ASCEND910B) {
        // do AST analyze to extract kernel type and printf/assert
        AscPlugin::AscAstDeviceAnalyzer deviceAnalyzer(configInfo.source);
        if (deviceAnalyzer.Process() != ASC_SUCCESS) {
            ASC_LOGE("AscAstAnalyzer run failed. Please check log.");
            return ASC_FAILURE;
        }
    }

    if (!manager.GetAclrtHeaderPath().empty()) {
        GenerateAclrtHeader(manager.GetAclrtHeaderPath());
    }
    PrologueResult res = {ORIGIN_KERNEL_PREFIX, DEVICE_STUB_PREFIX};
    return DumpResultInfo(res, result);
}

int32_t PluginGenKernel(const char** result, const char* info)
{
    ASC_CHECK_NULLPTR(result, "PluginGenKernel");
    ASC_CHECK_NULLPTR(info, "PluginGenKernel");

    KernelInfo kernelInfo;
    int32_t fromJsonRes = FromJson(kernelInfo, info);
    if (fromJsonRes != ASC_SUCCESS) {
        return fromJsonRes;
    }
    auto &manager = InfoManager::GetInstance();
    static auto flag = manager.SetKernelFuncFlag();
    (void)flag;

    std::string deviceStub;
    std::string metaInfo;
    std::unordered_set<KernelMetaType> kernelType;
    if (manager.GetShortSocVersion() != ShortSocVersion::ASCEND910B) { // deviceStub generate by bisheng in 71
        auto &&[kType, kfcScene] = GetKernelFuncScene(kernelInfo);
        auto &&[deviceResult, devStub, meta] = GetDeviceCode(kernelInfo, kType, kfcScene);
        if (deviceResult != 0) {
            return ASC_FAILURE;
        }
        deviceStub = std::move(devStub);
        metaInfo = std::move(meta);
        kernelType = std::move(kType);
    } else {
        kernelType.insert(KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2);
    }

    std::string hostStub  = GetHostStubCode(kernelInfo, kernelType);

    PluginKernelType pluginKtype = PluginKernelType::MIX; // if multi kernel type means core_ratio(x, y) => mix
    if (kernelType.size() == 1) {
        KernelMetaType curKernelType = ExtractKernelType(kernelType);
        pluginKtype = META_KTYPE_TO_KTYPE.at(curKernelType);
    }
    GenKernelResult res = {hostStub, deviceStub, metaInfo, pluginKtype};
    return DumpResultInfo(res, result);
}

int32_t PluginEpilogue(const char** result)
{
    ASC_CHECK_NULLPTR(result, "PluginEpilogue");

    CompileOptionManager mng = CompileOptionManager();
    auto deviceCubeExtraCompileOptions = mng.GetDeviceCompileOptions(CoreType::CUBE);
    auto deviceVecExtraCompileOptions = mng.GetDeviceCompileOptions(CoreType::VEC);
    auto hostExtraCompileOptions = mng.GetHostCompileOptions();
    auto functionRegisterCode = FunctionRegistryImpl();
    if (deviceCubeExtraCompileOptions.empty() && deviceVecExtraCompileOptions.empty()) {
        return ASC_FAILURE;
    }
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
        "-lascendc_runtime", "-lascendcl", "-lruntime", "-lerror_manager", "-lprofapi", "-lunified_dlog", "-lmmpa",
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