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
 * \file asc_json_string.cpp
 * \brief
 */

#include "asc_json_string.h"
#include "asc_utils.h"
#include "asc_log.h"
#include "asc_info_manager.h"
#include <fstream>
#include <string>

#include <securec.h>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

#define ASC_CHECK_JSON_ARG_EXIST(jsonObj, argName, structName)                                          \
    do {                                                                                                \
        if (!(jsonObj).contains(argName)) {                                                             \
            ASC_LOGE("Cannot find argument [%s] in struct %s from json string.", argName, structName);  \
            return AscPlugin::ASC_JSONSTR_ARG_MISSING;                                                  \
        }                                                                                               \
    } while (0)

namespace AscPlugin {

void to_json(nlohmann::json& jsonObj, const PreCompileOptsResult& result) {
    jsonObj = nlohmann::json{
        {"CompileOptions", result.compileOptions}
    };
}

void to_json(nlohmann::json& jsonObj, const PrologueResult& result) {
    jsonObj = nlohmann::json{
        {"OriginPrefix", result.originPrefix},
        {"DeviceStubPrefix", result.deviceStubPrefix}
    };
}

void to_json(nlohmann::json& jsonObj, const GenKernelResult& result) {
    jsonObj = nlohmann::json{
        {"HostStub", result.hostStub},
        {"DeviceStub", result.deviceStub},
        {"MetaInfo", result.metaInfo},
        {"Type", result.type}
    };
}

void to_json(nlohmann::json& jsonObj, const EpilogueResult& result) {
    jsonObj = nlohmann::json{
        {"FunctionRegisterCode", result.functionRegisterCode},
        {"HostExtraCompileOptions", result.hostExtraCompileOptions},
        {"DeviceCubeExtraCompileOptions", result.deviceCubeExtraCompileOptions},
        {"DeviceVecExtraCompileOptions", result.deviceVecExtraCompileOptions}
    };
}

void to_json(nlohmann::json& jsonObj, const FatbinLinkResult& result) {
    jsonObj = nlohmann::json{
        {"ExtraFatbinHostLinkOptions", result.extraFatbinHostLinkOptions},
        {"BinaryRegisterCode", result.binaryRegisterCode}
    };
}

void from_json(const nlohmann::json& jsonObj, PrologueConfig& config)
{
    jsonObj.at("SaveTemp").get_to(config.saveTemp);
    jsonObj.at("Verbose").get_to(config.verbose);
    jsonObj.at("GenMode").get_to(config.genMode);
    jsonObj.at("NpuSoc").get_to(config.npuSoc);
    jsonObj.at("NpuArch").get_to(config.npuArch);
    jsonObj.at("LogPath").get_to(config.logPath);
    jsonObj.at("TmpPath").get_to(config.tmpPath);
    jsonObj.at("Source").get_to(config.source);
    jsonObj.at("BinaryPtrName").get_to(config.binaryPtrName);
    jsonObj.at("BinaryLenName").get_to(config.binaryLenName);
    jsonObj.at("CompileArgs").get_to(config.compileArgs);
}

void from_json(const nlohmann::json& jsonObj, Param& param)
{
    jsonObj.at("Type").get_to(param.type);
    jsonObj.at("Name").get_to(param.name);
    jsonObj.at("HasDefaultValue").get_to(param.hasDefaultValue);
    jsonObj.at("DefaultValue").get_to(param.defaultValue);
    jsonObj.at("Attribute").get_to(param.attribute);
    jsonObj.at("TypeClass").get_to(param.typeClass);
}

void from_json(const nlohmann::json& jsonObj, CoreRatio& ratio)
{
    jsonObj.at("IsCoreRatio").get_to(ratio.isCoreRatio);
    jsonObj.at("CubeNum").get_to(ratio.cubeNum);
    jsonObj.at("VecNum").get_to(ratio.vecNum);
}

void from_json(const nlohmann::json& jsonObj, TemplateInstance& instance)
{
    jsonObj.at("TemplateInstantiationArguments").get_to(instance.templateInstantiationArguments);
    jsonObj.at("InstanceKernelParameters").get_to(instance.instanceKernelParameters);
    jsonObj.at("InstanceMangledName").get_to(instance.instanceMangledName);
    jsonObj.at("InstanceMangledNameConsiderPrefix").get_to(instance.instanceMangledNameConsiderPrefix);
    jsonObj.at("Ratio").get_to(instance.ratio);
}

void from_json(const nlohmann::json& jsonObj, KernelInfo& info)
{
    jsonObj.at("KernelName").get_to(info.kernelName);
    jsonObj.at("KernelMangledName").get_to(info.kernelMangledName);
    jsonObj.at("KernelMangledNameConsiderPrefix").get_to(info.kernelMangledNameConsiderPrefix);
    jsonObj.at("FileName").get_to(info.fileName);
    jsonObj.at("LineNum").get_to(info.lineNum);
    jsonObj.at("ColNum").get_to(info.colNum);
    jsonObj.at("Namespaces").get_to(info.namespaces);
    jsonObj.at("KernelParameters").get_to(info.kernelParameters);
    jsonObj.at("KernelAttributes").get_to(info.kernelAttributes);
    jsonObj.at("Ratio").get_to(info.ratio);
    jsonObj.at("IsTemplate").get_to(info.isTemplate);
    jsonObj.at("TemplateParameters").get_to(info.templateParameters);
    jsonObj.at("TemplateInstances").get_to(info.templateInstances);
}

int32_t FromJson(PrologueConfig& config, const char* jsonStr)
{
    nlohmann::json jsonObj = nlohmann::json::parse(jsonStr);
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "SaveTemp", "PrologueConfig");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "Verbose", "PrologueConfig");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "GenMode", "PrologueConfig");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "NpuSoc", "PrologueConfig");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "NpuArch", "PrologueConfig");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "LogPath", "PrologueConfig");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "TmpPath", "PrologueConfig");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "Source", "PrologueConfig");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "BinaryPtrName", "PrologueConfig");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "BinaryLenName", "PrologueConfig");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "CompileArgs", "PrologueConfig");
    config = jsonObj.get<PrologueConfig>();
    if (!config.npuSoc.empty() && SOC_VERSION_MAP.find(config.npuSoc) == SOC_VERSION_MAP.end()) {
        ASC_LOGE("SocVersion: [%s]: is not supported.", config.npuSoc.c_str());
        return ASC_SOC_NOT_SUPPORT;
    }
    if (config.genMode != GenMode::AICORE_ONLY && config.genMode != GenMode::HOST_AICORE) {
        ASC_LOGE("GenMode: [%u] is not supported.", static_cast<uint32_t>(config.genMode));
        return ASC_FAILURE;
    }
    if (CCE_AICORE_ARCH_MAP.find(config.npuArch) == CCE_AICORE_ARCH_MAP.end()) {
        ASC_LOGE("NpuArch: [%s]: is not supported.", config.npuArch.c_str());
        return ASC_FAILURE;
    }
    if (!config.npuSoc.empty() && CCE_AICORE_ARCH_MAP.at(config.npuArch) != SOC_VERSION_MAP.at(config.npuSoc)) {
        ASC_LOGE("NpuArch: [%s] and NpuSoc: [%s] does not match.", config.npuArch.c_str(), config.npuSoc.c_str());
        return ASC_FAILURE;
    }

    if (config.compileArgs.empty()) {
        ASC_LOGE("CompileArgs should not be empty.");
        return ASC_FAILURE;
    }

    if (config.saveTemp) {
        if(!config.logPath.empty() && CheckAndGetFullPath(config.logPath).empty()) {
            ASC_LOGE("Log Folder path: [%s] does not exist!", config.logPath.c_str());
            return ASC_FAILURE;
        }
        if(!config.tmpPath.empty() && CheckAndGetFullPath(config.tmpPath).empty()) {
            ASC_LOGE("Temp Folder path: [%s] does not exist!", config.tmpPath.c_str());
            return ASC_FAILURE;
        }
    }

    ASC_CHECK_EMPTY_STR(config.source, "source", "PrologueConfig");
    ASC_CHECK_EMPTY_STR(config.binaryPtrName, "binaryPtrName", "PrologueConfig");
    ASC_CHECK_EMPTY_STR(config.binaryLenName, "binaryLenName", "PrologueConfig");

    ASC_LOGD("Extract PrologueConfig from following jsonStr successfully: %s.", jsonStr);
    return ASC_SUCCESS;
}

int32_t FromJson(KernelInfo& info, const char* jsonStr)
{
    nlohmann::json jsonObj = nlohmann::json::parse(jsonStr);
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "KernelName", "KernelInfo");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "KernelMangledName", "KernelInfo");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "KernelMangledNameConsiderPrefix", "KernelInfo");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "FileName", "KernelInfo");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "LineNum", "KernelInfo");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "ColNum", "KernelInfo");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "Namespaces", "KernelInfo");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "KernelParameters", "KernelInfo");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "KernelAttributes", "KernelInfo");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "Ratio", "KernelInfo");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "IsTemplate", "KernelInfo");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "TemplateParameters", "KernelInfo");
    ASC_CHECK_JSON_ARG_EXIST(jsonObj, "TemplateInstances", "KernelInfo");
    info = jsonObj.get<KernelInfo>();

    ASC_CHECK_EMPTY_STR(info.kernelName, "KernelName", "KernelInfo");
    ASC_CHECK_EMPTY_STR(info.kernelMangledName, "KernelMangledName", "KernelInfo");

    ASC_LOGD("Extract KernelInfo from following jsonStr successfully: %s.", jsonStr);
    return ASC_SUCCESS;
}

inline void WriteVector(std::ofstream& os, const std::vector<std::string>& vec)
{
    for (size_t i = 0; i < vec.size(); ++i) {
        os << "    [" << i << "] " << vec[i];
        if (i < vec.size() - 1) os << "\n";
    }
    os << "\n\n";
}

void WriteFields(std::ofstream& outFile, const PrologueResult& inputStruct)
{
    outFile << "PrologueResult.originPrefix\n" << inputStruct.originPrefix << "\n\n"
            << "PrologueResult.deviceStubPrefix\n" << inputStruct.deviceStubPrefix << "\n\n";
}

void WriteFields(std::ofstream& outFile, const GenKernelResult& inputStruct)
{
    outFile << "GenKernelResult.hostStub\n" << inputStruct.hostStub << "\n\n"
            << "GenKernelResult.deviceStub\n" << inputStruct.deviceStub << "\n\n"
            << "GenKernelResult.metaInfo\n" << inputStruct.metaInfo << "\n\n";
    outFile << "GenKernelResult.kernelType\n";
    switch (inputStruct.type) {
        case PluginKernelType::MIX:
            outFile << "MIX";
            break;
        case PluginKernelType::AIC:
            outFile << "AIC";
            break;
        case PluginKernelType::AIV:
            outFile << "AIV";
            break;
        default:
            outFile << "UNKNOWN";
    }
    outFile << "\n\n";
}

void WriteFields(std::ofstream& outFile, const PreCompileOptsResult& inputStruct)
{
    outFile << "PreCompileOptsResult.compileOptions\n";
    WriteVector(outFile, inputStruct.compileOptions);
}

void WriteFields(std::ofstream& outFile, const EpilogueResult& inputStruct)
{
    outFile << "EpilogueResult.functionRegisterCode\n" << inputStruct.functionRegisterCode << "\n\n";
    outFile << "EpilogueResult.hostExtraCompileOptions\n";
    WriteVector(outFile, inputStruct.hostExtraCompileOptions);
    outFile << "EpilogueResult.deviceCubeExtraCompileOptions\n";
    WriteVector(outFile, inputStruct.deviceCubeExtraCompileOptions);
    outFile << "EpilogueResult.deviceVecExtraCompileOptions\n";
    WriteVector(outFile, inputStruct.deviceVecExtraCompileOptions);
}

void WriteFields(std::ofstream& outFile, const FatbinLinkResult& inputStruct)
{
    outFile << "FatbinLinkResult.extraFatbinHostLinkOptions\n";
    WriteVector(outFile, inputStruct.extraFatbinHostLinkOptions);
    outFile << "FatbinLinkResult.binaryRegisterCode\n" << inputStruct.binaryRegisterCode << "\n\n";
}

template <typename T>
int32_t DumpResultInfo(T& inputStruct, const char** result)
{
    static const std::unordered_map<std::string, std::string> ascStructType2LogFileMap = {
        {std::string(typeid(PreCompileOptsResult).name()), "/PreCompileOptsResult.log"},
        {std::string(typeid(PrologueResult).name()), "/PrologueResult.log"},
        {std::string(typeid(GenKernelResult).name()), "/GenKernelResult.log"},
        {std::string(typeid(EpilogueResult).name()), "/EpilogueResult.log"},
        {std::string(typeid(FatbinLinkResult).name()), "/FatbinLinkResult.log"}
    };
    const std::string resultInfoTempPath = ascStructType2LogFileMap.at(std::string(typeid(T).name()));
    nlohmann::json jsonObj = inputStruct;
    const std::string jsonStr = jsonObj.dump();
    if (AscPlugin::InfoManager::GetInstance().SaveTempRequested()) {
        std::string fullPath = AscPlugin::InfoManager::GetInstance().GetTempPath() + resultInfoTempPath;
        std::ofstream outFile;
        outFile.open(fullPath, std::ios::app);
        if (!outFile) {
            ASC_LOGE("Failed to create outFile[%s]!", fullPath.c_str());
            return ASC_FAILURE;
        }
        WriteFields(outFile, inputStruct); // 字段写入通过重载实现
        outFile.close();
    }
    auto ptrRes = strdup(jsonStr.c_str());    // Note: this ptr will be freed by bisheng
    if (ptrRes == nullptr) {
        ASC_LOGE("DumpResultInfo failed because of strdup return nullptr. jsonStr is %s.", jsonStr.c_str());
        return ASC_NULLPTR;
    }
    *result = ptrRes;
    return ASC_SUCCESS;
}

template int32_t DumpResultInfo<PreCompileOptsResult>(PreCompileOptsResult&, const char**);
template int32_t DumpResultInfo<PrologueResult>(PrologueResult&, const char**);
template int32_t DumpResultInfo<GenKernelResult>(GenKernelResult&, const char**);
template int32_t DumpResultInfo<EpilogueResult>(EpilogueResult&, const char**);
template int32_t DumpResultInfo<FatbinLinkResult>(FatbinLinkResult&, const char**);

} // namespace AscPlugin