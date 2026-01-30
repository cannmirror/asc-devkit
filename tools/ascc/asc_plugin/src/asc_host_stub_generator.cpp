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
 * \file asc_host_stub_generator.cpp
 * \brief
 */

#include "asc_host_stub_generator.h"

#include <string>
#include <vector>
#include <atomic>
#include <cstdio>

#include "asc_log.h"
#include "asc_utils.h"

namespace AscPlugin {

static const std::unordered_map<KernelMetaType, const char*> KTYPE_TO_LAUNCH_PARAMS = {
    {KernelMetaType::KERNEL_TYPE_AICORE, "2"},           // CodeMode::AIC in constants.py
    {KernelMetaType::KERNEL_TYPE_VECTOR_CORE, "2"},      // CodeMode::AIC in constants.py
    {KernelMetaType::KERNEL_TYPE_MIX_VECTOR_CORE, "4"},
    {KernelMetaType::KERNEL_TYPE_AIV_ONLY, "5"},
    {KernelMetaType::KERNEL_TYPE_AIC_ONLY, "6"},
    {KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0, "7"},
    {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0, "8"},
    {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1, "9"},
    {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2, "10"}
};


class UniqueFuncName {
public:
    static std::string Generate(const std::string& prefix) {
        static std::atomic<size_t> counter{0};
        return prefix + std::to_string(counter++);
    }
};
AscHostStubGenerator::AscHostStubGenerator(const KernelInfo& kernelInfo,
    const std::unordered_set<KernelMetaType>& kernelType) : kernelInfo_(kernelInfo), kernelType_(kernelType)
{
    // init stringstream
    std::string buffer;
    constexpr size_t codeBuffLen = 16 * 1024;
    buffer.reserve(codeBuffLen);
    kernelCallStub_.str(std::move(buffer));
}

std::string AscHostStubGenerator::GenStubFuncDecl() const
{
    std::string functionEntryReplace = "";
    auto &infoManager = InfoManager::GetInstance();
    ShortSocVersion shortSoc = infoManager.GetShortSocVersion();
    std::string paramsList = "";
    if (shortSoc == ShortSocVersion::ASCEND950) {
        paramsList = "(uint32_t __ascendc_blockDim, uint32_t __ascendc_ubufDynamicSize, void* __ascendc_stream";
    } else {
        paramsList = "(uint32_t __ascendc_blockDim, void* __ascendc_hold, void* __ascendc_stream";
    }
    for (auto &param : kernelInfo_.kernelParameters) {
        paramsList += ", " + param.type + " " + param.name;
    }
    paramsList += ")";
    if (hasAnonymousSpace_) {
        for (const auto& spaceName : kernelInfo_.namespaces) {
            if (spaceName == std::string(ANONYMOUS_NAME)) {
                functionEntryReplace += "namespace {\n";
            } else {
                functionEntryReplace += "namespace " + spaceName + " {\n";
            }
        }
    }
    std::string tempParamDecl;
    if (kernelInfo_.isTemplate) {
        tempParamDecl = "template<";
        for (size_t i = 0; i < kernelInfo_.templateParameters.size(); i++) {
            if (i > 0) {
                tempParamDecl += ", ";
            }
            tempParamDecl += kernelInfo_.templateParameters[i].type + " " + kernelInfo_.templateParameters[i].name;
        }
        tempParamDecl += ">";
    }
    if (!kernelInfo_.isTemplate) {
        functionEntryReplace += "void " + kernelNameWithNameSpace_ + paramsList;
    } else {
        functionEntryReplace += tempParamDecl + " void " + kernelNameWithNameSpace_ + paramsList;
    }
    return functionEntryReplace;
}

// TEMPLATE_TYPE = 1, // typename or class
// TEMPLATE_INT = 2, // num or enum
// TEMPLATE_DECL = 3, // const auto& ...
// TEMPLATE_TEMPLATE = 4, // template <typename, typename> class ...
std::string AscHostStubGenerator::ManglingNameJudgeCode()
{
    std::string judgeCode = std::string();
    if (kernelInfo_.templateInstances.empty()) {
        judgeCode += "    const char* __ascendc_manglingName = \"" + kernelInfo_.kernelMangledName + "\";\n";
        return judgeCode;
    }
    judgeCode += "    const char* __ascendc_manglingName = nullptr;\n";
    KernelMetaType defaultKtype = ExtractKernelType(kernelType_);
    for (size_t j = 0; j < kernelInfo_.templateInstances.size(); j++) {
        TemplateInstance instFuncInfo = kernelInfo_.templateInstances[j];
        KernelMetaType kType = GetBishengKTypeByCoreRatio(instFuncInfo.ratio, defaultKtype);
        judgeCode += "    if constexpr (";
        for (size_t i = 0; i < instFuncInfo.templateInstantiationArguments.size(); ++i) {
            if (i > 0) {
                judgeCode += " && ";
            }
            const auto& declTempArgs = kernelInfo_.templateParameters[i];
            const auto& instTempArgs = instFuncInfo.templateInstantiationArguments[i];
            if (declTempArgs.typeClass == ParamType::TEMPLATE_TYPE) {
                judgeCode += "AscendC::Std::is_same<" + declTempArgs.name + ", " + instTempArgs + ">::value";
            } else if (declTempArgs.typeClass == ParamType::TEMPLATE_INT) {
                judgeCode += declTempArgs.name + " == static_cast<" + declTempArgs.type + ">(" + instTempArgs + ")";
            } else if (declTempArgs.typeClass == ParamType::TEMPLATE_DECL) {
                judgeCode += "&" + declTempArgs.name + " == &" + instTempArgs;
            } else if (declTempArgs.typeClass == ParamType::TEMPLATE_TEMPLATE) {
                std::string typeFuncName = UniqueFuncName::Generate("__AsccIsMyType");
                judgeCode += typeFuncName + "<" + declTempArgs.name + ">::value";
                typeJudgePreCode_ << "template <" + declTempArgs.type + " " + declTempArgs.name + ">\n";
                typeJudgePreCode_ << "struct " + typeFuncName + " : AscendC::Std::false_type {};\n";
                typeJudgePreCode_ << "template <>\n";
                typeJudgePreCode_ << "struct " + typeFuncName + "<" + instTempArgs +
                                         "> : AscendC::Std::true_type {};\n";
            }
        }
        judgeCode += ") {\n";
        judgeCode += "        __ascendc_manglingName = \"" + instFuncInfo.instanceMangledName + "\";\n";
        judgeCode += "        __ascendc_kType = " + std::string(KTYPE_TO_LAUNCH_PARAMS.at(kType)) + ";\n";
        judgeCode += "    }\n";
    }
    judgeCode += "    if (__ascendc_manglingName == nullptr) {\n";
    judgeCode += "        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "
                    "\"call kernel function failure!\");\n";
    judgeCode += "        return;\n";
    judgeCode += "    }\n";
    return judgeCode;
}

inline std::string MapParamTypeToVoid(std::string paramType)
{
    return (paramType == "uint8_t *" || paramType == "unsigned char *") ? "void*" : paramType;
}

void AscHostStubGenerator::ParseKernelName()
{
    // judge the anonymous space
    auto it = std::find(kernelInfo_.namespaces.begin(), kernelInfo_.namespaces.end(), std::string(ANONYMOUS_NAME));
    if (it != kernelInfo_.namespaces.end()) {
        hasAnonymousSpace_ = true;
    }
    std::string kernelNameSpace = "";
    if (!hasAnonymousSpace_) {
        for (auto &nameSpace : kernelInfo_.namespaces) {
            kernelNameSpace += nameSpace + "::";
        }
    }
    kernelNameWithNameSpace_ = kernelNameSpace + kernelInfo_.kernelName;
}

void AscHostStubGenerator::GenStubFuncImpl()
{
    auto& infoManager = InfoManager::GetInstance();
    KernelMetaType defaultKtype = ExtractKernelType(kernelType_);
    kernelCallStub_ << GenStubFuncDecl() << "\n{\n";
    kernelCallStub_ << "    struct {\n";
    for (auto& param : kernelInfo_.kernelParameters) {
        kernelCallStub_ << "        alignas(((alignof(" << MapParamTypeToVoid(param.type) << ") + 3) >> 2) << 2) "
                        << MapParamTypeToVoid(param.type) << " " << param.name << ";\n";
    }
    kernelCallStub_ << "    } __ascendc_args {";
    for (auto& param : kernelInfo_.kernelParameters) {
        kernelCallStub_ << param.name << ", ";
    }
    kernelCallStub_ << "};\n";

    // args declare code
    kernelCallStub_ << "    uint32_t __ascendc_ret;\n";
    kernelCallStub_ << "    const char* __ascendc_name = \"" << kernelInfo_.kernelName << "\";\n";

    // when no template, only has 1 kernel type
    kernelCallStub_ << "    uint32_t __ascendc_kType = " << KTYPE_TO_LAUNCH_PARAMS.at(defaultKtype) << ";\n";
    kernelCallStub_ << ManglingNameJudgeCode();

    const char* fmtLaunchAndProfiling =
        "    __ascendc_ret = AscPluginGenerator::LaunchAndProfiling(__ascendc_manglingName, "
        "__ascendc_blockDim, __ascendc_stream, (void **)&__ascendc_args, sizeof(__ascendc_args), "
        "__ascendc_kType, %s);\n";
    constexpr uint32_t bufMaxSize = 512;
    char buffer[bufMaxSize];
    ShortSocVersion shortSoc = infoManager.GetShortSocVersion();
    if (shortSoc == ShortSocVersion::ASCEND950) {
        snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, fmtLaunchAndProfiling, "__ascendc_ubufDynamicSize");
    } else {
        snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, fmtLaunchAndProfiling, "0");
    }
    kernelCallStub_ << buffer;

    kernelCallStub_ << "    if(__ascendc_ret != 0) {\n";
    kernelCallStub_ << "        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "
                       "\"kernel launch failure!\");\n";
    kernelCallStub_ << "        return;\n";
    kernelCallStub_ << "    }\n";
    kernelCallStub_ << "}\n";
    if (hasAnonymousSpace_) {
        for (size_t i = 0; i < kernelInfo_.namespaces.size(); ++i) {
            kernelCallStub_ << "}\n";
        }
    }
}

std::string AscHostStubGenerator::GenCode()
{
    ASC_LOGI("Kernel [%s] : generate host stub.", kernelInfo_.kernelName.c_str());
    ParseKernelName();
    GenStubFuncImpl();
    ASC_LOGD("type judge code is [\n%s\n]", typeJudgePreCode_.str().c_str());
    ASC_LOGD("host stub code is [\n%s\n]", kernelCallStub_.str().c_str());
    return typeJudgePreCode_.str() + kernelCallStub_.str();
}
} // namespace AscPlugin
