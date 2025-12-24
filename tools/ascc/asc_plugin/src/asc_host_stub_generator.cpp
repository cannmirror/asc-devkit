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
    const std::unordered_set<KernelMetaType>& kernelType) : kernelInfo_(kernelInfo), kernelType_(kernelType) {}

std::string AscHostStubGenerator::GenStubFuncDecl(bool hasNameSpace, bool hasAnonymousSpace) const
{
    std::string functionEntryReplace = "";
    std::string paramsList = "(uint32_t __ascendc_blockDim, void* __ascendc_hold, void* __ascendc_stream";
    for (auto &param : kernelInfo_.kernelParameters) {
        paramsList += ", " + param.type + " " + param.name;
    }
    paramsList += ")";
    std::string kernelNameSpace = "";
    if (hasAnonymousSpace) {
        for (const auto& spaceName : kernelInfo_.namespaces) {
            if (spaceName == std::string(ANONYMOUS_NAME)) {
                functionEntryReplace += "namespace {\n";
            } else {
                functionEntryReplace += "namespace " + spaceName + " {\n";
            }
        }
    } else {
        for (auto &nameSpace : kernelInfo_.namespaces) {
            kernelNameSpace += nameSpace + "::";
        }
    }
    std::string kernelName = hasNameSpace ? kernelNameSpace + kernelInfo_.kernelName : kernelInfo_.kernelName;
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
        functionEntryReplace += "void " + kernelName + paramsList;
    } else {
        functionEntryReplace += tempParamDecl + " void " + kernelName + paramsList;
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

void AscHostStubGenerator::GenStubFuncImpl()
{
    auto &infoManager = InfoManager::GetInstance();
    uint32_t maxCoreNum = infoManager.GetMaxCoreNum();
    bool isSupportFifoDump = infoManager.IsSupportFifoDump();
    KernelMetaType defaultKtype = ExtractKernelType(kernelType_);
    std::ostringstream funcImplCode;
    bool hasAnonymous = false;
    auto it = std::find(kernelInfo_.namespaces.begin(), kernelInfo_.namespaces.end(), std::string(ANONYMOUS_NAME));
    if (it != kernelInfo_.namespaces.end()) {
        hasAnonymous = true;
    }
    funcImplCode << GenStubFuncDecl(/* hasNameSpace = */true, hasAnonymous) << "\n{\n";
    funcImplCode << "    struct {\n";
    if (!isSupportFifoDump && infoManager.IsDumpOn()) {
        funcImplCode << "        void* __ascendc_dump;\n";
    }
    for (auto &param : kernelInfo_.kernelParameters) {
        funcImplCode << "        alignas(((alignof(" << MapParamTypeToVoid(param.type) << ") + 3) >> 2) << 2) "
                     << MapParamTypeToVoid(param.type) << " " << param.name << ";\n";
    }
    funcImplCode << "    } __ascendc_args {";
    if (!isSupportFifoDump && infoManager.IsDumpOn()) {
        funcImplCode << "nullptr, ";
    }
    for (auto &param : kernelInfo_.kernelParameters) {
        funcImplCode << param.name << ", ";
    }
    funcImplCode << "};\n";

    // args declare code
    funcImplCode << "    uint32_t __ascendc_ret;\n";
    if (!isSupportFifoDump && infoManager.IsDumpOn()) {
        funcImplCode << "    constexpr uint32_t __ascendc_one_core_dump_size = "
                     << std::to_string(infoManager.GetOneCoreDumpSize()) << ";\n";
        funcImplCode << "    AllocAscendMemDevice(&(__ascendc_args.__ascendc_dump), __ascendc_one_core_dump_size * "
                     << maxCoreNum << ");\n";
    }
    funcImplCode << "    const char* __ascendc_name = \"" << kernelInfo_.kernelName << "\";\n";

    // when no template, only has 1 kernel type
    funcImplCode << "    uint32_t __ascendc_kType = " << KTYPE_TO_LAUNCH_PARAMS.at(defaultKtype) << ";\n";
    funcImplCode << ManglingNameJudgeCode();

    if (!isSupportFifoDump && infoManager.IsDumpOn() && infoManager.HasAssert()) {
        funcImplCode << "    __ascendc_ret = "
                        "AscPluginGenerator::ascendc_set_exception_dump_info(__ascendc_one_core_dump_size);\n";
        funcImplCode << "    if(__ascendc_ret != 0) {\n";
        funcImplCode << "        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "
                        "\"init assert dump failure!\");\n";
        funcImplCode << "        return;\n";
        funcImplCode << "    }\n";
    }
    funcImplCode << "    __ascendc_ret = AscPluginGenerator::LaunchAndProfiling(__ascendc_manglingName, "
        "__ascendc_blockDim, __ascendc_stream, (void **)&__ascendc_args, sizeof(__ascendc_args), __ascendc_kType);\n";
    funcImplCode << "    if(__ascendc_ret != 0) {\n";
    funcImplCode << "        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "
                    "\"kernel launch failure!\");\n";
    funcImplCode << "        return;\n";
    funcImplCode << "    }\n";
    funcImplCode << "    AscPluginGenerator::GetHandleUnregisterInst();\n";
    if (!isSupportFifoDump && infoManager.IsDumpOn() && infoManager.HasPrintf()) {
        funcImplCode << "    Adx::AdumpPrintWorkSpace(__ascendc_args.__ascendc_dump, __ascendc_one_core_dump_size * "
                     << maxCoreNum << ", __ascendc_stream, __ascendc_name);\n";
    }
    if (!isSupportFifoDump && infoManager.IsDumpOn()) {
        funcImplCode << "    FreeAscendMemDevice(__ascendc_args.__ascendc_dump);\n";
    }
    funcImplCode << "}\n";
    if (hasAnonymous) {
        for (size_t i = 0; i < kernelInfo_.namespaces.size(); ++i) {
            funcImplCode << "}\n";
        }
    }
    kernelCallStub_ << funcImplCode.str();
}

std::string AscHostStubGenerator::GenCode()
{
    ASC_LOGI("Kernel [%s] : generate host stub.", kernelInfo_.kernelName.c_str());
    GenStubFuncImpl();
    ASC_LOGD("type judge code is [\n%s\n]", typeJudgePreCode_.str().c_str());
    ASC_LOGD("host stub code is [\n%s\n]", kernelCallStub_.str().c_str());
    return typeJudgePreCode_.str() + kernelCallStub_.str();
}
} // namespace AscPlugin
