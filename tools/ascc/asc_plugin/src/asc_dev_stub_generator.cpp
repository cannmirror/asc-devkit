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
 * \file asc_dev_stub_generator.cpp
 * \brief
 */

#include "asc_dev_stub_generator.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <tuple>

#include "asc_log.h"
#include "asc_utils.h"
#include "asc_info_manager.h"

namespace AscPlugin {
namespace {
enum class ParamJoinType : uint8_t {
    ONLY_NAME = 0,
    HAS_TYPE = 1,
    HAS_TYPE_ATTRI = 2,
};

inline std::string JoinParamWithComma(
    const std::vector<Param> &params, ParamJoinType type = ParamJoinType::HAS_TYPE_ATTRI)
{
    if (params.empty()) {
        return "";
    }
    std::string res;
    size_t counter = 0;
    for (const auto& param : params) {
        if (counter > 0) {
            res += ", ";
        }
        if (type == ParamJoinType::HAS_TYPE_ATTRI && !param.attribute.empty()) {
            res += param.attribute + " ";
        }
        if (type == ParamJoinType::HAS_TYPE || type == ParamJoinType::HAS_TYPE_ATTRI) {
            res += param.type + " ";
        }
        res += param.name;
        ++counter;
    }
    return res;
}

inline std::string JoinStringWithDelimiter(const std::vector<std::string>& params, const char* delimiter)
{
    if (params.empty()) {
        return "";
    }
    std::string res;
    size_t counter = 0;
    for (const auto& str : params) {
        if (counter > 0) {
            res += delimiter;
        }
        res += str;
        ++counter;
    }
    return res;
}
}

AscDevStubGenerator::AscDevStubGenerator(const KernelInfo &kernelInfo, const std::unordered_set<KernelMetaType>& kernelType,
    const KfcScene &kfcScene) : kernelInfo_(kernelInfo), kernelType_(kernelType), kfcScene_(kfcScene)
{
    // init stringstream
    std::string buffer;
    constexpr size_t codeBuffLen = 16 * 1024;
    buffer.reserve(codeBuffLen);
    codeStream_.str(std::move(buffer));
    socVersion_ = InfoManager::GetInstance().GetShortSocVersion();
}

std::string AscDevStubGenerator::GetWorkspaceArgName() const
{
    for (const auto& param : kernelInfo_.kernelParameters) {
        if (param.attribute.find(std::string("cce_kfc_workspace")) != std::string::npos) {
            ASC_LOGI("Kernel [%s] : the kernel utilizes the workspace.", kernelInfo_.kernelName.c_str());
            return param.name;
        }
    }
    ASC_LOGI("Kernel [%s] : the kernel does not utilize the workspace.", kernelInfo_.kernelName.c_str());
    return "";
}

void AscDevStubGenerator::GenStubFuncDecl(const std::string& globalSymbol, const std::vector<Param>& args,
    const KernelMetaType& kernelType)
{
    std::string fixGlobalSymbol = std::string(DEVICE_STUB_PREFIX) + globalSymbol;
    codeStream_ << "extern \"C\" ";
    if (kernelType == KernelMetaType::KERNEL_TYPE_AIV_ONLY ||
        kernelType == KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0) {
        codeStream_ << "__attribute__((aiv)) ";
    } else if (kernelType == KernelMetaType::KERNEL_TYPE_AIC_ONLY ||
               kernelType == KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0) {
        codeStream_ << "__attribute__((aic)) ";
    }
    codeStream_ << "__global__ __aicore__ void " << fixGlobalSymbol << "(";
    codeStream_ << JoinParamWithComma(args) << ")\n";
    auto& infoManager = InfoManager::GetInstance();
    // collect global mangling name for host codegen
    infoManager.AddGlobalSymbolInfo(fixGlobalSymbol, kernelType, kernelInfo_.fileName, kernelInfo_.lineNum,
        kernelInfo_.colNum, kfcScene_);
}

void AscDevStubGenerator::StubFuncWorkSpaceImpl()
{
    std::string workspaceArgName = GetWorkspaceArgName();
    if (workspaceArgName.empty() || kfcScene_ == KfcScene::Close) {
        ASC_LOGI("Kernel [%s] : the kernel no need to implement workspace.", kernelInfo_.kernelName.c_str());
        return;
    }
    codeStream_ << "    GM_ADDR ascendc_workspace_param = " << workspaceArgName << ";\n";
    codeStream_ << "    AscendC::SetSysWorkspaceForce(ascendc_workspace_param);\n";
    codeStream_ << "    GM_ADDR ascendc_workspace_usr = AscendC::GetUserWorkspace(ascendc_workspace_param);\n";
    if (socVersion_ != ShortSocVersion::ASCEND950) {
        codeStream_ << "    if constexpr (g_coreType == AscendC::AIC) {\n";
        codeStream_ << "        matmul::clearWorkspace(ascendc_workspace_param);\n";
        codeStream_ << "    }\n";
    }
    codeStream_ << "    " << workspaceArgName << " = ascendc_workspace_usr;\n";
}

void AscDevStubGenerator::StubFuncCallImpl(const std::string& templateArgs)
{
    codeStream_ << "    " << ORIGIN_KERNEL_PREFIX << kernelInfo_.kernelName;
    if (!templateArgs.empty()) {
        codeStream_ << "<" << templateArgs << ">";
    }
    codeStream_ << "(" << JoinParamWithComma(kernelInfo_.kernelParameters, ParamJoinType::ONLY_NAME) << ");\n";
    if (socVersion_ == ShortSocVersion::ASCEND950) {
        codeStream_ << "    pipe_barrier(PIPE_ALL);\n";
        codeStream_ << "    dsb(mem_dsb_t::DSB_ALL);\n";
        codeStream_ << "    dci();\n";
    }
}

void AscDevStubGenerator::GenStubFuncImpl(const std::string& templateArgs)
{
    codeStream_ << "{\n";
    StubFuncWorkSpaceImpl();
    StubFuncCallImpl(templateArgs);
    codeStream_ << "}\n";
}

void AscDevStubGenerator::GenStubKernelFunc(const bool hasAnonymous)
{
    KernelMetaType curKernelType = ExtractKernelType(kernelType_);
    if (!kernelInfo_.namespaces.empty()) {
        if (!hasAnonymous) {
            codeStream_ << "namespace " << JoinStringWithDelimiter(kernelInfo_.namespaces, "::") << " {\n";
        } else {
            for (const auto& spaceName : kernelInfo_.namespaces) {
                if (spaceName == std::string(ANONYMOUS_NAME)) {
                    codeStream_ << "namespace {\n";
                } else {
                    codeStream_ << "namespace " << spaceName << " {\n";
                }
            }
        }
    }
    GenStubFuncDecl(kernelInfo_.kernelMangledName, kernelInfo_.kernelParameters, curKernelType);
    GenStubFuncImpl();
    if (kernelInfo_.namespaces.empty()) {
        return;
    }
    size_t closeCount = hasAnonymous ? kernelInfo_.namespaces.size() : 1;
    for (size_t i = 0; i < closeCount; ++i) {
        codeStream_ << "}\n";
    }
}

std::string AscDevStubGenerator::GetTempArgsList(const TemplateInstance &tempInst)
{
    std::vector<std::string> templateArgList;
    for (size_t i = 0; i < kernelInfo_.templateParameters.size(); ++i) {
        std::string templateArgName = kernelInfo_.templateParameters[i].typeClass == ParamType::TEMPLATE_INT
                                          ? std::string("static_cast<") + kernelInfo_.templateParameters[i].type +
                                                ">(" + tempInst.templateInstantiationArguments[i] + ")"
                                          : tempInst.templateInstantiationArguments[i];
        templateArgList.emplace_back(templateArgName);
    }
    return JoinStringWithDelimiter(templateArgList, ", ");
}

void AscDevStubGenerator::GenStubKernelFunc(const bool hasAnonymous, const TemplateInstance tempInst)
{
    if (!kernelInfo_.namespaces.empty()) {
        if (!hasAnonymous) {
            codeStream_ << "namespace " << JoinStringWithDelimiter(kernelInfo_.namespaces, "::") << " {\n";
        } else {
            for (const auto& spaceName : kernelInfo_.namespaces) {
                if (spaceName == std::string(ANONYMOUS_NAME)) {
                    codeStream_ << "namespace {\n";
                } else {
                    codeStream_ << "namespace " << spaceName << " {\n";
                }
            }
        }
    }
    KernelMetaType defaultKtype = ExtractKernelType(kernelType_);
    GenStubFuncDecl(tempInst.instanceMangledName, tempInst.instanceKernelParameters,
        GetBishengKTypeByCoreRatio(tempInst.ratio, defaultKtype));
    GenStubFuncImpl(GetTempArgsList(tempInst));
    if (kernelInfo_.namespaces.empty()) {
        return;
    }
    size_t closeCount = hasAnonymous ? kernelInfo_.namespaces.size() : 1;
    for (size_t i = 0; i < closeCount; ++i) {
        codeStream_ << "}\n";
    }
}

std::string AscDevStubGenerator::GenCode()
{
    ASC_LOGI("Kernel [%s] : generate device stub function.", kernelInfo_.kernelName.c_str());
    bool hasAnonymous = false;
    auto it = std::find(kernelInfo_.namespaces.begin(), kernelInfo_.namespaces.end(), std::string(ANONYMOUS_NAME));
    if (it != kernelInfo_.namespaces.end()) {
        hasAnonymous = true;
    }
    if (kernelInfo_.isTemplate) {
        for (const auto& inst : kernelInfo_.templateInstances) {
            // isMixInst = inst.==mix
            GenStubKernelFunc(hasAnonymous, inst);
        }
    } else {
        GenStubKernelFunc(hasAnonymous);
    }

    ASC_LOGD(
        "Kernel [%s] : device stub function is [\n%s\n]", kernelInfo_.kernelName.c_str(), codeStream_.str().c_str());
    return codeStream_.str();
}

} // namespace AscPlugin
