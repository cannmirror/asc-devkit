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
constexpr size_t CODE_BUFFER_LEN = 16 * 1024;
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
    buffer.reserve(CODE_BUFFER_LEN);
    codeStream_.str(std::move(buffer));

    // init dump flag
    bool isDumpCloseManual = !InfoManager::GetInstance().UserDumpRequested();
    bool printfFlag = InfoManager::GetInstance().HasPrintf();
    bool assertFlag = InfoManager::GetInstance().HasAssert();
    dumpTypeIsNotNone_ = !isDumpCloseManual && (printfFlag || assertFlag);
    dumpTypeIsPrintf_ = !isDumpCloseManual && printfFlag;
    dumpAscendCStamp_ = false;

    // save origin params name
    originParamsCallList_ = JoinParamWithComma(kernelInfo_.kernelParameters, ParamJoinType::ONLY_NAME);
    workspaceArgName_ = GetWorkspaceArgName();
}

std::string AscDevStubGenerator::GetWorkspaceArgName() const
{
    const auto& hasWorkspace = InfoManager::GetInstance().HasWorkspace();
    const auto& hasTiling = InfoManager::GetInstance().HasTiling();
    if (hasWorkspace && !hasTiling) {
        const size_t idx = kernelInfo_.kernelParameters.size() - 1;
        if (idx >= kernelInfo_.kernelParameters.size()) {
            ASC_LOGW(
                "Kernel [%s] : use workspace failure, must have GM_ADDR argument.", kernelInfo_.kernelName.c_str());
            return "";
        }
        return kernelInfo_.kernelParameters[kernelInfo_.kernelParameters.size() - 1].name;
    } else if (hasWorkspace && hasTiling) {
        const size_t idx = kernelInfo_.kernelParameters.size() - 2;
        if (idx >= kernelInfo_.kernelParameters.size()) {
            ASC_LOGW("Kernel [%s] : use workspace failure, when both 'HAVE_TILING' and 'HAVE_WORKSPACE' macros are "
                     "specified, the number of function arguments must exceed 2.",
                kernelInfo_.kernelName.c_str());
            return "";
        }
        return kernelInfo_.kernelParameters[kernelInfo_.kernelParameters.size() - 2].name;
    }
    for (const auto& param : kernelInfo_.kernelParameters) {
        if (param.attribute.find(std::string("kfc_workspace")) != std::string::npos) {
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

void AscDevStubGenerator::StubFuncDumpAndHardSyncImpl(const bool& isMix, const bool& isHardSync)
{
    if (dumpTypeIsNotNone_ || dumpAscendCStamp_) {
        if (isMix) {
            codeStream_ << "    AscendC::InitDump(true, __ascendc_dump_addr, ONE_CORE_DUMP_SIZE);\n";
        } else {
            codeStream_ << "    AscendC::InitDump(false, __ascendc_dump_addr, ONE_CORE_DUMP_SIZE);\n";
        }
        if (dumpAscendCStamp_) {
            codeStream_ << "    "
                        "AscendC::AscendCTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_INIT_"
                        "DUMP));\n";
        }
    }
    if (isHardSync) {
        codeStream_ << "    icache_preload(1);\n";
        codeStream_ << "    if (g_sysFftsAddr != nullptr) {\n";
        codeStream_ << "        set_ffts_base_addr((uint64_t)g_sysFftsAddr);\n";
        codeStream_ << "    }\n";
        if (dumpAscendCStamp_) {
            codeStream_ <<
                "    "
                "AscendC::AscendCTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_FFTS_ADDR));\n";
        }
    }
    if (dumpTypeIsPrintf_) {
        codeStream_ << "    uint64_t __ascendc_timestamp = 0;\n";
        codeStream_ << "    uint64_t __ascendc_version = 0;\n";
        codeStream_ << "     __gm__ char* __ascendc_version_str = nullptr;\n";
        codeStream_ << "    GetCannVersion(__ascendc_version_str, __ascendc_version, __ascendc_timestamp);\n";
        codeStream_ << "    if (__ascendc_timestamp == 0) {\n";
        codeStream_ << "        AscendC::printf(\"[WARNING]: CANN TimeStamp is invalid, ";
        codeStream_ << "CANN TimeStamp is %u\\n\", __ascendc_timestamp);\n    } else {\n";
        codeStream_ << "        AscendC::printf(\"CANN Version: %s, TimeStamp: %u\\n\", ";
        codeStream_ << "(__gm__ const char*)(__ascendc_version_str), __ascendc_timestamp);\n";
        codeStream_ << "    }\n";
    }
}

void AscDevStubGenerator::StubFuncWorkSpaceImpl(const bool& isMix)
{
    if (workspaceArgName_.empty() || kfcScene_ == KfcScene::Close) {
        ASC_LOGI("Kernel [%s] : the kernel no need to implement workspace.", kernelInfo_.kernelName.c_str());
        return;
    }
    codeStream_ << "    GM_ADDR ascendc_workspace_param;\n";
    codeStream_ << "    GM_ADDR ascendc_workspace_usr;\n";
    codeStream_ << "    ascendc_workspace_param = " << workspaceArgName_ << ";\n";
    if (isMix) {
        codeStream_ << "    if (ascendc_workspace_param == nullptr) {\n";
        codeStream_ << "        return;\n";
        codeStream_ << "    }\n";
    }
    codeStream_ << "    AscendC::SetSysWorkspaceForce(ascendc_workspace_param);\n";
    codeStream_ << "    ascendc_workspace_usr = AscendC::GetUserWorkspace(ascendc_workspace_param);\n";
    if (isMix && kfcScene_ == KfcScene::Open) {
        codeStream_ << "    if constexpr (g_coreType == AscendC::AIC) {\n";
        codeStream_ << "        matmul::clearWorkspace(ascendc_workspace_param);\n";
        if (dumpAscendCStamp_) {
            codeStream_ << "        AscendC::AscendCTimeStamp(static_cast<uint32_t>";
            codeStream_ << "(AscendC::TimeStampId::TIME_STAMP_WRAP_CLEAR_WK_SPAC));\n";
        }
        codeStream_ << "    }\n";
    }
    codeStream_ << "    " << workspaceArgName_ << " = ascendc_workspace_usr;\n";
}

void AscDevStubGenerator::StubFuncCallImpl(const std::string& templateArgs)
{
    codeStream_ << "    " << ORIGIN_KERNEL_PREFIX << kernelInfo_.kernelName;
    if (!templateArgs.empty()) {
        codeStream_ << "<" << templateArgs << ">";
    }
    codeStream_ << "(" << originParamsCallList_ << ");\n";
    ShortSocVersion shortSoc = InfoManager::GetInstance().GetShortSocVersion();
    if (shortSoc == ShortSocVersion::ASCEND910_95) {
        codeStream_ << "    pipe_barrier(PIPE_ALL);\n";
        codeStream_ << "    dsb(mem_dsb_t::DSB_ALL);\n";
        codeStream_ << "    dci();\n";
    }
}

std::pair<bool, bool> AscDevStubGenerator::GetArchInfo(const ShortSocVersion& socVersion) const
{
    bool isMix = false;
    bool isHardSync = false;
    KernelMetaType curKernelType = ExtractKernelType(kernelType_);
    if (socVersion == ShortSocVersion::ASCEND910B) {
        if (kernelType_.size() > 1) { // core_ratio(x, y) is always mix
            isMix = true;
        } else {
            KernelMetaType kernelType = curKernelType;
            isMix = kernelType == KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0 ||
                kernelType == KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0 ||
                kernelType == KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1 ||
                kernelType == KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2;
        }
        isHardSync = isMix;
    }
    ASC_LOGI("Kernel [%s] : is mix [%s], is hard sync [%s].", kernelInfo_.kernelName.c_str(), isMix ? "true" : "false",
        isHardSync ? "true" : "false");
    return {isMix, isHardSync};
}

void AscDevStubGenerator::GenStubFuncImpl(const bool& isMix, const bool& isHardSync, const std::string& templateArgs)
{
    codeStream_ << "{\n";
    StubFuncDumpAndHardSyncImpl(isMix, isHardSync);
    StubFuncWorkSpaceImpl(isMix);
    StubFuncCallImpl(templateArgs);
    codeStream_ << "}\n";
}

void AscDevStubGenerator::UpdateParams()
{
    if (dumpTypeIsNotNone_) {
        kernelInfo_.kernelParameters.emplace(kernelInfo_.kernelParameters.begin(), "uint8_t *", "__ascendc_dump_addr",
            false, "", "__attribute__((cce_global))");
        for (auto &inst : kernelInfo_.templateInstances) {
            inst.instanceKernelParameters.emplace(inst.instanceKernelParameters.begin(),
                "uint8_t *",
                "__ascendc_dump_addr",
                false,
                "",
                "__attribute__((cce_global))");
        }
    }
    kernelInfo_.kernelParameters.emplace_back("uint8_t *", "__ascendc_overflow_status", false, "",
        "__attribute__((cce_global))");
    for (auto &inst : kernelInfo_.templateInstances) {
        inst.instanceKernelParameters.emplace_back(
            "uint8_t *", "__ascendc_overflow_status", false, "", "__attribute__((cce_global))");
    }
}

void AscDevStubGenerator::GenStubKernelFunc(const bool& isMix, const bool& isHardSync)
{
    KernelMetaType curKernelType = ExtractKernelType(kernelType_);
    if (!kernelInfo_.namespaces.empty()) {
        codeStream_ << "namespace " << JoinStringWithDelimiter(kernelInfo_.namespaces, "::") << " {\n";
    }
    GenStubFuncDecl(kernelInfo_.kernelMangledName, kernelInfo_.kernelParameters, curKernelType);
    GenStubFuncImpl(isMix, isHardSync, "");
    if (!kernelInfo_.namespaces.empty()) {
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

void AscDevStubGenerator::GenStubKernelFunc(const bool& isMix, const bool& isHardSync, const TemplateInstance& tempInst)
{
    if (!kernelInfo_.namespaces.empty()) {
        codeStream_ << "namespace " << JoinStringWithDelimiter(kernelInfo_.namespaces, "::") << " {\n";
    }
    KernelMetaType defaultKtype = ExtractKernelType(kernelType_);
    GenStubFuncDecl(tempInst.instanceMangledName, tempInst.instanceKernelParameters,
        GetBishengKTypeByCoreRatio(tempInst.ratio, defaultKtype));
    const std::string& templateArgs = GetTempArgsList(tempInst);
    GenStubFuncImpl(isMix, isHardSync, templateArgs);
    if (!kernelInfo_.namespaces.empty()) {
        codeStream_ << "}\n";
    }
}

void AscDevStubGenerator::GenCodeForL2Cache()
{
    auto& infoManager = InfoManager::GetInstance();
    if (infoManager.IsFirstKernel() && infoManager.IsL2CacheEnabled() && !infoManager.HasOpSystemCfg()) {
        codeStream_ << "inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg = {0};\n";
    }
}

std::string AscDevStubGenerator::GenCode()
{
    ASC_LOGI("Kernel [%s] : generate device stub function.", kernelInfo_.kernelName.c_str());
    const auto [isMix, isHardSync] = GetArchInfo(socVersion_);
    UpdateParams();
    GenCodeForL2Cache();
    if (kernelInfo_.isTemplate) {
        for (const auto& inst : kernelInfo_.templateInstances) {
            // isMixInst = inst.==mix
            GenStubKernelFunc(isMix, isHardSync, inst);
        }
    } else {
        GenStubKernelFunc(isMix, isHardSync);
    }

    ASC_LOGD(
        "Kernel [%s] : device stub function is [\n%s\n]", kernelInfo_.kernelName.c_str(), codeStream_.str().c_str());
    return codeStream_.str();
}

} // namespace AscPlugin
