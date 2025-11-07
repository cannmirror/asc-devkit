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
 * \file asc_dev_meta_generator.cpp
 * \brief
 */

#include "asc_dev_meta_generator.h"

#include <string>
#include <array>
#include <unordered_map>

#include "asc_log.h"
#include "asc_utils.h"
#include "asc_info_manager.h"

namespace AscPlugin {
static constexpr size_t CODE_BUFFER_LEN = 8 * 1024;
static constexpr size_t META_INFO_LEN = 3;
static const std::unordered_map<KernelMetaType, std::array<const char*, META_INFO_LEN>> KERNEL_TYPE_TO_META_INFO = {
    // 310P. Note that KERNEL_TYPE_MIX_VECTOR_CORE does not have section
    {KernelMetaType::KERNEL_TYPE_AICORE, {"FunLevelKType", "{{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_AICORE}", ""}},
    // 910B
    {KernelMetaType::KERNEL_TYPE_AIC_ONLY, {"FunLevelKType", "{{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_AIC}", ""}},
    {KernelMetaType::KERNEL_TYPE_AIV_ONLY, {"FunLevelKType", "{{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_AIV}", ""}},
    {KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0,
        {"FunLevelMixCoreType",
            "{{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIV_MAIN}",
            "{{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 0, 1}"}},
    {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0,
        {"FunLevelMixCoreType",
            "{{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN}",
            "{{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 0}"}},
    {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1,
        {"FunLevelMixCoreType",
            "{{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN}",
            "{{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 1}"}},
    {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2,
        {"FunLevelMixCoreType",
            "{{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN}",
            "{{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2}"}}};

AscDevMetaGenerator::AscDevMetaGenerator(const KernelInfo &kernelInfo,
    const std::unordered_set<KernelMetaType>& kernelType) : kernelInfo_(kernelInfo), kernelType_(kernelType)
{
    std::string buffer;
    buffer.reserve(CODE_BUFFER_LEN);
    codeStream_.str(std::move(buffer));
}

void AscDevMetaGenerator::GenKtypeStruct(const KernelMetaType& kernelType, const char* globalSymbol,
    const char* kernelTypeSuffix)
{
    if (KERNEL_TYPE_TO_META_INFO.find(kernelType) == KERNEL_TYPE_TO_META_INFO.end()) {
        ASC_LOGW("Kernel [%s] : kernel type is illegal. Please check.", kernelInfo_.kernelName.c_str());
        return;
    }
    const auto [structType, ktypeSection, mixRationSection] = KERNEL_TYPE_TO_META_INFO.at(kernelType);
    codeStream_ << "static const struct " << structType << " " << globalSymbol << "_meta_section ";
    codeStream_ << "__attribute__((used, section(\".ascend.meta.";
    codeStream_ << globalSymbol << kernelTypeSuffix << "\"))) = { ";
    codeStream_ << ktypeSection << ", " << mixRationSection << " };";
}

void AscDevMetaGenerator::GenMetaSection(const char* globalSymbol, const KernelMetaType& kernelType)
{
    auto genKtypeWithArchMacro = [this, &globalSymbol, &kernelType](
                                     const char *archMacro, const char *suffix) {
        this->codeStream_ << "\n#if defined(" << archMacro << ")\n";
        this->GenKtypeStruct(kernelType, globalSymbol, suffix);
        this->codeStream_ << "\n#endif\n";
    };

    auto &manager = InfoManager::GetInstance();
    if (manager.GetShortSocVersion() == ShortSocVersion::ASCEND910B) {
        if (kernelType == KernelMetaType::KERNEL_TYPE_AIC_ONLY) {
            genKtypeWithArchMacro("__DAV_C220_CUBE__", "");
        } else if (kernelType == KernelMetaType::KERNEL_TYPE_AIV_ONLY) {
            genKtypeWithArchMacro("__DAV_C220_VEC__", "");
        } else if (kernelType == KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0) {
            genKtypeWithArchMacro("__DAV_C220_VEC__", "_mix_aiv");
        } else if (kernelType == KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0) {
            genKtypeWithArchMacro("__DAV_C220_CUBE__", "_mix_aic");
        } else {
            genKtypeWithArchMacro("__DAV_C220_CUBE__", "_mix_aic");
            genKtypeWithArchMacro("__DAV_C220_VEC__", "_mix_aiv");
        }
    } else if (manager.GetShortSocVersion() == ShortSocVersion::ASCEND310P) {
        if (kernelType == KernelMetaType::KERNEL_TYPE_AICORE) {
            genKtypeWithArchMacro("__DAV_M200__", "");
        }
    }
}

std::string AscDevMetaGenerator::GenCode()
{
    ASC_LOGI("Kernel [%s] : generate meta section.", kernelInfo_.kernelName.c_str());
    auto defaultKtype = ExtractKernelType(kernelType_);
    if (kernelInfo_.isTemplate) {
        for (const auto& inst : kernelInfo_.templateInstances) {
            GenMetaSection(inst.instanceMangledName.c_str(), GetBishengKTypeByCoreRatio(inst.ratio, defaultKtype));
        }
    } else {
        // when no template involved, always have 1 kernel type
        GenMetaSection(kernelInfo_.kernelMangledName.c_str(), defaultKtype);
    }
    ASC_LOGD(
            "Kernel [%s] : meta section is [\n%s\n]", kernelInfo_.kernelName.c_str(), codeStream_.str().c_str());
    return codeStream_.str();
}

} // namespace AscPlugin
