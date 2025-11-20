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
 * \file asc_dev_section_generate.cpp
 * \brief
 */

#include "asc_dev_section_generate.h"

#include <sstream>
#include <iostream>
#include <unordered_set>

#include "asc_log.h"
#include "asc_struct.h"
#include "asc_info_manager.h"
#include "asc_dev_stub_generator.h"
#include "asc_dev_meta_generator.h"

namespace AscPlugin {
std::string GetAscFeatureMetaSection(const uint32_t &featureFlag)
{
    std::string counterMacroDef =
        "#ifdef __ASC_FEATURE_META_INFO" + std::to_string(InfoManager::GetInstance().SetAndGetMetaFlagCounter());
    std::string ascMetaSecDef = R"(
namespace {
struct AscendCBinaryVersion {
	uint16_t type = 0;
	uint16_t len = 4;
	uint32_t version = 0;
};

struct AscendCFeatureFlag {
	uint16_t type = 4;
	uint16_t len = 8;
	uint32_t flag = 0;
	uint32_t resv = 0;
};

struct AscendCMetaInfo {
	AscendCBinaryVersion version;
	AscendCFeatureFlag feature;
};
}
)";
    std::string metaSection =
        std::string("static const struct AscendCMetaInfo __ascendc_meta_info__ __attribute__ ((used, section "
                    "(\".ascend.meta\"))) = {{0, 4, 0x01}, {4, 8, ") +
        std::to_string(featureFlag) + ", 0}};\n";
    return counterMacroDef + ascMetaSecDef + metaSection + "#endif\n";
}

std::tuple<int32_t, std::string, std::string> GetDeviceCode(const KernelInfo& kernelInfo,
    const std::unordered_set<KernelMetaType>& kernelType, const KfcScene& kfcScene)
{
    if (kernelInfo.kernelMangledNameConsiderPrefix.empty()) {
        ASC_LOGE("Kernel function info has no kernel mangling name consider prefix!");
        return {1, "", ""};
    }
    AscDevStubGenerator devStubGen = AscDevStubGenerator(kernelInfo, kernelType, kfcScene);
    std::string deviceStub = devStubGen.GenCode();
    AscDevMetaGenerator devMetaGen = AscDevMetaGenerator(kernelInfo, kernelType);
    std::string metaSection = devMetaGen.GenCode();
    metaSection.append(GetAscFeatureMetaSection(InfoManager::GetInstance().GetAscendMetaFlag()));

    return {0, deviceStub, metaSection};
}

} // namespace AscPlugin