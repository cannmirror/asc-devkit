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

    return {0, deviceStub, metaSection};
}

} // namespace AscPlugin