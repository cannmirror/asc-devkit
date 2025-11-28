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
 * \file asc_host_stub_generate.cpp
 * \brief
 */

#include "asc_host_code_generate.h"

#include <string>
#include <unordered_set>
#include "asc_log.h"
#include "asc_struct.h"
#include "asc_host_binary_generator.h"
#include "asc_host_stub_generator.h"

namespace AscPlugin {
std::string GetBinaryRegisterCode()
{
    AscHostBinaryGenerator hostBinaryGenerator = AscHostBinaryGenerator();
    return hostBinaryGenerator.GenCode();
}

std::string GetHostStubCode(const KernelInfo& kernelInfo, const std::unordered_set<KernelMetaType>& kernelType)
{
    AscHostStubGenerator hostStubGenerator = AscHostStubGenerator(kernelInfo, kernelType);
    return hostStubGenerator.GenCode();
}
} // namespace AscPlugin