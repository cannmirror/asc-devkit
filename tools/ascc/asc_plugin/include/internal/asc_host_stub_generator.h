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
 * \file asc_host_stub_generator.h
 * \brief
 */

#ifndef __INCLUDE_INTERNAL_ASC_HOST_STUB_GENERATOR_H__
#define __INCLUDE_INTERNAL_ASC_HOST_STUB_GENERATOR_H__

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <algorithm>

#include "asc_utils.h"
#include "asc_struct.h"

namespace AscPlugin {
class AscHostStubGenerator {
public:
    AscHostStubGenerator(const KernelInfo& kernelInfo, const std::unordered_set<KernelMetaType>& kernelType);
    std::string GenCode();

private:
    std::string GenStubFuncDecl(bool hasNameSpace, bool hasAnonymousSpace) const;
    std::string ManglingNameJudgeCode();
    void GenStubFuncImpl();

private:
    KernelInfo kernelInfo_;
    std::ostringstream typeJudgePreCode_;
    std::ostringstream kernelCallStub_;
    std::unordered_set<KernelMetaType> kernelType_ = {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2};
};
} // namespace AscPlugin
#endif // __INCLUDE_INTERNAL_ASC_HOST_STUB_GENERATOR_H__