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
 * \file asc_dev_meta_generator.h
 * \brief
 */

#ifndef __INCLUDE_INTERNAL_ASC_DEV_META_GENERATOR_H__
#define __INCLUDE_INTERNAL_ASC_DEV_META_GENERATOR_H__

#include <string>
#include <sstream>
#include <unordered_set>
#include "asc_struct.h"
#include "asc_utils.h"

namespace AscPlugin {
class AscDevMetaGenerator {
public:
    AscDevMetaGenerator(const KernelInfo& kernelInfo, const std::unordered_set<KernelMetaType>& kernelType);
    std::string GenCode();

private:
    void GenMetaSection(const char* globalSymbol, const KernelMetaType& kernelType);
    void GenKtypeStruct(const KernelMetaType &kernelType, const char *globalSymbol, const char *kernelTypeSuffix);

    KernelInfo kernelInfo_;
    std::unordered_set<KernelMetaType> kernelType_ = {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2};
    std::stringstream codeStream_;
};


} // namespace AscPlugin
#endif // __INCLUDE_INTERNAL_ASC_DEV_META_GENERATOR_H__