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
 * \file asc_dev_section_generate.h
 * \brief
 */

#ifndef __INCLUDE_INTERNAL_ASC_DEV_SECTION_GENERATE_H__
#define __INCLUDE_INTERNAL_ASC_DEV_SECTION_GENERATE_H__

#include <tuple>
#include <unordered_set>
#include "asc_struct.h"
#include "asc_utils.h"

namespace AscPlugin {
std::tuple<int32_t, std::string, std::string> GetDeviceCode(const KernelInfo& kernelInfo,
    const std::unordered_set<KernelMetaType>& kernelType, const KfcScene& kfcScene);
} // namespace AscPlugin
#endif // __INCLUDE_INTERNAL_ASC_DEV_SECTION_GENERATE_H__