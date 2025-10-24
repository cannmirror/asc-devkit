/**
 *Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
        
/*!
 * \file conv_arch.h
 * \brief
 */
           
#ifndef UTILS_CONV_ARCH_H
#define UTILS_CONV_ARCH_H

namespace Act {
namespace Conv {
namespace Arch {
struct Ascend910_95 {};
} // namespace Arch

constexpr static int64_t L0A_SIZE = 64 * 1024;
constexpr static int64_t L0B_SIZE = 64 * 1024;
constexpr static int64_t L0C_SIZE = 256 * 1024;
constexpr static int64_t L1_SIZE = 512 * 1024;

} // namespace Conv
} // namespace Act
#endif