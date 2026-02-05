/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file kernel_micro_struct_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_STRUCT_INTERFACE_H
#define ASCENDC_MODULE_MICRO_STRUCT_INTERFACE_H

#if (__NPU_ARCH__ == 3101) || defined(__ASC_NPU_HOST__)
#include "micro_api/dav_c310/kernel_micro_datatype_impl.h"
#elif __NPU_ARCH__ == 5102
#include "micro_api/dav_m510/kernel_micro_datatype_impl.h"
#elif __NPU_ARCH__ == 2103
#include "micro_api/dav_l210/kernel_micro_datatype_impl.h"
#elif __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_datatype_impl.h"
#elif __NPU_ARCH__ == 3103
#include "micro_api/dav_l310/kernel_micro_datatype_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_datatype_impl.h"
#endif

namespace AscendC {
namespace MicroAPI {

struct RegTrait {
    int REG_NUM = 1;
};

constexpr RegTrait RegTraitNumOne = {1};
constexpr RegTrait RegTraitNumTwo = {2};

template <typename T, const RegTrait& regTrait = RegTraitNumOne>
struct RegTensor {
    __simd_callee__ inline RegTensor(){};
    using ActualT = T;
    static constexpr RegTrait trait = regTrait;
    static constexpr int REG_NUM = trait.REG_NUM;
    using RegType = typename TypeGet<T>::T;
    RegType reg[trait.REG_NUM];

    __simd_callee__ inline operator RegType& ()
    {
        // only process one reg, two registers require explicit call
        return reg[0];
    }
    __simd_callee__ void Print() const;
};

} // namespace MicroAPI
} // namespace AscendC

#include "impl/micro_api/kernel_micro_struct_intf_impl.h"
#endif // ASCENDC_MODULE_MICRO_STRUCT_INTERFACE_H