/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file kernel_micro_common_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_MICRO_COMMON_INTERFACE_IMPL_H
#define ASCENDC_KERNEL_MICRO_COMMON_INTERFACE_IMPL_H

#include "kernel_tensor.h"

#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 2103) || \
    (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || (__NPU_ARCH__ == 3113) || defined(__ASC_NPU_HOST__)
#include "micro_api/kernel_micro_maskreg_intf.h"
#if __NPU_ARCH__ == 2103
#include "micro_api/dav_l210/kernel_micro_common_impl.h"
#elif __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_common_impl.h"
#elif __NPU_ARCH__ == 3103
#include "micro_api/dav_l310/kernel_micro_common_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_common_impl.h"
#else
#include "micro_api/dav_c310/kernel_micro_common_impl.h"
#endif
#endif

namespace AscendC {
namespace MicroAPI {
template <typename T, const RegTrait &regTrait>
__aicore__ void RegTensor<T, regTrait>::Print() const
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    std::ostringstream os;
    os.str("");
    uint32_t printLen;
    uint32_t blockNum;
    uint32_t rowNum;
    constexpr uint16_t b4RepElems = 512;
    constexpr uint16_t b4BlkElems = 64;
    if constexpr (std::is_same_v<T, fp4x2_e2m1_t> || std::is_same_v<T, fp4x2_e1m2_t>) {
        printLen = b4RepElems;
        blockNum = b4BlkElems;
        rowNum = printLen / blockNum;
    } else {
        printLen = VECTOR_REG_WIDTH / sizeof(T);
        blockNum = ONE_BLK_SIZE / sizeof(T);
        rowNum = printLen / blockNum;
    }
    const int32_t width = 4;
    for (int regIdx = 0; regIdx < regTrait.REG_NUM; regIdx++) {
        if constexpr (regTrait.REG_NUM == 2) {
            os << "reg[" << regIdx << "]:" << std::endl;
        }
        for (uint32_t i = 0; i < rowNum; i++) {
            os << std::setw(width) << std::setfill('0') << i * blockNum << " : ";
            for (uint32_t j = 0; j < blockNum; j++) {
                if constexpr ((sizeof(T) == sizeof(int8_t)) || (sizeof(T) == sizeof(bool))) {
                    os << static_cast<int32_t>(reg[regIdx][i * blockNum + j]) << " ";
                } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
                                    std::is_same_v<T, fp8_e4m3fn_t> || std::is_same_v<T, fp8_e5m2_t> ||
                                    std::is_same_v<T, hifloat8_t>) {
                    os << reg[regIdx][i * blockNum + j].ToFloat() << " ";
                } else if constexpr (std::is_same_v<T, fp4x2_e2m1_t> || std::is_same_v<T, fp4x2_e1m2_t>) {
                    os << ((bfloat16_t)reg[regIdx][i * blockNum + j]).ToFloat() << " ";
#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
                } else if constexpr (std::is_same_v<T, complex32>) {
                    const float real = (*(reinterpret_cast<const half*>(&reg[regIdx][i * blockNum + j]))).ToFloat();
                    const float image =
                        (*(reinterpret_cast<const half*>(&reg[regIdx][i * blockNum + j]) + 1)).ToFloat();
                    os << real << " " << image << " ";
                } else if (std::is_same_v<T, complex64>) {
                    const float real = *(reinterpret_cast<const float*>(&reg[regIdx][i * blockNum + j]));
                    const float image = *(reinterpret_cast<const float*>(&reg[regIdx][i * blockNum + j]) + 1);
                    os << real << " " << image << " ";
#endif
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    os << static_cast<int16_t>(reg[regIdx][i * blockNum + j].data & 0xf) << " "
                    << static_cast<int16_t>((reg[regIdx][i * blockNum + j].data >> 4) & 0xf) << " ";
                } else {
                    os << reg[regIdx][i * blockNum + j] << " ";
                }
            }
            os << std::endl;
        }
    }
    os << std::endl;
    std::cout << os.str();
#endif
}

template <MemType src, MemType dst>
__simd_callee__ inline void LocalMemBar()
{
    LocalMemBarImpl<src, dst>();
}
}  // namespace MicroAPI
}  // namespace AscendC
#endif  // ASCENDC_KERNEL_MICRO_COMMON_INTERFACE_IMPL_H