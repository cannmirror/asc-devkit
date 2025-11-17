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
 * \file kernel_operator_vec_scatter_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_SCATTER_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_SCATTER_IMPL_H


namespace AscendC {
/* **************************************************************************************************
 * scatter                                             *
 * ************************************************************************************************* */

// Scatter::Level 0 - mask bit mode
template <typename T>
__aicore__ inline void ScatterImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ uint32_t* dstOffsetLocal,
    const uint32_t dstLength, const uint32_t dstBaseAddr, const uint64_t mask[], const uint8_t repeatTime,
    const uint8_t srcRepStride)
{
    uint32_t offsetAddr = (uint64_t)dstLocal + dstBaseAddr;
#if ASCENDC_CPU_DEBUG
    uint64_t cpuAddr = (uint64_t)dstLocal + dstBaseAddr;
    SetModelScatterDst0Tensor(cpuAddr, dstLength);
#endif
    uint64_t config = 0;
    config |= (uint64_t(offsetAddr & 0xFFFFFFFF));
    config |= (uint64_t(repeatTime & 0xFF) << 56);
    config |= (uint64_t(srcRepStride & 0xFF) << 40);

    AscendCUtils::SetMask<T>(mask[1], mask[0]);
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        vscatter(dstOffsetLocal, (__ubuf__ uint16_t*)srcLocal, config);
    } else if constexpr (sizeof(T) == sizeof(uint32_t)) {
        vscatter(dstOffsetLocal, (__ubuf__ uint32_t*)srcLocal, config);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "data type should be b16 or b32"); });
    }
}

// Scatter::Level 0 - mask count mode
template <typename T>
__aicore__ inline void ScatterImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ uint32_t* dstOffsetLocal,
    const uint32_t dstLength, const uint32_t dstBaseAddr, const uint64_t mask, const uint8_t repeatTime,
    const uint8_t srcRepStride)
{
    uint32_t offsetAddr = (uint64_t)dstLocal + dstBaseAddr;
#if ASCENDC_CPU_DEBUG
    uint64_t cpuAddr = (uint64_t)dstLocal + dstBaseAddr;
    SetModelScatterDst0Tensor(cpuAddr, dstLength);
#endif
    uint64_t config = 0;
    config |= (uint64_t(offsetAddr & 0xFFFFFFFF));   // dstLocalBaseAddr, In isa is Xt[31:0]
    config |= (uint64_t(repeatTime & 0xFF) << 56);  // repeat times, Xt[63:56]
    config |= (uint64_t(srcRepStride & 0xFF) << 40); // src repeat stride size, Xt[47:40]

    AscendCUtils::SetMask<T>(mask);
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        vscatter(dstOffsetLocal, (__ubuf__ uint16_t*)srcLocal, config);
    } else if constexpr (sizeof(T) == sizeof(uint32_t)) {
        vscatter(dstOffsetLocal, (__ubuf__ uint32_t*)srcLocal, config);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "data type should be b16 or b32"); });
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_SCATTER_IMPL_H