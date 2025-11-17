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
#include "kernel_operator_common_impl.h"

namespace AscendC {
/* **************************************************************************************************
 * scatter                                             *
 * ************************************************************************************************* */

// Scatter::Level 0 - mask bit mode
template <typename T>
typename std::enable_if<(sizeof(T) == 2)>::type __aicore__ inline ScatterImpl(__ubuf__ T* dst, __ubuf__ T* src,
    __ubuf__ uint32_t* dstOffset, const uint32_t dstLength, const uint32_t dstBaseAddr, const uint64_t mask[2], const uint8_t repeatTime,
    const uint8_t srcRepStride)
{
    uint64_t offsetAddr = (uint64_t)dst + dstBaseAddr;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    SetVectorMask<uint16_t>(mask[1], mask[0]);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<uint32_t> index_src0Reg;
        RegTensor<uint32_t> index_src1Reg;
        RegTensor<uint16_t> index_dst0Reg;
        RegTensor<uint16_t> index_dst1Reg;
        RegTensor<uint16_t> indexReg;
        MaskReg preg = movp_b16();
        MaskReg index_preg = CreatePredicate<T>();
        for (uint16_t i = 0; i < repeatTime; i++)
        {
            AddrReg srcOffset = CreateAddrReg<T>(srcRepStride * B16_DATA_NUM_PER_BLOCK);
            AddrReg indexOffset = CreateAddrReg<uint32_t>(VECTOR_REG_WIDTH / B16_BYTE_SIZE);
            DataCopy<T>(srcReg, src, srcOffset);
            DataCopy<uint32_t>(index_src0Reg, dstOffset, indexOffset);
            DataCopy<uint32_t>(index_src1Reg, dstOffset + (VECTOR_REG_WIDTH / B32_BYTE_SIZE), indexOffset);
            DeInterleave<uint32_t>(index_src0Reg, index_src1Reg, index_src0Reg, index_src1Reg);
            Cast<uint16_t, uint32_t, Mode::ZEROING, SatMode::SAT, PartMode::EVEN>(index_dst0Reg, index_src0Reg, preg);
            Cast<uint16_t, uint32_t, Mode::ZEROING, SatMode::SAT, PartMode::ODD>(index_dst1Reg, index_src1Reg, preg);
            Or<uint16_t>(indexReg, index_dst0Reg, index_dst1Reg, index_preg);
            ShiftRights<uint16_t, uint16_t>(indexReg, indexReg, sizeof(T) / 2, preg);
            DataCopyScatter<T, uint16_t>((__ubuf__ T*)offsetAddr, srcReg, indexReg, preg);
        }
    }
}

template <typename T>
typename std::enable_if<(sizeof(T) == 4)>::type __aicore__ inline ScatterImpl(__ubuf__ T* dst, __ubuf__ T* src,
    __ubuf__ uint32_t* dstOffset, const uint32_t dstLength, const uint32_t dstBaseAddr, const uint64_t mask[2], const uint8_t repeatTime,
    const uint8_t srcRepStride)
{
    uint64_t offsetAddr = (uint64_t)dst + dstBaseAddr;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    SetVectorMask<uint32_t>(mask[1], mask[0]);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<uint32_t> dstOffsetReg;
        RegTensor<uint32_t> indexReg;
        MaskReg preg = movp_b32();
        for (uint16_t i = 0; i < repeatTime; i++)
        {
            AddrReg offset = CreateAddrReg<T>(srcRepStride * B32_DATA_NUM_PER_BLOCK);
            DataCopy<T>(srcReg, src, offset);
            DataCopy<uint32_t>(dstOffsetReg, dstOffset, offset);
            ShiftRights<uint32_t, uint32_t>(indexReg, dstOffsetReg, sizeof(T) / 2, preg);
            DataCopyScatter<T, uint32_t>((__ubuf__ T*)offsetAddr, srcReg, indexReg, preg);
        }
    }
}

// Scatter::Level 0 - mask count mode
template <typename T>
typename std::enable_if<(sizeof(T) == 2)>::type __aicore__ inline ScatterImpl(__ubuf__ T* dst, __ubuf__ T* src,
__ubuf__ uint32_t* dstOffset, const uint32_t dstLength, const uint32_t dstBaseAddr, const uint64_t mask, const uint8_t repeatTime,
const uint8_t srcRepStride)
{
    uint64_t offsetAddr = (uint64_t)dst + dstBaseAddr;
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<uint32_t> index_src0Reg;
        RegTensor<uint32_t> index_src1Reg;
        RegTensor<uint16_t> index_dst0Reg;
        RegTensor<uint16_t> index_dst1Reg;
        RegTensor<uint16_t> indexReg;
        uint32_t cnt = mask;
        MaskReg preg = CreatePredicate<T>(cnt);
        MaskReg index_preg = CreatePredicate<T>();
        for (uint16_t i = 0; i < repeatTime; i++)
        {
            AddrReg srcOffset = CreateAddrReg<T>(srcRepStride * B16_DATA_NUM_PER_BLOCK);
            AddrReg indexOffset = CreateAddrReg<uint32_t>(VECTOR_REG_WIDTH / B16_BYTE_SIZE);
            DataCopy<T>(srcReg, src, srcOffset);
            DataCopy<uint32_t>(index_src0Reg, dstOffset, indexOffset);
            DataCopy<uint32_t>(index_src1Reg, dstOffset + (VECTOR_REG_WIDTH / B32_BYTE_SIZE), indexOffset);
            DeInterleave<uint32_t>(index_src0Reg, index_src1Reg, index_src0Reg, index_src1Reg);
            Cast<uint16_t, uint32_t, Mode::ZEROING, SatMode::SAT, PartMode::EVEN>(index_dst0Reg, index_src0Reg, preg);
            Cast<uint16_t, uint32_t, Mode::ZEROING, SatMode::SAT, PartMode::ODD>(index_dst1Reg, index_src1Reg, preg);
            Or<uint16_t>(indexReg, index_dst0Reg, index_dst1Reg, index_preg);
            ShiftRights<uint16_t, uint16_t>(indexReg, indexReg, sizeof(T) / 2, preg);
            DataCopyScatter<T, uint16_t>((__ubuf__ T*)offsetAddr, srcReg, indexReg, preg);
        }
    }
}

template <typename T>
typename std::enable_if<(sizeof(T) == 4)>::type __aicore__ inline ScatterImpl(__ubuf__ T* dst, __ubuf__ T* src,
    __ubuf__ uint32_t* dstOffset, const uint32_t dstLength, const uint32_t dstBaseAddr, const uint64_t mask, const uint8_t repeatTime,
    const uint8_t srcRepStride)
{
    uint64_t offsetAddr = (uint64_t)dst + dstBaseAddr;
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<uint32_t> dstOffsetReg;
        RegTensor<uint32_t> indexReg;
        uint32_t cnt = mask;
        MaskReg preg = CreatePredicate<T>(cnt);
        for (uint16_t i = 0; i < repeatTime; i++)
        {
            AddrReg offset = CreateAddrReg<T>(srcRepStride * B32_DATA_NUM_PER_BLOCK);
            DataCopy<T>(srcReg, src, offset);
            DataCopy<uint32_t>(dstOffsetReg, dstOffset, offset);
            ShiftRights<uint32_t, uint32_t>(indexReg, dstOffsetReg, sizeof(T) / 2, preg);
            DataCopyScatter<T, uint32_t>((__ubuf__ T*)offsetAddr, srcReg, indexReg, preg);
        }
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_SCATTER_IMPL_H