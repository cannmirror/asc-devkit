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
 * \file kernel_operator_vec_gather_impl.h
 * \brief AscendC l310 support vector gather api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
#include "kernel_operator_common_impl.h"
namespace AscendC {

/* **************************************************************************************************
 * Gather                                                                                           *
 * **************************************************************************************************/
// gatherb::Level 0
template <typename T>
__aicore__ inline void GatherImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint32_t* offset,
    const uint32_t srcLength, uint8_t repeatTime, const GatherRepeatParams& repeatParams)
{
    uint32_t repeatStride = VECTOR_REG_WIDTH / ONE_BLK_SIZE;
    __VEC_SCOPE__
    {
        uint16_t dstRptStd = repeatParams.dstRepStride;
        uint8_t dstBlkStd = repeatParams.dstBlkStride;
        RegTensor<T> vDst;
        RegTensor<uint32_t> vregIndex;
        UnalignReg ureg;
        uint32_t sreg = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        MaskReg preg = CreatePredicate<T>(sreg);
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            AddrReg indexOffset = CreateAddrReg<uint32_t>(repeatStride);
            DataCopyUnAlignPre<uint32_t>(ureg, offset, indexOffset);
            DataCopyUnAlign<uint32_t>(vregIndex, ureg, offset, indexOffset, repeatStride);
            DataCopyGatherB(vDst, src0, vregIndex, preg);
            DataCopy(dst, vDst, dstBlkStd, i * dstRptStd, preg);
        }
    }
}

// for gather op
#define GATHER_OP_B8_MASK_COUNT_MODE(T, U)                                                                        \
    uint64_t mask1;                                                                                               \
    uint64_t mask2;                                                                                               \
    if (mask > ELE_CNT_B16) {                                                                                     \
        mask1 = ELE_CNT_B16;                                                                                      \
        mask2 = mask - ELE_CNT_B16;                                                                               \
    } else {                                                                                                      \
        mask1 = mask;                                                                                             \
        mask2 = 0;                                                                                                \
    }                                                                                                             \
    __VEC_SCOPE__                                                                                                 \
    {                                                                                                             \
        RegTensor<uint16_t> vreg0;                                                                                \
        RegTensor<uint16_t> vreg1;                                                                                \
        RegTensor<uint16_t> vregEven;                                                                             \
        RegTensor<uint16_t> vregOdd;                                                                              \
        RegTensor<uint16_t> vregIndex0;                                                                           \
        RegTensor<uint16_t> vregIndex1;                                                                           \
        RegTensor<uint8_t> vregOut;                                                                               \
        RegTensor<uint8_t> vregOut0;                                                                              \
        RegTensor<uint8_t> vregOut1;                                                                              \
        uint32_t sreg = (uint32_t)mask;                                                                           \
        MaskReg preg = CreatePredicate<T>(sreg);                                                                  \
        uint32_t sreg1 = (uint32_t)mask1;                                                                         \
        MaskReg preg1 = CreatePredicate<uint16_t>(sreg1);                                                         \
        uint32_t sreg2 = (uint32_t)mask2;                                                                         \
        MaskReg preg2 = CreatePredicate<uint16_t>(sreg2);                                                         \
        MaskReg cast_preg = CreatePredicate<uint16_t>();                                                          \
        for (uint16_t i = 0; i <= (uint16_t)repeatTime; ++i)                                                     \
        {                                                                                                         \
            AddrReg vgather_offset = CreateAddrReg<uint16_t>(ELE_CNT_B8);                                         \
            DataCopy<uint16_t, Dist::DIST_NORM>(vregIndex0, (__ubuf__ uint16_t *)srcOffset, vgather_offset); \
            DataCopy<uint16_t, Dist::DIST_NORM>(vregIndex1, (__ubuf__ uint16_t *)srcOffset + ELE_CNT_B16,    \
                                                vgather_offset);                                                  \
            DataCopyGather(vreg0, src + srcBaseAddr / sizeof(uint8_t), vregIndex0, preg1);                   \
            DataCopyGather(vreg1, src + srcBaseAddr / sizeof(uint8_t), vregIndex1, preg2);                   \
            DeInterleave<uint16_t>(vregEven, vregOdd, vreg0, vreg1);                                              \
            Cast<uint8_t, uint16_t, Mode::ZEROING, SatMode::SAT, PartMode::EVEN>(vregOut0, vregEven, cast_preg);  \
            Cast<uint8_t, uint16_t, Mode::ZEROING, SatMode::SAT, PartMode::ODD>(vregOut1, vregOdd, cast_preg);    \
            Or<uint8_t>(vregOut, vregOut0, vregOut1, preg);                                                       \
            if constexpr (std::is_same_v<T, uint8_t>) {                                                           \
                DataCopy(dst, vregOut, 1, i * dstRepStride, preg);                                           \
            } else {                                                                                              \
                DataCopy((__ubuf__ uint8_t*)dst, vregOut, 1, i * dstRepStride, preg);                        \
            }                                                                                                     \
        }                                                                                                         \
    }

#define GATHER_OP_B16B32_MASK_COUNT_MODE(T, U)                                                      \
    __VEC_SCOPE__                                                                                   \
    {                                                                                               \
        RegTensor<uint32_t> index_src0Reg;                                                          \
        RegTensor<uint32_t> index_src1Reg;                                                          \
        RegTensor<uint32_t> index_dst0Reg;                                                          \
        RegTensor<uint32_t> index_dst1Reg;                                                          \
        RegTensor<uint16_t> indexReg;                                                               \
        RegTensor<uint16_t> indexReg0;                                                              \
        RegTensor<uint16_t> indexReg1;                                                              \
        MaskReg index_preg = CreatePredicate<U>();                                                  \
        RegTensor<T> vDst;                                                                          \
        uint32_t sreg = (uint32_t)mask;                                                             \
        MaskReg preg = CreatePredicate<T>(sreg);                                                    \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                              \
        if constexpr (sizeof(T) == sizeof(uint16_t) && sizeof(U) == sizeof(uint32_t))               \
        {                                                                                           \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i)                                    \
            {                                                                                       \
                AddrReg indexOffset = CreateAddrReg<uint32_t>(VECTOR_REG_WIDTH / B16_BYTE_SIZE);    \
                DataCopy<uint32_t, Dist::DIST_NORM>(index_src0Reg, srcOffset, indexOffset);    \
                DataCopy<uint32_t, Dist::DIST_NORM>(index_src1Reg,                                  \
                    srcOffset + (VECTOR_REG_WIDTH / B32_BYTE_SIZE), indexOffset);              \
                DeInterleave<uint32_t>(index_dst0Reg, index_dst1Reg, index_src0Reg, index_src1Reg); \
                Cast<uint16_t, uint32_t, Mode::ZEROING, SatMode::SAT, PartMode::EVEN>(indexReg0,    \
                    index_dst0Reg, index_preg);                                                     \
                Cast<uint16_t, uint32_t, Mode::ZEROING, SatMode::SAT, PartMode::ODD>(indexReg1,     \
                    index_dst1Reg, index_preg);                                                     \
                Or<uint16_t>(indexReg, indexReg0, indexReg1, preg);                                 \
                ShiftRights<uint16_t, uint16_t>(indexReg, indexReg, sizeof(T) / 2, preg);           \
                DataCopyGather(vDst, src + srcBaseAddr / sizeof(T), indexReg, preg);           \
                DataCopy(dst, vDst, 1, i * dstRepStride, preg);                                \
            }                                                                                       \
        } else {                                                                                    \
            RegTensor<U> vregIndex;                                                                 \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i)                                    \
            {                                                                                       \
                DataCopy(vregIndex, srcOffset, i * sregLower);                                 \
                ShiftRights<U, U>(vregIndex, vregIndex, sizeof(T) / 2, preg);                       \
                DataCopyGather(vDst, src + srcBaseAddr / sizeof(T), vregIndex, preg);          \
                DataCopy(dst, vDst, 1, i * dstRepStride, preg);                                \
            }                                                                                       \
        }                                                                                           \
    }

// gather::Level 0 - mask count mode
template <typename T, typename U = uint32_t>
__aicore__ inline void GatherImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ U* srcOffset,
    const uint32_t srcLength, const uint32_t srcBaseAddr, const uint64_t mask, const uint8_t repeatTime,
    const uint16_t dstRepStride)
{
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        GATHER_OP_B8_MASK_COUNT_MODE(T, U);
    } else if constexpr (sizeof(T) == sizeof(uint16_t) || sizeof(T) == sizeof(uint32_t)) {
        GATHER_OP_B16B32_MASK_COUNT_MODE(T, U);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "data type should be b8 b16 or b32"); });
    }
}

#define GATHER_OP_B16_MASK_BIT_MODE(T, U)                                                           \
    SetVectorMask<uint16_t>(mask[1], mask[0]);                                                      \
    __VEC_SCOPE__                                                                                   \
    {                                                                                               \
        RegTensor<uint32_t> index_src0Reg;                                                          \
        RegTensor<uint32_t> index_src1Reg;                                                          \
        RegTensor<uint32_t> index_dst0Reg;                                                          \
        RegTensor<uint32_t> index_dst1Reg;                                                          \
        RegTensor<uint16_t> indexReg;                                                               \
        RegTensor<uint16_t> indexReg0;                                                              \
        RegTensor<uint16_t> indexReg1;                                                              \
        RegTensor<T> vDst;                                                                          \
        MaskReg preg = movp_b16();                                                                  \
        MaskReg index_preg = CreatePredicate<U>();                                                  \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                              \
        if constexpr (sizeof(T) == sizeof(uint16_t) && sizeof(U) == sizeof(uint32_t)) {             \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {                                  \
                AddrReg indexOffset = CreateAddrReg<uint32_t>(VECTOR_REG_WIDTH / B16_BYTE_SIZE);    \
                DataCopy<uint32_t, Dist::DIST_NORM>(index_src0Reg, srcOffset, indexOffset);    \
                DataCopy<uint32_t, Dist::DIST_NORM>(index_src1Reg,                                  \
                    srcOffset + (VECTOR_REG_WIDTH / B32_BYTE_SIZE), indexOffset);              \
                DeInterleave<uint32_t>(index_dst0Reg, index_dst1Reg, index_src0Reg, index_src1Reg); \
                Cast<uint16_t, uint32_t, Mode::ZEROING, SatMode::SAT, PartMode::EVEN>(indexReg0,    \
                    index_dst0Reg, index_preg);                                                     \
                Cast<uint16_t, uint32_t, Mode::ZEROING, SatMode::SAT, PartMode::ODD>(indexReg1,     \
                    index_dst1Reg, index_preg);                                                     \
                Or<uint16_t>(indexReg, indexReg0, indexReg1, preg);                                 \
                ShiftRights<uint16_t, uint16_t>(indexReg, indexReg, sizeof(T) / 2, preg);           \
                DataCopyGather(vDst, src + srcBaseAddr / sizeof(T), indexReg, preg);           \
                DataCopy(dst, vDst, 1, i * dstRepStride, preg);                                \
            }                                                                                       \
        } else {                                                                                    \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {                                  \
                DataCopy(indexReg, srcOffset, i * sregLower);                                  \
                ShiftRights<uint16_t, uint16_t>(indexReg, indexReg, sizeof(T) / 2, preg);           \
                DataCopyGather(vDst, src + srcBaseAddr / sizeof(T), indexReg, preg);           \
                DataCopy(dst, vDst, 1, i * dstRepStride, preg);                                \
            }                                                                                       \
        }                                                                                           \
    }

#define GATHER_OP_B32_MASK_BIT_MODE(T, U)                                                    \
    SetVectorMask<uint32_t>(mask[1], mask[0]);                                               \
    __VEC_SCOPE__                                                                            \
    {                                                                                        \
        RegTensor<T> vDst;                                                                   \
        RegTensor<U> vregIndex;                                                              \
        MaskReg preg = movp_b32();                                                           \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                       \
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {                               \
            DataCopy(vregIndex, srcOffset, i * sregLower);                              \
            ShiftRights<U, U>(vregIndex, vregIndex, sizeof(T) / 2, preg);                    \
            DataCopyGather(vDst, src + srcBaseAddr / sizeof(T), vregIndex, preg);       \
            DataCopy(dst, vDst, 1, i * dstRepStride, preg);                             \
        }                                                                                    \
    }

// gather::Level 0 - mask bit mode
template <typename T, typename U = uint32_t>
__aicore__ inline void GatherImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ U* srcOffset,
    const uint32_t srcLength, const uint32_t srcBaseAddr, const uint64_t mask[2], const uint8_t repeatTime,
    const uint16_t dstRepStride)
{
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        GATHER_OP_B16_MASK_BIT_MODE(T, U);
    } else if constexpr (sizeof(T) == sizeof(uint32_t)) {
        GATHER_OP_B32_MASK_BIT_MODE(T, U);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "data type should be b16 or b32"); });
    }
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
