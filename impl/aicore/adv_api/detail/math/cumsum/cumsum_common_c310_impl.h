/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cumsum_common_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATH_CUMSUM_CUMSUM_COMMON_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_MATH_CUMSUM_CUMSUM_COMMON_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "math/cumsum_utils.h"
namespace AscendC {

namespace Internal {
template <typename T>
__aicore__ inline void LoadDataWithT(
    __local_mem__ T* src, MicroAPI::RegTensor<float>& dstReg, MicroAPI::MaskReg& dstPreg, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<T> srcOrigin;
        DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcOrigin, src + srcOffset);
        Cast<float, T, layoutZMrgZ>(dstReg, srcOrigin, dstPreg);
    } else { // this branch: only support float
        DataCopy(dstReg, src + srcOffset);
    }
}

template <typename T>
__aicore__ inline void SaveDataWithT(
    __local_mem__ T* dst, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& dstPreg, uint32_t dstOffset)
{
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<T> regT;
        Cast<T, float, LayoutZMrgZRndRSatNS>(regT, srcReg, dstPreg);
        DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + dstOffset, regT, dstPreg);
    } else {
        DataCopy(dst + dstOffset, srcReg, dstPreg);
    }
}

// process by tempBuffer
// T: fp16-> U: fp32
// T: fp32-> U: fp16
template <typename U, typename T>
__aicore__ inline void CumSumCopyWithCast(
    const LocalTensor<U>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t outter, const uint32_t inner)
{
    __local_mem__ T* src = (__local_mem__ T*)srcTensor.GetPhyAddr();
    __local_mem__ U* dst = (__local_mem__ U*)dstTensor.GetPhyAddr();
    constexpr uint16_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(float));
    uint32_t count = inner;
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, sregLower));
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> srcReg;
        MicroAPI::MaskReg preg;
        for (uint16_t j = 0; j < static_cast<uint16_t>(outter); j++) {
            count = inner;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = MicroAPI::UpdateMask<float>(count);
                LoadDataWithT<T>(src, srcReg, preg, j * inner + i * sregLower);
                SaveDataWithT<U>(dst, srcReg, preg, j * inner + i * sregLower);
            }
        }
    }
}

template <typename T>
__aicore__ inline void CumSumCopyOut(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t dstOutter, const uint32_t srcInner)
{
    __local_mem__ T* src = (__local_mem__ T*)srcTensor.GetPhyAddr();
    __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
    constexpr uint16_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    uint16_t repeatTimes = CeilDivision(srcInner, sregLower);
    uint32_t count = srcInner;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::MaskReg preg;
        for (uint16_t j = 0; j < static_cast<uint16_t>(dstOutter); j++) {
            count = srcInner;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = MicroAPI::UpdateMask<T>(count);
                MicroAPI::DataCopy(srcReg, src + j * srcInner + i * sregLower);
                MicroAPI::DataCopy(dst + j * srcInner + i * sregLower, srcReg, preg);
            }
        }
    }
}

__aicore__ inline void CumSumFirstDimBinary(const LocalTensor<float>& dstTensor, uint32_t outter, uint32_t inner)
{
    constexpr uint32_t bound = 16;
    constexpr uint32_t halfSize = 2;
    uint32_t outterAlign = 0;
    for (uint32_t i = 0; i < bound; i++) {
        if (outter <= (1U << i)) {
            outterAlign = (1U << i);
            break;
        }
    }
    uint32_t round = outterAlign / halfSize;

    uint32_t currRound = 1;
    __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
    constexpr uint16_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(float));
    uint16_t repeatTimes = CeilDivision(inner, sregLower);

    while (round >= 1) {
        uint32_t currRound1 = 1 << (currRound - 1);
        uint32_t currRound2 = 1 << currRound;
        uint16_t indexRepeatTimes = static_cast<uint16_t>(outterAlign / currRound2);
        uint16_t jRepeatTimes = static_cast<uint16_t>(currRound1);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> src0Reg;
            MicroAPI::RegTensor<float> src1Reg;
            MicroAPI::RegTensor<float> dstReg;
            MicroAPI::MaskReg preg;

            for (uint16_t index = 0; index < indexRepeatTimes; index++) {
                // Position of the prefix sum in the previous round
                uint32_t line0 = currRound1 - 1 + index * currRound2;
                for (uint16_t j = 0; j < jRepeatTimes; j++) {
                    uint32_t line1 = line0 + j + 1;
                    uint32_t extent = 1;
                    if (line1 > outter - 1) {
                        extent = 0;
                    }
                    for (uint16_t k = 0; k < static_cast<uint16_t>(extent); k++) {
                        uint32_t count = inner;
                        for (uint16_t i = 0; i < repeatTimes; i++) {
                            preg = MicroAPI::UpdateMask<float>(count);
                            MicroAPI::DataCopy(src0Reg, dst + line0 * inner + i * sregLower);
                            MicroAPI::DataCopy(src1Reg, dst + line1 * inner + i * sregLower);
                            MicroAPI::Add(dstReg, src0Reg, src1Reg, preg);
                            MicroAPI::DataCopy(dst + line1 * inner + i * sregLower, dstReg, preg);
                        }
                    }
                }
            }
        }
        round = round / halfSize;
        currRound += 1;
    }
}

template <typename D, typename T, const MicroAPI::RegTrait& Trait, const uint16_t vlSize>
__aicore__ inline void TransposeCommonGather(__local_mem__ D* dstAddr, __local_mem__ T* srcAddr, uint32_t forLoop1,
    uint32_t forLoop2, uint32_t srcStride1, uint32_t srcStride2)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename D = float, typename T = float, const MicroAPI::RegTrait& Trait, const uint16_t vlSize>
__aicore__ inline void TransposeCommonGather(__local_mem__ float* dstAddr, __local_mem__ float* srcAddr,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride1, uint32_t srcStride2)
{
    uint32_t tail = forLoop2 % vlSize;
    uint32_t count = tail;
    uint16_t mainLoop = forLoop2 / vlSize;
    uint32_t dtypeSize = sizeof(float);
    uint32_t tailLoop = tail > 0 ? 1 : 0;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t, Trait> indexReg;
        MicroAPI::RegTensor<T, Trait> srcReg;
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL, Trait>();
        MicroAPI::MaskReg indexFullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL, Trait>();
        MicroAPI::MaskReg mainMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL, Trait>();
        MicroAPI::MaskReg tailMask = MicroAPI::UpdateMask<float, Trait>(count);
        MicroAPI::UnalignReg ureg0;
        Arange((MicroAPI::RegTensor<int32_t, Trait>&)indexReg, static_cast<int32_t>(0));
        Muls(indexReg, indexReg, srcStride2, indexFullMask);
        for (uint16_t j = 0; j < static_cast<uint16_t>(forLoop1); j++) {
            uint64_t hoistDstAddr = (uint64_t)dstAddr + (uint64_t)(j * forLoop2 * dtypeSize);
            for (uint16_t k = 0; k < static_cast<uint16_t>(mainLoop); k++) {
                DataCopyGather(srcReg, srcAddr + j * srcStride1 + k * vlSize * srcStride2, indexReg, mainMask);
                MicroAPI::DataCopyUnAlign(((__local_mem__ float*&)hoistDstAddr), srcReg, ureg0, vlSize);
            }
            for (uint16_t k = 0; k < static_cast<uint16_t>(tailLoop); k++) {
                DataCopyGather(srcReg, srcAddr + j * srcStride1 + mainLoop * vlSize * srcStride2, indexReg, tailMask);
                MicroAPI::DataCopyUnAlign(((__local_mem__ float*&)hoistDstAddr), srcReg, ureg0, tail);
            }
            MicroAPI::DataCopyUnAlignPost(((__local_mem__ float*&)hoistDstAddr), ureg0, 0);
        }
    }
}

template <typename D = float, typename T = half, const MicroAPI::RegTrait& Trait, const uint16_t vlSize>
__aicore__ inline void TransposeCommonGather(__local_mem__ float* dstAddr, __local_mem__ half* srcAddr,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride1, uint32_t srcStride2)
{
    uint32_t tail = forLoop2 % vlSize;
    uint32_t count = tail;
    uint16_t mainLoop = forLoop2 / vlSize;
    uint32_t dtypeSize = sizeof(float);
    uint32_t tailLoop = tail > 0 ? 1 : 0;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t, Trait> indexReg;
        MicroAPI::RegTensor<half, Trait> srcReg;
        MicroAPI::RegTensor<float, Trait> vreg;
        MicroAPI::RegTensor<uint16_t> zeroReg;
        MicroAPI::RegTensor<half> castReg;
        MicroAPI::RegTensor<uint16_t> tmpReg;
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL, Trait>();
        MicroAPI::Duplicate(zeroReg, static_cast<uint16_t>(0), fullMask);
        MicroAPI::MaskReg indexFullMask = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL, Trait>();
        MicroAPI::MaskReg mainMask = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL, Trait>();
        MicroAPI::MaskReg tailMask = MicroAPI::UpdateMask<half, Trait>(count);
        MicroAPI::UnalignReg ureg0;
        Arange((MicroAPI::RegTensor<int16_t, Trait>&)indexReg, static_cast<int16_t>(0));
        Muls(indexReg, indexReg, static_cast<uint16_t>(srcStride2), indexFullMask);
        for (uint16_t j = 0; j < static_cast<uint16_t>(forLoop1); j++) {
            uint64_t hoistDstAddr = (uint64_t)dstAddr + (uint64_t)(j * forLoop2 * dtypeSize);
            for (uint16_t k = 0; k < static_cast<uint16_t>(mainLoop); k++) {
                DataCopyGather(srcReg, srcAddr + j * srcStride1 + k * vlSize * srcStride2, indexReg, mainMask);
                MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t>&)castReg, (MicroAPI::RegTensor<uint16_t>&)tmpReg,
                    (MicroAPI::RegTensor<uint16_t>&)srcReg, (MicroAPI::RegTensor<uint16_t>&)zeroReg);
                Cast<float, half, layoutZMrgZ>(vreg, castReg, mainMask);
                MicroAPI::DataCopyUnAlign(((__local_mem__ float*&)hoistDstAddr), vreg, ureg0, vlSize);
            }
            for (uint16_t k = 0; k < static_cast<uint16_t>(tailLoop); k++) {
                DataCopyGather(srcReg, srcAddr + j * srcStride1 + mainLoop * vlSize * srcStride2, indexReg, tailMask);
                MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t>&)castReg, (MicroAPI::RegTensor<uint16_t>&)tmpReg,
                    (MicroAPI::RegTensor<uint16_t>&)srcReg, (MicroAPI::RegTensor<uint16_t>&)zeroReg);
                Cast<float, half, layoutZMrgZ>(vreg, castReg, mainMask);
                MicroAPI::DataCopyUnAlign(((__local_mem__ float*&)hoistDstAddr), vreg, ureg0, tail);
            }
            MicroAPI::DataCopyUnAlignPost(((__local_mem__ float*&)hoistDstAddr), ureg0, 0);
        }
    }
}

template <typename D = half, typename T = float, const MicroAPI::RegTrait& Trait, const uint16_t vlSize>
__aicore__ inline void TransposeCommonGather(__local_mem__ half* dstAddr, __local_mem__ float* srcAddr,
    uint32_t forLoop1, uint32_t forLoop2, uint32_t srcStride1, uint32_t srcStride2)
{
    uint32_t tail = forLoop2 % vlSize;
    uint32_t count = tail;
    uint16_t mainLoop = forLoop2 / vlSize;
    uint32_t dtypeSize = sizeof(half);
    uint32_t tailLoop = tail > 0 ? 1 : 0;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t, Trait> indexReg;
        MicroAPI::RegTensor<float, Trait> srcReg;
        MicroAPI::RegTensor<half, Trait> vreg;
        MicroAPI::RegTensor<uint16_t> zeroReg;
        MicroAPI::RegTensor<half> castReg;
        MicroAPI::RegTensor<uint16_t> tmpReg;
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL, Trait>();
        MicroAPI::Duplicate(zeroReg, static_cast<uint16_t>(0), fullMask);
        MicroAPI::MaskReg indexFullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL, Trait>();
        MicroAPI::MaskReg mainMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL, Trait>();
        MicroAPI::MaskReg tailMask = MicroAPI::UpdateMask<float, Trait>(count);
        MicroAPI::UnalignReg ureg0;
        Arange((MicroAPI::RegTensor<int32_t, Trait>&)indexReg, static_cast<int32_t>(0));
        Muls(indexReg, indexReg, srcStride2, indexFullMask);
        for (uint16_t j = 0; j < static_cast<uint16_t>(forLoop1); j++) {
            uint64_t hoistDstAddr = (uint64_t)dstAddr + (uint64_t)(j * forLoop2 * dtypeSize);
            for (uint16_t k = 0; k < static_cast<uint16_t>(mainLoop); k++) {
                DataCopyGather(srcReg, srcAddr + j * srcStride1 + k * vlSize * srcStride2, indexReg, mainMask);
                Cast<half, float, LayoutZMrgZRndRSatNS>(vreg, srcReg, fullMask);
                MicroAPI::DeInterleave((MicroAPI::RegTensor<uint16_t>&)castReg, (MicroAPI::RegTensor<uint16_t>&)tmpReg,
                    (MicroAPI::RegTensor<uint16_t>&)vreg, (MicroAPI::RegTensor<uint16_t>&)zeroReg);
                MicroAPI::DataCopyUnAlign(((__local_mem__ half*&)hoistDstAddr), castReg, ureg0, vlSize);
            }
            for (uint16_t k = 0; k < static_cast<uint16_t>(tailLoop); k++) {
                DataCopyGather(srcReg, srcAddr + j * srcStride1 + mainLoop * vlSize * srcStride2, indexReg, tailMask);
                Cast<half, float, LayoutZMrgZRndRSatNS>(vreg, srcReg, fullMask);
                MicroAPI::DeInterleave((MicroAPI::RegTensor<uint16_t>&)castReg, (MicroAPI::RegTensor<uint16_t>&)tmpReg,
                    (MicroAPI::RegTensor<uint16_t>&)vreg, (MicroAPI::RegTensor<uint16_t>&)zeroReg);
                MicroAPI::DataCopyUnAlign(((__local_mem__ half*&)hoistDstAddr), castReg, ureg0, tail);
            }
            MicroAPI::DataCopyUnAlignPost(((__local_mem__ half*&)hoistDstAddr), ureg0, 0);
        }
    }
}

/*
scene: { shape:[A, B], format:"ND"} -->{ shape:[B, A], format:"ND"};
Src: T
Dst: D
1. need cast
TransposeAB [A, B] half => [B, A] float
TransposeAB [A, B] float => [B, A] half
2. no need cast
TransposeAB [A, B] float => [B, A] float
*/
template <typename D, typename T>
__aicore__ inline void TransposeAB(
    const LocalTensor<D>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t outter, uint32_t inner)
{
    uint32_t srcStride1 = 1;
    uint32_t srcStride2 = inner;
    constexpr uint16_t vlSize = VECTOR_REG_WIDTH / sizeof(float);
    TransposeCommonGather<D, T, MicroAPI::RegTraitNumOne, vlSize>((__local_mem__ D*)dstTensor.GetPhyAddr(),
        (__local_mem__ T*)srcTensor.GetPhyAddr(), inner, outter, srcStride1, srcStride2);
}
} // namespace Internal

template <typename T>
__aicore__ inline void CumSumCopyLastRow(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t len)
{
    __local_mem__ T* src = (__local_mem__ T*)srcTensor.GetPhyAddr();
    __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
    constexpr uint16_t sregLower = static_cast<uint16_t>(VECTOR_REG_WIDTH / sizeof(T));
    uint32_t count = len;
    uint16_t repeatTimes = CeilDivision(count, sregLower);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::MaskReg preg;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = MicroAPI::UpdateMask<T>(count);
            MicroAPI::DataCopy(srcReg, src + i * sregLower);
            MicroAPI::DataCopy(dst + i * sregLower, srcReg, preg);
        }
    }
}

template <typename T>
__aicore__ inline void CumSumLastDim(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    LocalTensor<T> tempBuffer, const CumSumInfo& cumSumInfo)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = float>
__aicore__ inline void CumSumLastDim(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    LocalTensor<float> tempBuffer, const CumSumInfo& cumSumInfo)
{
    uint16_t alignOutter =
        (cumSumInfo.outter + NCHW_CONV_ADDR_LIST_SIZE - 1) / NCHW_CONV_ADDR_LIST_SIZE * NCHW_CONV_ADDR_LIST_SIZE;
    const uint32_t minTmpBufferSize = alignOutter * cumSumInfo.inner * 2 * sizeof(float); // 2: tempBuffer + tempBuffer2
    const uint32_t tmpBufferSize = tempBuffer.GetSize() * sizeof(float);
    LocalTensor<float> tempBuffer2 = tempBuffer[alignOutter * cumSumInfo.inner];
#if ASCENDC_CPU_DEBUG
    ASCENDC_ASSERT((tmpBufferSize >= minTmpBufferSize), {
        KERNEL_LOG(KERNEL_ERROR,
            "tmpBufferSize can't smaller than minTmpBufferSize, tmpBufferSize is %u, minTmpBufferSize is %u!",
            tmpBufferSize, minTmpBufferSize);
    });
#endif
    Internal::TransposeAB(tempBuffer, srcTensor, alignOutter, cumSumInfo.inner);
    Internal::CumSumFirstDimBinary(tempBuffer, cumSumInfo.inner, alignOutter);
    Internal::TransposeAB(tempBuffer2, tempBuffer, cumSumInfo.inner, alignOutter);
    Internal::CumSumCopyOut(dstTensor, tempBuffer2, cumSumInfo.outter, cumSumInfo.inner);
}

template <typename T = half>
__aicore__ inline void CumSumLastDim(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor,
    LocalTensor<half> tempBufferHalf, const CumSumInfo& cumSumInfo)
{
    uint16_t alignOutter =
        (cumSumInfo.outter + NCHW_CONV_ADDR_LIST_SIZE - 1) / NCHW_CONV_ADDR_LIST_SIZE * NCHW_CONV_ADDR_LIST_SIZE;

    const uint32_t minTmpBufferSize = alignOutter * cumSumInfo.inner * 2 * sizeof(half); // 2: tempBuffer + tempBuffer2
    LocalTensor<float> tempBuffer = tempBufferHalf.ReinterpretCast<float>();
    const uint32_t tmpBufferSize = tempBuffer.GetSize() * sizeof(float);
    LocalTensor<float> tempBuffer2 = tempBuffer[alignOutter * cumSumInfo.inner];
    LocalTensor<half> tempBuffer2Half = tempBuffer2.ReinterpretCast<half>();
#if ASCENDC_CPU_DEBUG
    ASCENDC_ASSERT((tmpBufferSize >= minTmpBufferSize), {
        KERNEL_LOG(KERNEL_ERROR,
            "tmpBufferSize can't smaller than minTmpBufferSize, tmpBufferSize is %u, minTmpBufferSize is %u!",
            tmpBufferSize, minTmpBufferSize);
    });
#endif
    Internal::TransposeAB(tempBuffer, srcTensor, alignOutter, cumSumInfo.inner);
    Internal::CumSumFirstDimBinary(tempBuffer, cumSumInfo.inner, alignOutter);
    Internal::TransposeAB(tempBuffer2Half, tempBuffer, cumSumInfo.inner, alignOutter);
    Internal::CumSumCopyOut(dstTensor, tempBuffer2Half, cumSumInfo.outter, cumSumInfo.inner);
}

template <typename T>
__aicore__ inline void CumSumFirstDim(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    LocalTensor<uint8_t>& sharedTmpBuffer, const CumSumInfo& cumSumInfo)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = float>
__aicore__ inline void CumSumFirstDim(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    LocalTensor<uint8_t>& sharedTmpBuffer, const CumSumInfo& cumSumInfo)
{
    Internal::CumSumCopyOut(dstTensor, srcTensor, cumSumInfo.outter, cumSumInfo.inner);
    Internal::CumSumFirstDimBinary(dstTensor, cumSumInfo.outter, cumSumInfo.inner);
}

template <typename T = half>
__aicore__ inline void CumSumFirstDim(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor,
    LocalTensor<uint8_t>& sharedTmpBuffer, const CumSumInfo& cumSumInfo)
{
    const uint32_t minTmpBufferSize = cumSumInfo.outter * cumSumInfo.inner * sizeof(float);
    const uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
#if ASCENDC_CPU_DEBUG
    ASCENDC_ASSERT((tmpBufferSize >= minTmpBufferSize), {
        KERNEL_LOG(KERNEL_ERROR,
            "tmpBufferSize can't smaller than minTmpBufferSize, tmpBufferSize is %u, minTmpBufferSize is %u!",
            tmpBufferSize, minTmpBufferSize);
    });
#endif
    LocalTensor<float> tmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    Internal::CumSumCopyWithCast(tmpBuffer, srcTensor, cumSumInfo.outter, cumSumInfo.inner);
    Internal::CumSumFirstDimBinary(tmpBuffer, cumSumInfo.outter, cumSumInfo.inner);
    Internal::CumSumCopyWithCast(dstTensor, tmpBuffer, cumSumInfo.outter, cumSumInfo.inner);
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_MATH_CUMSUM_CUMSUM_COMMON_C310_IMPL_H
