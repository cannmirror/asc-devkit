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
 * \file kernel_operator_vec_bilinearinterpolation_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BILINEARINTERPOLATION_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BILINEARINTERPOLATION_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_operator_vec_template_impl.h"

namespace AscendC {
namespace Internal {
template <bool isMaskBitMode, typename T>
__simd_vf__ inline void BilinearInterpolationRepeatModeLevel0VFImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint32_t* src0Offset,
    __ubuf__ T* src1, const maskStruct maskArrayStruct, const uint64_t maskCount, const uint8_t hRepeat, const uint8_t vRepeat,
    uint16_t dstBlkStride, uint16_t vROffset, __ubuf__ uint64_t* maskBuf)
{
    uint32_t count = VecMicroGetCount<true, true, isMaskBitMode>(maskArrayStruct.maskArray, maskCount, maskBuf);
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<T>();;
    MicroAPI::MaskReg maskReg = VecMicroGetMaskReg<T, true, true, isMaskBitMode>(maskBuf, count);
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::RegTensor<T> srcReg0;
    MicroAPI::RegTensor<T> srcReg1;
    MicroAPI::RegTensor<T> tmpReg;
    MicroAPI::RegTensor<uint32_t> indexReg;

    for (uint16_t i = 0; i < static_cast<uint16_t>(vRepeat); ++i) {
        MicroAPI::Duplicate(dstReg, static_cast<T>(0), maskFull);
        for (uint16_t j = 0; j < static_cast<uint16_t>(hRepeat); ++j) {
            MicroAPI::DataCopy(indexReg, src0Offset + i * hRepeat * DEFAULT_REPEAT_STRIDE +
                                j * DEFAULT_REPEAT_STRIDE);
            MicroAPI::DataCopyGatherB(srcReg0, src0, indexReg, maskFull);
            
            MicroAPI::DataCopy<T,  MicroAPI::LoadDist::DIST_E2B_B16>(srcReg1, src1 +
                                i * hRepeat * DEFAULT_BLK_NUM + j * DEFAULT_BLK_NUM);

            MicroAPI::Mul(tmpReg, srcReg0, srcReg1, maskReg);
            MicroAPI::Add(dstReg, tmpReg, dstReg, maskReg);
        }
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + i * vROffset, dstReg, dstBlkStride, maskReg);
    }
}

template <bool isMaskBitMode, typename T>
__simd_vf__ inline void BilinearInterpolationNoRepeatModeLevel0VFImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint32_t* src0Offset,
    __ubuf__ T* src1, const maskStruct maskArrayStruct, const uint64_t maskCount, const uint8_t hRepeat, const uint8_t vRepeat,
    uint16_t dstBlkStride, uint16_t vROffset, __ubuf__ uint64_t* maskBuf)
{
    uint32_t count = VecMicroGetCount<true, true, isMaskBitMode>(maskArrayStruct.maskArray, maskCount, maskBuf);
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<T>();
    MicroAPI::MaskReg maskReg = VecMicroGetMaskReg<T, true, true, isMaskBitMode>(maskBuf, count);
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::RegTensor<T> srcReg0;
    MicroAPI::RegTensor<T> srcReg1;
    MicroAPI::RegTensor<T> tmpReg;
    MicroAPI::RegTensor<uint32_t> indexReg;

    for (uint16_t i = 0; i < static_cast<uint16_t>(vRepeat); ++i) {
        MicroAPI::Duplicate(dstReg, static_cast<T>(0), maskFull);
        for (uint16_t j = 0; j < static_cast<uint16_t>(hRepeat); ++j) {
            MicroAPI::DataCopy(indexReg, src0Offset + i * hRepeat * DEFAULT_REPEAT_STRIDE +
                                j * DEFAULT_REPEAT_STRIDE);
            MicroAPI::DataCopyGatherB(srcReg0, src0, indexReg, maskFull);
            
            MicroAPI::DataCopy<T,  MicroAPI::LoadDist::DIST_BRC_B16>(srcReg1, src1 + i * hRepeat + j);
            MicroAPI::Mul(tmpReg, srcReg0, srcReg1, maskReg);
            MicroAPI::Add(dstReg, tmpReg, dstReg, maskReg);
        }
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + i * vROffset, dstReg, dstBlkStride, maskReg);
    }
}
 
template <bool isMaskBitMode, typename T>
__aicore__ inline void VecBilinearInterpolationLevel0Template(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint32_t* src0Offset,
    __ubuf__ T* src1, const uint64_t maskArray[], const uint64_t maskCount, const uint8_t hRepeat, const uint8_t vRepeat,
    bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset)
{
    if constexpr (isMaskBitMode) {
        ASCENDC_ASSERT(maskCount == 0, "maskCount must be 0 when isMaskBitMode is true.");
    } else {
        ASCENDC_ASSERT(maskArray == nullptr, "maskArray must be nullptr when isMaskBitMode is false.");
    }

    uint16_t maskArraySize = (maskArray == nullptr) ? 0 : MASK_ARRAY_SIZE;
    maskStruct maskArrayStruct;
    for (uint16_t i = 0; i < maskArraySize; i++) {
        maskArrayStruct.maskArray[i] = maskArray[i];
    }

    if constexpr (isMaskBitMode) {
        SetVectorMask<T>(maskArray[1], maskArray[0]); // set mask to SPR.MASK, movp in VF
    }
    if (!repeatMode) {
        BilinearInterpolationNoRepeatModeLevel0VFImpl<isMaskBitMode, T>(dst, src0, src0Offset, src1,
            maskArrayStruct, maskCount, hRepeat, vRepeat, dstBlkStride, vROffset, nullptr);
    } else {
        BilinearInterpolationRepeatModeLevel0VFImpl<isMaskBitMode, T>(dst, src0, src0Offset, src1,
            maskArrayStruct, maskCount, hRepeat, vRepeat, dstBlkStride, vROffset, nullptr);
    }
}
} // namespace Internal

// BilinearInterpolation level-0 normal
template <typename T>
__aicore__ inline void BilinearInterpolationCalc(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,
    const LocalTensor<uint32_t> &src0OffsetLocal, const LocalTensor<T> &src1Local, uint64_t mask, uint8_t hRepeat,
    bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat,
    const LocalTensor<uint8_t> &sharedTmpBuffer)
{
    using PrimType = PrimT<T>;
    static_assert(SupportType<PrimType, half>(),
        "BilinearInterpolation api only support half on current device");

    __ubuf__ PrimType* dst = (__ubuf__ PrimType*)dstLocal.GetPhyAddr();
    __ubuf__ PrimType* src0 = (__ubuf__ PrimType*)src0Local.GetPhyAddr();
    __ubuf__ uint32_t* src0Offset = (__ubuf__ uint32_t*)src0OffsetLocal.GetPhyAddr();
    __ubuf__ PrimType* src1 = (__ubuf__ PrimType*)src1Local.GetPhyAddr();
    Internal::VecBilinearInterpolationLevel0Template<false>(dst, src0, src0Offset, src1, nullptr, mask,
                                                    hRepeat, vRepeat, repeatMode, dstBlkStride, vROffset);
}

// BilinearInterpolation level-0 bitwise
template <typename T>
__aicore__ inline void BilinearInterpolationCalc(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,
    const LocalTensor<uint32_t> &src0OffsetLocal, const LocalTensor<T> &src1Local, uint64_t mask[], uint8_t hRepeat,
    bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat,
    const LocalTensor<uint8_t> &sharedTmpBuffer)
{
    using PrimType = PrimT<T>;
    static_assert(SupportType<PrimType, half>(),
        "BilinearInterpolation api only support half on current device");

    __ubuf__ PrimType* dst = (__ubuf__ PrimType*)dstLocal.GetPhyAddr();
    __ubuf__ PrimType* src0 = (__ubuf__ PrimType*)src0Local.GetPhyAddr();
    __ubuf__ uint32_t* src0Offset = (__ubuf__ uint32_t*)src0OffsetLocal.GetPhyAddr();
    __ubuf__ PrimType* src1 = (__ubuf__ PrimType*)src1Local.GetPhyAddr();
    Internal::VecBilinearInterpolationLevel0Template<true>(dst, src0, src0Offset, src1, mask, 0,
                                                    hRepeat, vRepeat, repeatMode, dstBlkStride, vROffset);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BILINEARINTERPOLATION_IMPL_H