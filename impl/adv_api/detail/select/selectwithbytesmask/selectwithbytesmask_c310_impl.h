/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIB_SELECT_SELECT_WITH_BYTES_MASK_C310_IMPL_H
#define LIB_SELECT_SELECT_WITH_BYTES_MASK_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_utils.h"

namespace AscendC {
template <typename T, typename U, CMPMODE cmpMode>
__aicore__ inline void RegTensorToMaskReg(MicroAPI::RegTensor<U> &vMaskReg0, MicroAPI::RegTensor<U> &vMaskReg1,
    MicroAPI::MaskReg &localMask0, MicroAPI::MaskReg &maskReg0)
{
    MicroAPI::MaskReg maskReg1;
    MicroAPI::MaskReg localMask1;
    if constexpr (sizeof(U) == 1) {
        MicroAPI::CompareScalar<uint8_t, cmpMode>(localMask0, (MicroAPI::RegTensor<uint8_t> &)vMaskReg0,
            static_cast<uint8_t>(0), maskReg0);
    } else if constexpr (sizeof(T) == 2 && sizeof(U) == 4) {
        MicroAPI::MaskUnPack(maskReg1, maskReg0);
        MicroAPI::CompareScalar<U, cmpMode>(localMask0, vMaskReg0, static_cast<U>(0), maskReg1);
        MicroAPI::CompareScalar<U, cmpMode>(localMask1, vMaskReg1, static_cast<U>(0), maskReg1);
        MicroAPI::MaskDeInterleave<T>(localMask0, localMask1, localMask0, localMask1);
    } else {
        MicroAPI::CompareScalar<U, cmpMode>(localMask0, vMaskReg0, static_cast<U>(0), maskReg0);
    }
}

template <typename T, typename U, bool reverse = false>
__aicore__ inline void SelectWithBytesMaskPerAxisImpl(__local_mem__ T *dstUb, __local_mem__ T *src0Ub, T src1,
    __local_mem__ U *maskUb, const uint32_t firstAxis, const uint32_t srcLastAxis, const uint32_t maskLastAxis)
{
    MicroAPI::RegTensor<T> vSrcReg0;
    MicroAPI::RegTensor<T> vSrcReg1;
    MicroAPI::RegTensor<T> vDstReg;
    MicroAPI::RegTensor<U> vMaskReg0;
    MicroAPI::RegTensor<U> vMaskReg1;
    MicroAPI::MaskReg maskReg0;
    MicroAPI::MaskReg localMask0;
    MicroAPI::Duplicate(vSrcReg1, src1);
    uint32_t sreg;
    uint32_t sregLower = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint16_t repeatTimes = static_cast<uint16_t>(DivCeil(srcLastAxis, sregLower));
    for (uint16_t loopH = 0; loopH < static_cast<uint16_t>(firstAxis); ++loopH) {
        sreg = srcLastAxis;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            maskReg0 = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy<T>(vSrcReg0, src0Ub + loopH * srcLastAxis + i * sregLower);
            if constexpr (sizeof(T) == 2 && sizeof(U) == 1) {
                MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                    (MicroAPI::RegTensor<uint8_t> &)vMaskReg0,
                    (__local_mem__ uint8_t *)maskUb + loopH * maskLastAxis + i * sregLower);
            } else if constexpr (sizeof(T) == 2 && sizeof(U) == 4) {
                MicroAPI::DataCopy<U>(vMaskReg0, maskUb + loopH * maskLastAxis + i * sregLower);
                MicroAPI::DataCopy<U>(vMaskReg1, maskUb + loopH * maskLastAxis + i * sregLower + sregLower / 2);
            } else if constexpr (sizeof(T) == 4 && sizeof(U) == 1) {
                MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    (MicroAPI::RegTensor<uint8_t> &)vMaskReg0,
                    (__local_mem__ uint8_t *)maskUb + loopH * maskLastAxis + i * sregLower);
            } else if constexpr (sizeof(T) == 4 && sizeof(U) == 2) {
                MicroAPI::DataCopy<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(vMaskReg0,
                    maskUb + loopH * maskLastAxis + i * sregLower);
            } else if constexpr (sizeof(T) == sizeof(U)) {
                MicroAPI::DataCopy<U>(vMaskReg0, maskUb + loopH * maskLastAxis + i * sregLower);
            }

            if constexpr (!reverse) {
                RegTensorToMaskReg<T, U, CMPMODE::EQ>(vMaskReg0, vMaskReg1, localMask0, maskReg0);
            } else {
                RegTensorToMaskReg<T, U, CMPMODE::NE>(vMaskReg0, vMaskReg1, localMask0, maskReg0);
            }

            MicroAPI::Select(vDstReg, vSrcReg0, vSrcReg1, localMask0);
            MicroAPI::DataCopy<T>(dstUb + loopH * srcLastAxis + i * sregLower, vDstReg, maskReg0);
        }
    }
}

template <typename T, typename U, bool reverse = false>
__aicore__ inline void SelectWithBytesMaskProcess(const LocalTensor<T> &dst, const LocalTensor<T> &src0, T src1,
    const LocalTensor<U> &mask, const SelectWithBytesMaskShapeInfo &info)
{
    __local_mem__ T *src0Ub = (__local_mem__ T *)src0.GetPhyAddr();
    __local_mem__ T *dstUb = (__local_mem__ T *)dst.GetPhyAddr();
    __local_mem__ U *maskUb = (__local_mem__ U *)mask.GetPhyAddr();
    const uint32_t firstAxis = static_cast<uint32_t>(info.firstAxis);
    const uint32_t srcLastAxis = static_cast<uint32_t>(info.srcLastAxis);
    const uint32_t maskLastAxis = static_cast<uint32_t>(info.maskLastAxis);
    VF_CALL<SelectWithBytesMaskPerAxisImpl<T, U, reverse>>(dstUb, src0Ub, src1, maskUb, firstAxis, srcLastAxis,
        maskLastAxis);
}

// Selects Values from two sources and put into dst according to the mask values.
// True: Select scalar, False: select src.
template <typename T, typename U, bool isReuseMask, bool reverse = false>
__aicore__ inline __inout_pipe__(V) void SelectWithBytesMaskImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
    T src1, const LocalTensor<U> &mask, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const SelectWithBytesMaskShapeInfo &info)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, float, half>(), "SelectWithBytesMask do not support this type on current device");
    static_assert(SupportType<U, bool, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "SelectWithBytesMask do not support this type on current device");
    CheckTensorPos<T>(dst, Hardware::UB, "dst", "VECIN / VECCALC / VECOUT", "SelectWithBytesMask");
    CheckTensorPos<T>(src0, Hardware::UB, "src", "VECIN / VECCALC / VECOUT", "SelectWithBytesMask");
    CheckTensorPos<U>(mask, Hardware::UB, "mask", "VECIN / VECCALC / VECOUT", "SelectWithBytesMask");
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT",
        "SelectWithBytesMask");
    ASCENDC_ASSERT((info.srcLastAxis * sizeof(T) % ONE_BLK_SIZE == 0), {
        KERNEL_LOG(KERNEL_ERROR, "srcLastAxis should be 32B aligned, current srcLastAxis is %u", info.srcLastAxis);
    });
    ASCENDC_ASSERT((info.maskLastAxis * sizeof(U) % ONE_BLK_SIZE == 0), {
        KERNEL_LOG(KERNEL_ERROR, "maskLastAxis should be 32B aligned, current maskLastAxis is %u", info.maskLastAxis);
    });
    ASCENDC_ASSERT((info.maskLastAxis % BLOCK_CUBE == 0), {
        KERNEL_LOG(KERNEL_ERROR, "maskLastAxis should be multiples of 16, current maskLastAxis is %u",
            info.maskLastAxis);
    });

    const uint32_t firstAxis = info.firstAxis;
    const uint32_t srcLastAxis = info.srcLastAxis;
    const uint32_t maskLastAxis = info.maskLastAxis;
    const uint32_t srcSize = src0.GetSize();

    ASCENDC_ASSERT((srcSize == firstAxis * srcLastAxis),
                   { KERNEL_LOG(KERNEL_ERROR, "ShapeInfo must be match with src Tensor size."); });
    ASCENDC_ASSERT((mask.GetSize() == firstAxis * maskLastAxis),
                   { KERNEL_LOG(KERNEL_ERROR, "ShapeInfo must be match with mask Tensor size."); });
    ASCENDC_ASSERT((maskLastAxis >= srcLastAxis),
                   { KERNEL_LOG(KERNEL_ERROR, "maskLastAxis must be greater than or equal to srcLastAxis."); });

    SelectWithBytesMaskProcess<T, U, reverse>(dst, src0, src1, mask, info);
}
} // namespace AscendC
#endif // LIB_SELECT_SELECT_WITH_BYTES_MASK_C310_IMPL_H
