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
 * \file layernormgrad_c310_utils.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_LAYERNORMGRAD_REGBASE_C310_LAYERNORMGRAD_C310_UTILS_H_
#define IMPL_NORMALIZATION_LAYERNORMGRAD_REGBASE_C310_LAYERNORMGRAD_C310_UTILS_H_

#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "include/adv_api/normalization/layernormgrad_utils.h"

namespace AscendC {

namespace Internal {

namespace LayernormGrad {

constexpr MicroAPI::CastTrait castTraitHalfToFloat = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait castTraitFloatToHalf = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// load a float register from float or half UB
template <typename T>
__simd_callee__ inline void LoadDataWithT(
    __ubuf__ T* src, MicroAPI::RegTensor<float>& dstReg, MicroAPI::MaskReg& dstPreg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> srcTmpReg;
        DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcTmpReg, src + offset);
        Cast<float, T, castTraitHalfToFloat>(dstReg, srcTmpReg, dstPreg);
    } else { // this branch: only support float
        DataCopy(dstReg, src + offset);
    }
}

// fill a float register from float or half UB
template <typename T>
__simd_callee__ inline void FillDataWithT(
    __ubuf__ T* src, MicroAPI::RegTensor<float>& dstReg, MicroAPI::MaskReg& dstPreg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> srcTmpReg;
        DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(srcTmpReg, src + offset);
        Cast<float, T, castTraitHalfToFloat>(dstReg, srcTmpReg, dstPreg);
    } else { // this branch: only support float
        DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dstReg, src + offset);
    }
}

// store data from float register to float or half UB
template <typename T>
__simd_callee__ inline void StoreDataWithT(
    __ubuf__ T* dst, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& srcPreg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> dstTmpReg;
        // cast back to half
        MicroAPI::Cast<T, float, castTraitFloatToHalf>(dstTmpReg, srcReg, srcPreg);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
            (MicroAPI::RegTensor<uint16_t>&)dstTmpReg, (MicroAPI::RegTensor<uint32_t>&)dstTmpReg);
        MicroAPI::MaskPack(srcPreg, srcPreg);
        MicroAPI::DataCopy(dst + offset, dstTmpReg, srcPreg);
    } else { // this branch: only support float
        DataCopy(dst + offset, srcReg, srcPreg);
    }
}
} // namespace LayernormGrad
} // namespace Internal

struct LayerNormGradParams {
    __aicore__ LayerNormGradParams(
        uint32_t b, uint32_t s, uint32_t h, float lastDimValueBack, float lastDimValueBackMulTwo)
    {
        bLength = b;
        sLength = s;
        hLength = h;

        oneOverH = lastDimValueBack;
        twoOverH = lastDimValueBackMulTwo;
    }

    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;

    float oneOverH;
    float twoOverH;
};
} // namespace AscendC
#endif // IMPL_NORMALIZATION_LAYERNORMGRAD_REGBASE_C310_LAYERNORMGRAD_C310_UTILS_H_