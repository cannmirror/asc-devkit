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
 * \file layernormgradbeta_c310_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_LAYERNORMGRAD_LAYERNORMGRADBETA_C310_IMPL_H
#define IMPL_NORMALIZATION_LAYERNORMGRAD_LAYERNORMGRADBETA_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
namespace LayerNormGradBetaAPI {

constexpr MicroAPI::CastTrait castTraitF16F32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait castTraitF32F16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
__simd_callee__ inline void LoadSrcData(MicroAPI::RegTensor<float>& srcReg, __ubuf__ T* src0, uint16_t index,
                                   MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> srcTmpReg;
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    if constexpr (std::is_same<T, half>::value) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcTmpReg, src0 + index * eleCountPerVL);
        MicroAPI::Cast<float, T, castTraitF16F32>(srcReg, srcTmpReg, mask);
    } else {
        MicroAPI::DataCopy(srcReg, src0 + index * eleCountPerVL);
    }
}

template <typename T>
__simd_callee__ inline void StoreDstData(__ubuf__ T* dst, MicroAPI::RegTensor<float>& dstReg, uint16_t index,
                                    MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> dstTmpReg;
    MicroAPI::MaskReg tmpMask = mask;
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);

    if constexpr (std::is_same<T, half>::value) {
        MicroAPI::Cast<T, float, castTraitF32F16>(dstTmpReg, dstReg, tmpMask);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t>&)dstTmpReg,
                                                                          (MicroAPI::RegTensor<uint32_t>&)dstTmpReg);
        MicroAPI::MaskPack(tmpMask, tmpMask);
        MicroAPI::DataCopy(dst + index * eleCountPerVL, dstTmpReg, tmpMask);
    } else {
        MicroAPI::DataCopy(dst + index * eleCountPerVL, dstReg, tmpMask);
    }
}

template <typename T>
__simd_vf__ inline void ReduceSumN(__ubuf__ T* outputPdGamma, __ubuf__ T* outputPdBeta, __ubuf__ T* resForGamma,
                                  __ubuf__ T* inputDy, const uint32_t bsLength, const uint32_t hLength)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> inputDyReg, resForGammaReg;
    MicroAPI::RegTensor<float> outputPdGammaReg, outputPdBetaReg;
    MicroAPI::RegTensor<float> tmpReg;

    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = DivCeil(hLength, eleCountPerVL);
    uint32_t count = hLength;

    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = MicroAPI::UpdateMask<float>(count);
        MicroAPI::Duplicate(outputPdGammaReg, 0.0f, mask);
        MicroAPI::Duplicate(outputPdBetaReg, 0.0f, mask);
        for (uint16_t j = 0; j < bsLength; j++) {
            LoadSrcData(inputDyReg, inputDy + j * hLength, i, mask);
            LoadSrcData(resForGammaReg, resForGamma + j * hLength, i, mask);

            MicroAPI::Mul(tmpReg, inputDyReg, resForGammaReg, mask);
            MicroAPI::Add(outputPdGammaReg, outputPdGammaReg, tmpReg, mask);
            MicroAPI::Add(outputPdBetaReg, outputPdBetaReg, inputDyReg, mask);
        }
        StoreDstData(outputPdGamma, outputPdGammaReg, i, mask);
        StoreDstData(outputPdBeta, outputPdBetaReg, i, mask);
    }
}
} // namespace LayerNormGradBetaAPI

template <typename T, bool isReuseSource = false>
__aicore__ inline void
LayerNormGradBetaCheckParams(const LocalTensor<T>& outputPdGamma, const LocalTensor<T>& outputPdBeta,
                             const LocalTensor<T>& resForGamma, const LocalTensor<T>& inputDy,
                             const LocalTensor<uint8_t>& sharedTmpBuffer, const LayerNormGradBetaTiling& tiling)
{
    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    CheckTensorPos<T>(outputPdGamma, Hardware::UB, "outputPdGamma", "VECIN / VECCALC / VECOUT", "LayerNormGradBeta");
    CheckTensorPos<T>(outputPdBeta, Hardware::UB, "outputPdBeta", "VECIN / VECCALC / VECOUT", "LayerNormGradBeta");
    CheckTensorPos<T>(resForGamma, Hardware::UB, "resForGamma", "VECIN / VECCALC / VECOUT", "LayerNormGradBeta");
    CheckTensorPos<T>(inputDy, Hardware::UB, "inputDy", "VECIN / VECCALC / VECOUT", "LayerNormGradBeta");
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT",
                            "LayerNormGradBeta");

    constexpr uint32_t alignLen = 32;
    ASCENDC_ASSERT((tiling.hLength * sizeof(T) % alignLen == 0), {
        KERNEL_LOG(KERNEL_ERROR, "The value of hLength * sizeof(T) must be an integer multiple of 32.");
    });
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void LayerNormGradBetaImpl(const LocalTensor<T>& outputPdGamma, const LocalTensor<T>& outputPdBeta,
                                             const LocalTensor<T>& resForGamma, const LocalTensor<T>& inputDy,
                                             const LocalTensor<uint8_t>& sharedTmpBuffer,
                                             const LayerNormGradBetaTiling& tiling)
{
    LayerNormGradBetaCheckParams<T, isReuseSource>(outputPdGamma, outputPdBeta, resForGamma, inputDy, sharedTmpBuffer,
                                                   tiling);
    __ubuf__ T* outputPdGammaDst = (__ubuf__ T*)outputPdGamma.GetPhyAddr();
    __ubuf__ T* outputPdBetaDst = (__ubuf__ T*)outputPdBeta.GetPhyAddr();
    __ubuf__ T* resForGammaSrc = (__ubuf__ T*)resForGamma.GetPhyAddr();
    __ubuf__ T* inputDySrc = (__ubuf__ T*)inputDy.GetPhyAddr();

    LayerNormGradBetaAPI::ReduceSumN<T>(outputPdGammaDst, outputPdBetaDst, resForGammaSrc, inputDySrc,
                                                 tiling.bsLength, tiling.hLength);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void LayerNormGradBetaImpl(const LocalTensor<T>& outputPdGamma, const LocalTensor<T>& outputPdBeta,
                                             const LocalTensor<T>& resForGamma, const LocalTensor<T>& inputDy,
                                             LayerNormGradBetaTiling& tiling)
{
    LocalTensor<uint8_t> sharedTmpBuffer; // partial derivation
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    LayerNormGradBetaImpl<T, isReuseSource>(outputPdGamma, outputPdBeta, resForGamma, inputDy, sharedTmpBuffer, tiling);
}
} // namespace AscendC
#endif // IMPL_NORMALIZATION_LAYERNORMGRAD_LAYERNORMGRADBETA_C310_IMPL_H
