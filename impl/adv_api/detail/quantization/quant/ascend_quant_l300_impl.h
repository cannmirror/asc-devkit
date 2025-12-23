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
 * \file ascend_quant_l300_impl.h
 * \brief
 */
#ifndef LIB_ASCEND_QUANT_ASCEND_QUANT_L300_IMPL_H
#define LIB_ASCEND_QUANT_ASCEND_QUANT_L300_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "include/adv_api/quantization/ascend_quant_utils.h"

namespace AscendC {
constexpr uint32_t ASCENDC_QUANT_B16_VF_LEN = GetVecLen() / sizeof(uint16_t);
constexpr uint32_t ASCENDC_QUANT_B32_VF_LEN = GetVecLen() / sizeof(uint32_t);

enum class AscendQuantPolicy : int32_t {
    PER_TENSOR,
    PER_CHANNEL,
    PER_TOKEN,
    PER_GROUP,
    PER_CHANNEL_PER_GROUP,
    PER_TOKEN_PER_GROUP
};

struct AscendQuantParam {
  uint32_t m;
  uint32_t n;
  uint32_t calCount;
  uint32_t groupSize = 0;
};

template <typename dstT, typename srcT>
__simd_vf__ inline void QuantPertensorForB8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    const float scale, const float offset, const uint32_t calCount)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<half> f16Vreg;
    MicroAPI::RegTensor<dstT> s8vreg;

    uint32_t sregLower = (uint32_t)ASCENDC_QUANT_B16_VF_LEN;
    uint32_t sreg = (uint32_t)calCount;
    uint16_t repeat = CeilDivision(calCount, sregLower);

    for (uint16_t i = 0; i < (uint16_t)repeat; ++i) {
        preg = MicroAPI::UpdateMask<uint16_t>(sreg);

        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(f16Vreg, srcUb + i * sregLower);

        MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, static_cast<half>(scale),
            preg);
        MicroAPI::Adds<half, half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, static_cast<half>(offset),
            preg);
        if constexpr (SupportType<dstT, int8_t>()) {
            MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(s8vreg, f16Vreg, preg);
        } else {
            MicroAPI::Cast<dstT, half, LayoutZMrgZRndASatS>(s8vreg, f16Vreg, preg);
        }
        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstUb + i * sregLower, s8vreg, preg);
    }
}

/* **************************************************************************************************
 * pertensor process for int8/hif8 output                                             *
 * ************************************************************************************************* */
template <typename dstT, typename srcT>
__aicore__ inline void QuantPertensorForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const float scale, const float offset, const uint32_t calCount)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    QuantPertensorForB8VF<dstT, srcT>(dstUb, srcUb, scale, offset, calCount);
}

template <typename dstT, typename srcT>
__simd_vf__ inline void QuantPertensorForB8VF(__ubuf__ dstT* dstUb, __ubuf__ float* srcUb,
    const float scale, const float offset, const uint32_t calCount)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<half> f16Vreg;
    MicroAPI::RegTensor<dstT> s8vreg;

    uint32_t sregLower = (uint32_t)ASCENDC_QUANT_B32_VF_LEN;
    uint32_t sreg = (uint32_t)calCount;
    uint16_t repeat = CeilDivision(calCount, sregLower);

    for (uint16_t i = 0; i < (uint16_t)repeat; ++i) {
        preg = MicroAPI::UpdateMask<uint32_t>(sreg);
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(f32vreg, srcUb + i * sregLower);
        MicroAPI::Cast<half, float, LayoutZMrgZRndRSatS>(f16Vreg, f32vreg, preg);

        MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, static_cast<half>(scale),
            preg);
        MicroAPI::Adds<half, half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, static_cast<half>(offset),
            preg);

        if constexpr (SupportType<dstT, int8_t>()) {
            MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(s8vreg, f16Vreg, preg);
        } else {
            MicroAPI::Cast<dstT, half, LayoutZMrgZRndASatS>(s8vreg, f16Vreg, preg);
        }
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>
            ((MicroAPI::RegTensor<uint16_t> &)s8vreg, (MicroAPI::RegTensor<uint32_t> &)s8vreg);
        MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(preg, preg);
        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstUb + i * sregLower, s8vreg, preg);
    }
}

template <typename dstT, typename srcT>
__aicore__ inline void QuantPertensorForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<float>& srcTensor,
    const float scale, const float offset, const uint32_t calCount)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ float* srcUb = (__ubuf__ float*)srcTensor.GetPhyAddr();
    QuantPertensorForB8VF<dstT, srcT>(dstUb, srcUb, scale, offset, calCount);
}

template <typename T, bool isReuseSource = false, const AscendQuantConfig &config>
__aicore__ inline void AscendQuantImpl(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    static_assert(SupportType<T, half, float>(),
        "This AscendQuant only support half/float input dtype");
    
    const uint32_t calCountReal = config.calcCount != 0 ? config.calcCount : calCount;
    ASCENDC_ASSERT((calCountReal <= srcTensor.GetSize() && calCountReal <= dstTensor.GetSize() && calCountReal >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]",
            calCountReal, srcTensor.GetSize(), dstTensor.GetSize());
    });
    QuantPertensorForB8<int8_t, T>(dstTensor, srcTensor, scale, offset, calCountReal);
}

template <typename dstT, typename srcT>
__simd_vf__ inline void QuantPertensorForFp8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    const float scale, const float offset, const uint32_t calCount)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<srcT> b16vreg;
    MicroAPI::RegTensor<dstT> b8vreg;

    uint32_t sregLower = (uint32_t)ASCENDC_QUANT_B32_VF_LEN;
    uint32_t sreg = (uint32_t)calCount;
    uint16_t repeat = CeilDivision(calCount, sregLower);

    for (uint16_t i = 0; i < (uint16_t)repeat; ++i) {
        preg = MicroAPI::UpdateMask<uint32_t>(sreg);
        if constexpr (SupportType<srcT, half>()) {
            MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(b16vreg, srcUb + i * sregLower);
            MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32vreg, b16vreg, preg);
        } else {
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(f32vreg, srcUb + i * sregLower);
        }

        MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, static_cast<float>(scale),
            preg);
        MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, static_cast<float>(offset),
            preg);

        MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(b8vreg, f32vreg, preg);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>
            ((MicroAPI::RegTensor<uint16_t> &)b8vreg, (MicroAPI::RegTensor<uint32_t> &)b8vreg);
        MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(preg, preg);
        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstUb + i * sregLower, b8vreg, preg);
    }
}

/* **************************************************************************************************
 * pertensor process for fp8 output                                             *
 * ************************************************************************************************* */
template <typename dstT, typename srcT>
__aicore__ inline void QuantPertensorForFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const float scale, const float offset, const uint32_t calCount)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    QuantPertensorForFp8VF<dstT, srcT>(dstUb, srcUb, scale, offset, calCount);
}

template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuantImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    static_assert(SupportType<srcT, half, float>(),
        "This AscendQuant only support half/float input dtype");
    static_assert(SupportType<dstT, int8_t>(),
        "This AscendQuant only support int8_t output dtype");
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize() && calCount <= dstTensor.GetSize() && calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]",
            calCount, srcTensor.GetSize(), dstTensor.GetSize());
    });
    QuantPertensorForB8<dstT, srcT>(dstTensor, srcTensor, scale, offset, calCount); // for int8 output
}

template <typename dstT, typename srcT>
__simd_vf__ inline void QuantPerchannelForFp8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ srcT* scaleUb, __ubuf__ srcT* offsetUb, const uint32_t scaleCount,
    const uint32_t rowNum)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<float> offsetf32vreg;
    MicroAPI::RegTensor<float> scalef32vreg;

    MicroAPI::RegTensor<srcT> b16vreg;
    MicroAPI::RegTensor<srcT> offsetB16Vreg;
    MicroAPI::RegTensor<srcT> scaleB16Vreg;
    MicroAPI::RegTensor<dstT> b8vreg;

    uint32_t sregLower = (uint32_t)ASCENDC_QUANT_B32_VF_LEN;

    for (uint16_t i = 0; i < (uint16_t)rowNum; ++i) {
        uint32_t sreg = (uint32_t)scaleCount;
        uint16_t repeat = CeilDivision(scaleCount, sregLower);
        for (uint16_t j = 0; j < (uint16_t)repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            if constexpr (SupportType<srcT, half>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(b16vreg,
                    srcUb + i * scaleCount + j * sregLower);
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(scaleB16Vreg,
                    scaleUb + j * sregLower);
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(offsetB16Vreg,
                    offsetUb + j * sregLower);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32vreg, b16vreg, preg);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(scalef32vreg, scaleB16Vreg, preg);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(offsetf32vreg, offsetB16Vreg, preg);
            } else {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(f32vreg,
                    srcUb + i * scaleCount + j * sregLower);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(scalef32vreg, scaleUb + j * sregLower);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(offsetf32vreg, offsetUb + j * sregLower);
            }

            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, scalef32vreg, preg);
            MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, offsetf32vreg, preg);

            MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(b8vreg, f32vreg, preg);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>
                ((MicroAPI::RegTensor<uint16_t> &)b8vreg, (MicroAPI::RegTensor<uint32_t> &)b8vreg);
            MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(preg, preg);
            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstUb + i * scaleCount + j * sregLower, 
                b8vreg, preg);
        }
    }
}

/* **************************************************************************************************
 * perchannel process                                              *
 * ************************************************************************************************* */
template <typename dstT, typename srcT>
__aicore__ inline void QuantPerchannelForFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<srcT>& scaleTensor, const LocalTensor<srcT>& offsetTensor, const uint32_t scaleCount,
    const uint32_t rowNum)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ srcT* scaleUb = (__ubuf__ srcT*)scaleTensor.GetPhyAddr();
    __ubuf__ srcT* offsetUb = (__ubuf__ srcT*)offsetTensor.GetPhyAddr();
    QuantPerchannelForFp8VF<dstT, srcT>(dstUb, srcUb, scaleUb, offsetUb, scaleCount, rowNum);
}

template <typename dstT, typename srcT>
__simd_vf__ inline void QuantPerchannelForB8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ srcT* scaleUb, __ubuf__ srcT* offsetUb, const uint32_t scaleCount,
    const uint32_t rowNum)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<half> f16Vreg;
    MicroAPI::RegTensor<dstT> s8vreg;
    MicroAPI::RegTensor<half> scaleVreg;
    MicroAPI::RegTensor<half> offsetVreg;

    uint32_t sregLower = (uint32_t)ASCENDC_QUANT_B16_VF_LEN;

    for (uint16_t i = 0; i < (uint16_t)rowNum; ++i) {
        uint32_t sreg = (uint32_t)scaleCount;
        uint16_t repeat = CeilDivision(scaleCount, sregLower);
        for (uint16_t j = 0; j < (uint16_t)repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint16_t>(sreg);
            uint32_t srcOffset = i * scaleCount + j * sregLower;

            // half
            MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(f16Vreg, srcUb + srcOffset);
            MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(offsetVreg, offsetUb + j * sregLower);
            MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb + j * sregLower);

            MicroAPI::Mul<half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, scaleVreg, preg);
            MicroAPI::Add<half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, offsetVreg, preg);

            if constexpr (SupportType<dstT, int8_t>()) {
                MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(s8vreg, f16Vreg, preg);
            } else {
                MicroAPI::Cast<dstT, half, LayoutZMrgZRndASatS>(s8vreg, f16Vreg, preg);
            }

            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstUb + srcOffset, s8vreg, preg);
        }
    }
}

template <typename dstT, typename srcT>
__aicore__ inline void QuantPerchannelForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<srcT>& scaleTensor, const LocalTensor<srcT>& offsetTensor, const uint32_t scaleCount,
    const uint32_t rowNum)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ srcT* scaleUb = (__ubuf__ srcT*)scaleTensor.GetPhyAddr();
    __ubuf__ srcT* offsetUb = (__ubuf__ srcT*)offsetTensor.GetPhyAddr();
    QuantPerchannelForB8VF<dstT, srcT>(dstUb, srcUb, scaleUb, offsetUb, scaleCount, rowNum);
}

template <typename dstT, typename srcT>
__simd_vf__ inline void QuantPerchannelForB8VF(__ubuf__ dstT* dstUb, __ubuf__ float* srcUb,
    __ubuf__ float* scaleUb, __ubuf__ float* offsetUb, const uint32_t scaleCount,
    const uint32_t rowNum)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<half> f16vreg;
    MicroAPI::RegTensor<dstT> b8vreg;
    MicroAPI::RegTensor<half> scalevreg;
    MicroAPI::RegTensor<half> offsetvreg;
    MicroAPI::RegTensor<float> scaleB32Vreg;
    MicroAPI::RegTensor<float> offsetB32Vreg;

    uint32_t sregLower = (uint32_t)ASCENDC_QUANT_B32_VF_LEN;

    for (uint16_t i = 0; i < (uint16_t)rowNum; ++i) {
        uint32_t sreg = (uint32_t)scaleCount;
        uint16_t repeat = CeilDivision(scaleCount, sregLower);
        for (uint16_t j = 0; j < (uint16_t)repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);

            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(f32vreg,
                srcUb + i * scaleCount + j * sregLower);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(offsetB32Vreg, offsetUb + j * sregLower);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(scaleB32Vreg, scaleUb + j * sregLower);

            MicroAPI::Cast<half, float, LayoutZMrgZRndRSatS>(f16vreg, f32vreg, preg);
            MicroAPI::Cast<half, float, LayoutZMrgZRndRSatS>(offsetvreg, offsetB32Vreg, preg);
            MicroAPI::Cast<half, float, LayoutZMrgZRndRSatS>(scalevreg, scaleB32Vreg, preg);

            MicroAPI::Mul<half, MicroAPI::MaskMergeMode::ZEROING>(f16vreg, f16vreg, scalevreg, preg);
            MicroAPI::Add<half, MicroAPI::MaskMergeMode::ZEROING>(f16vreg, f16vreg, offsetvreg, preg);

            if constexpr (SupportType<dstT, int8_t>()) {
                MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(b8vreg, f16vreg, preg);
            } else {
                MicroAPI::Cast<dstT, half, LayoutZMrgZRndASatS>(b8vreg, f16vreg, preg);
            }
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>
                ((MicroAPI::RegTensor<uint16_t> &)b8vreg, (MicroAPI::RegTensor<uint32_t> &)b8vreg);
            MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(preg, preg);
            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstUb + i * scaleCount + j * sregLower, 
                b8vreg, preg);
        }
    }
}

template <typename dstT, typename srcT>
__aicore__ inline void QuantPerchannelForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& scaleTensor, const LocalTensor<float>& offsetTensor, const uint32_t scaleCount,
    const uint32_t rowNum)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ float* srcUb = (__ubuf__ float*)srcTensor.GetPhyAddr();
    __ubuf__ float* scaleUb = (__ubuf__ float*)scaleTensor.GetPhyAddr();
    __ubuf__ float* offsetUb = (__ubuf__ float*)offsetTensor.GetPhyAddr();
    QuantPerchannelForB8VF<dstT, srcT>(dstUb, srcUb, scaleUb, offsetUb, scaleCount, rowNum);
}

template <typename dstT, typename srcT>
__simd_vf__ inline void QuantPerchannelForB8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ srcT* scaleUb, const srcT offset, const uint32_t scaleCount, const uint32_t rowNum)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<half> f16Vreg;
    MicroAPI::RegTensor<dstT> s8vreg;
    MicroAPI::RegTensor<half> scaleVreg;
    uint32_t sregLower = (uint32_t)ASCENDC_QUANT_B16_VF_LEN;

    for (uint16_t i = 0; i < (uint16_t)rowNum; ++i) {
        uint32_t sreg = (uint32_t)scaleCount;
        uint16_t repeat = CeilDivision(scaleCount, sregLower);
        for (uint16_t j = 0; j < (uint16_t)repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint16_t>(sreg);

            // half
            MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(f16Vreg,
                srcUb + i * scaleCount + j * sregLower);
            MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb + j * sregLower);
            MicroAPI::Mul<half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, scaleVreg, preg);
            MicroAPI::Adds<half, half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, offset, preg);

            if constexpr (SupportType<dstT, int8_t>()) {
                MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(s8vreg, f16Vreg, preg);
            } else {
                MicroAPI::Cast<dstT, half, LayoutZMrgZRndASatS>(s8vreg, f16Vreg, preg);
            }

            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstUb + i * scaleCount + j * sregLower,
                s8vreg, preg);
        }
    }
}

template <typename dstT, typename srcT>
__aicore__ inline void QuantPerchannelForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<srcT>& scaleTensor, const srcT offset, const uint32_t scaleCount, const uint32_t rowNum)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ srcT* scaleUb = (__ubuf__ srcT*)scaleTensor.GetPhyAddr();
    QuantPerchannelForB8VF<dstT, srcT>(dstUb, srcUb, scaleUb, offset, scaleCount, rowNum);
}

template <typename dstT, typename srcT>
__simd_vf__ inline void QuantPerchannelForB8VF(__ubuf__ dstT* dstUb, __ubuf__ float* srcUb,
    __ubuf__ float* scaleUb, const float offset, const uint32_t scaleCount, const uint32_t rowNum)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<half> f16Vreg;
    MicroAPI::RegTensor<dstT> b8vreg;
    MicroAPI::RegTensor<half> scaleVreg;
    MicroAPI::RegTensor<float> scaleB32Vreg;

    uint32_t sregLower = (uint32_t)ASCENDC_QUANT_B32_VF_LEN;

    for (uint16_t i = 0; i < (uint16_t)rowNum; ++i) {
        uint32_t sreg = (uint32_t)scaleCount;
        uint16_t repeat = CeilDivision(scaleCount, sregLower);
        for (uint16_t j = 0; j < (uint16_t)repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);

            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(f32vreg,
                srcUb + i * scaleCount + j * sregLower);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(scaleB32Vreg, scaleUb + j * sregLower);

            MicroAPI::Cast<half, float, LayoutZMrgZRndRSatS>(f16Vreg, f32vreg, preg);
            MicroAPI::Cast<half, float, LayoutZMrgZRndRSatS>(scaleVreg, scaleB32Vreg, preg);

            MicroAPI::Mul<half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, scaleVreg, preg);
            MicroAPI::Adds<half, half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg,
                static_cast<half>(offset), preg);

            if constexpr (SupportType<dstT, int8_t>()) {
                MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(b8vreg, f16Vreg, preg);
            } else {
                MicroAPI::Cast<dstT, half, LayoutZMrgZRndASatS>(b8vreg, f16Vreg, preg);
            }
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>
                ((MicroAPI::RegTensor<uint16_t> &)b8vreg, (MicroAPI::RegTensor<uint32_t> &)b8vreg);
            MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(preg, preg);
            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstUb + i * scaleCount + j * sregLower, 
                b8vreg, preg);
        }
    }
}

template <typename dstT, typename srcT>
__aicore__ inline void QuantPerchannelForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& scaleTensor, const float offset, const uint32_t scaleCount, const uint32_t rowNum)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ float* srcUb = (__ubuf__ float*)srcTensor.GetPhyAddr();
    __ubuf__ float* scaleUb = (__ubuf__ float*)scaleTensor.GetPhyAddr();
    QuantPerchannelForB8VF<dstT, srcT>(dstUb, srcUb, scaleUb, offset, scaleCount, rowNum);
}

template <typename dstT, typename srcT>
__simd_vf__ inline void QuantPerchannelForFp8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ srcT* scaleUb, const srcT offset, const uint32_t scaleCount, const uint32_t rowNum)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<float> scalef32vreg;
    MicroAPI::RegTensor<srcT> b16vreg;
    MicroAPI::RegTensor<srcT> scaleB16Vreg;
    MicroAPI::RegTensor<dstT> b8vreg;

    uint32_t sregLower = (uint32_t)ASCENDC_QUANT_B32_VF_LEN;

    for (uint16_t i = 0; i < (uint16_t)rowNum; ++i) {
        uint32_t sreg = (uint32_t)scaleCount;
        uint16_t repeat = CeilDivision(scaleCount, sregLower);
        for (uint16_t j = 0; j < (uint16_t)repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            if constexpr (SupportType<srcT, half>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(b16vreg,
                    srcUb + i * scaleCount + j * sregLower);
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(scaleB16Vreg,
                    scaleUb + j * sregLower);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32vreg, b16vreg, preg);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(scalef32vreg, scaleB16Vreg, preg);
            } else {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(f32vreg,
                    srcUb + i * scaleCount + j * sregLower);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(scalef32vreg, scaleUb + j * sregLower);
            }

            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, scalef32vreg, preg);
            MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg,
                static_cast<float>(offset), preg);

            MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(b8vreg, f32vreg, preg);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>
                ((MicroAPI::RegTensor<uint16_t> &)b8vreg, (MicroAPI::RegTensor<uint32_t> &)b8vreg);
            MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(preg, preg);
            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstUb + i * scaleCount + j * sregLower, 
                b8vreg, preg);
        }
    }
}

template <typename dstT, typename srcT>
__aicore__ inline void QuantPerchannelForFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<srcT>& scaleTensor, const srcT offset, const uint32_t scaleCount, const uint32_t rowNum)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ srcT* scaleUb = (__ubuf__ srcT*)scaleTensor.GetPhyAddr();
    QuantPerchannelForFp8VF<dstT, srcT>(dstUb, srcUb, scaleUb, offset, scaleCount, rowNum);
}

template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuantImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<srcT>& scaleTensor,
    const LocalTensor<srcT>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount,
    const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(scaleTensor, "scaleTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(offsetTensor, "offsetTensor", "VECIN, VECOUT, VECCALC");
    static_assert(SupportType<srcT, half, float>(),
        "This AscendQuant only support half/float input dtype");
    static_assert(SupportType<dstT, int8_t>(),
        "This AscendQuant only support int8_t output dtype");
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize() && calCount <= dstTensor.GetSize() && calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]",
            calCount, srcTensor.GetSize(), dstTensor.GetSize());
    });
    ASCENDC_ASSERT((scaleCount > 0 && scaleCount  == offsetCount),
            { KERNEL_LOG(KERNEL_ERROR, "scaleCount must be greater than 0 and equal to offsetCount!"); });
    ASCENDC_ASSERT((calCount % 32 == 0 && calCount % scaleCount == 0),
            { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of 32 and scaleCount!"); });
    ASCENDC_ASSERT((scaleCount  == offsetCount),
            { KERNEL_LOG(KERNEL_ERROR, "scaleCount equal to offsetCount!"); });
    const uint32_t rowNum = calCount / scaleCount;
    QuantPerchannelForB8<dstT, srcT>(dstTensor, srcTensor, scaleTensor, offsetTensor, scaleCount,
        rowNum); // for int8 output
}

template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuantImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<srcT>& scaleTensor, const srcT offset,
    const uint32_t scaleCount, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(scaleTensor, "scaleTensor", "VECIN, VECOUT, VECCALC");
    static_assert(SupportType<srcT, half, float>(),
        "This AscendQuant only support half/float input dtype");
    static_assert(SupportType<dstT, int8_t>(),
        "This AscendQuant only support int8_t output dtype");
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize() && calCount <= dstTensor.GetSize() && calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]",
            calCount, srcTensor.GetSize(), dstTensor.GetSize());
    });
    ASCENDC_ASSERT((scaleCount > 0),
            { KERNEL_LOG(KERNEL_ERROR, "scaleCount must be greater than 0"); });
    ASCENDC_ASSERT((calCount % 32 == 0 && calCount % scaleCount == 0),
            { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of 32 and scaleCount!"); });
    const uint32_t rowNum = calCount / scaleCount;
    QuantPerchannelForB8<dstT, srcT>(dstTensor, srcTensor, scaleTensor, offset, scaleCount,
        rowNum); // for int8 output
}

template <typename T, bool isReuseSource = false, const AscendQuantConfig &config>
__aicore__ inline void AscendQuantImpl(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const T offset,
    const uint32_t scaleCount, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(scaleTensor, "scaleTensor", "VECIN, VECOUT, VECCALC");
    static_assert(SupportType<T, half, float>(),
        "This AscendQuant only support half/float input dtype");

    constexpr bool enableConfig = config.calcCount != 0 && config.scaleCount != 0;
    const uint32_t calCountReal = enableConfig ? config.calcCount : calCount;
    const uint32_t scaleCountReal = enableConfig ? config.scaleCount : scaleCount;

    ASCENDC_ASSERT((calCountReal <= srcTensor.GetSize() && calCountReal <= dstTensor.GetSize() && calCountReal >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]",
            calCountReal, srcTensor.GetSize(), dstTensor.GetSize());
    });
    ASCENDC_ASSERT((scaleCountReal > 0),
            { KERNEL_LOG(KERNEL_ERROR, "scaleCount must be greater than 0"); });
    ASCENDC_ASSERT((calCountReal % 32 == 0 && calCountReal % scaleCountReal == 0),
            { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of 32 and scaleCount!"); });
    const uint32_t rowNum = calCountReal / scaleCountReal;
    QuantPerchannelForB8<int8_t, T>(dstTensor, srcTensor, scaleTensor, offset, scaleCountReal,
        rowNum); // for int8 output
}

template <typename T, bool isReuseSource = false, const AscendQuantConfig &config>
__aicore__ inline void AscendQuantImpl(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor,
    const uint32_t scaleCount, const uint32_t offsetCount, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(scaleTensor, "scaleTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(offsetTensor, "offsetTensor", "VECIN, VECOUT, VECCALC");
    static_assert(SupportType<T, half, float>(),
        "This AscendQuant only support half/float input dtype");

    constexpr bool enableConfig = config.calcCount != 0 && config.scaleCount != 0 && config.offsetCount != 0;
    const uint32_t calCountReal = enableConfig ? config.calcCount : calCount;
    const uint32_t scaleCountReal = enableConfig ? config.scaleCount : scaleCount;
    const uint32_t offsetCountReal = enableConfig ? config.offsetCount : offsetCount;

    ASCENDC_ASSERT((calCountReal <= srcTensor.GetSize() && calCountReal <= dstTensor.GetSize() && calCountReal >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]",
            calCountReal, srcTensor.GetSize(), dstTensor.GetSize());
    });
    ASCENDC_ASSERT((scaleCountReal > 0 && scaleCountReal  == offsetCountReal),
            { KERNEL_LOG(KERNEL_ERROR, "scaleCount must be greater than 0 and equal to offsetCount!"); });
    ASCENDC_ASSERT((calCountReal % 32 == 0 && calCountReal % scaleCountReal == 0),
            { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of 32 and scaleCount!"); });
    const uint32_t rowNum = calCountReal / scaleCountReal;
    QuantPerchannelForB8<int8_t, T>(dstTensor, srcTensor, scaleTensor, offsetTensor, scaleCountReal,
        rowNum); // for int8 output
}

template <typename scaleT>
__aicore__ constexpr inline float ConvertToFloat(const scaleT& offset)
{
    return static_cast<float>(offset);
}

template <typename scaleT, const AscendQuantConfig& config>
__simd_callee__ inline void GetPerTokenScaleAndOffset(__ubuf__ scaleT* scaleAddr,
                                                 __ubuf__ scaleT* offsetAddr,
                                                 MicroAPI::RegTensor<scaleT>& scaleVreg,
                                                 MicroAPI::RegTensor<scaleT>& offsetVreg)
{
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B16>(scaleVreg, scaleAddr);
    } else {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B32>(scaleVreg, scaleAddr);
    }
}

template <typename scaleT>
__simd_callee__ inline void GetPerTokenScale(__ubuf__ scaleT* scaleAddr,
                                        MicroAPI::RegTensor<scaleT>& scaleVreg)
{
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B16>(scaleVreg, scaleAddr);
    } else {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B32>(scaleVreg, scaleAddr);
    }
}

template <typename dstT, typename scaleT>
__simd_callee__ inline void StoreRes(__ubuf__ dstT* dstAddr, MicroAPI::RegTensor<dstT>& vreg,
                                MicroAPI::MaskReg& preg)
{
    if (SupportType<scaleT, float>()) {
        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(dstAddr, vreg, preg);
    } else {
        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstAddr, vreg, preg);
    }
}

template <typename T>
__simd_callee__ inline void GetPerGroupScale(__ubuf__ T* scaleUb, const int32_t start, const AscendQuantParam& para,
                                        const AscendQuantConfig& config, MicroAPI::RegTensor<T>& scaleReg)
{
    // use vgather to get perGroup scale/offset
    uint32_t groupSize = para.groupSize;
    if constexpr (SupportType<T, half>()) {
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int16_t> vci_vreg;
        MicroAPI::RegTensor<uint16_t> index_vreg;
        MicroAPI::RegTensor<uint16_t> gsize_vreg;
        MicroAPI::Duplicate(gsize_vreg, static_cast<uint16_t>(groupSize));
        MicroAPI::Arange(vci_vreg, static_cast<int16_t>(start));
        MicroAPI::Div(index_vreg, (MicroAPI::RegTensor<uint16_t> &)vci_vreg, gsize_vreg, preg);
        MicroAPI::DataCopyGather(scaleReg, scaleUb, index_vreg, preg);
    } else {
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int32_t> vci_vreg;
        MicroAPI::RegTensor<uint32_t> index_vreg;
        MicroAPI::RegTensor<uint32_t> gsize_vreg;
        MicroAPI::Duplicate(gsize_vreg, static_cast<uint32_t>(groupSize));
        MicroAPI::Arange(vci_vreg, static_cast<int32_t>(start));
        MicroAPI::Div(index_vreg, (MicroAPI::RegTensor<uint32_t> &)vci_vreg, gsize_vreg, preg);
        MicroAPI::DataCopyGather(scaleReg, scaleUb, index_vreg, preg);
    }
}

template <typename T>
__simd_callee__ inline void GetPerGroupOffset(__ubuf__ T* offsetUb, const int32_t start, const AscendQuantParam& para,
                                         const AscendQuantConfig& config, MicroAPI::RegTensor<T>& offsetReg)
{
    // use vgather to get perGroup scale/offset
    uint32_t groupSize = para.groupSize;
    if constexpr (SupportType<T, half>()) {
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int16_t> vci_vreg;
        MicroAPI::RegTensor<uint16_t> index_vreg;
        MicroAPI::RegTensor<uint16_t> gsize_vreg;
        MicroAPI::Duplicate(gsize_vreg, static_cast<uint16_t>(groupSize));
        MicroAPI::Arange(vci_vreg, static_cast<int16_t>(start));
        MicroAPI::Div(index_vreg, (MicroAPI::RegTensor<uint16_t> &)vci_vreg, gsize_vreg, preg);
        MicroAPI::DataCopyGather(offsetReg, offsetUb, index_vreg, preg);
    } else {
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int32_t> vci_vreg;
        MicroAPI::RegTensor<uint32_t> index_vreg;
        MicroAPI::RegTensor<uint32_t> gsize_vreg;
        MicroAPI::Duplicate(gsize_vreg, static_cast<uint32_t>(groupSize));
        MicroAPI::Arange(vci_vreg, static_cast<int32_t>(start));
        MicroAPI::Div(index_vreg, (MicroAPI::RegTensor<uint32_t> &)vci_vreg, gsize_vreg, preg);
        MicroAPI::DataCopyGather(offsetReg, offsetUb, index_vreg, preg);
    }
}

template <typename scaleT>
__simd_callee__ inline void GenerateZeroVreg(MicroAPI::RegTensor<scaleT>& zeroVreg)
{
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::MaskReg b16FullPreg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(zeroVreg, static_cast<scaleT>(0), b16FullPreg);
    } else {
        MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(zeroVreg, static_cast<scaleT>(0), b32FullPreg);
    }
}

template <typename scaleT, const AscendQuantConfig& config>
__simd_callee__ inline void GetPerGroupScaleEntry(__ubuf__ scaleT* scaleAddr, const AscendQuantParam& para,
                                             int32_t start, MicroAPI::MaskReg& preg,
                                             MicroAPI::RegTensor<float>& f32ScaleVreg)
{
    MicroAPI::RegTensor<scaleT> zeroVreg;
    GenerateZeroVreg<scaleT>(zeroVreg);
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::RegTensor<scaleT> oriScaleVreg;
        MicroAPI::RegTensor<scaleT> tempVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        GetPerGroupScale(scaleAddr, start, para, config, oriScaleVreg);
        MicroAPI::Interleave(scaleVreg, tempVreg, oriScaleVreg, zeroVreg);
        MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32ScaleVreg, scaleVreg, preg);
    } else {
        GetPerGroupScale(scaleAddr, start, para, config, f32ScaleVreg);
    }
}

template <typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void GetPerGroupOffsetEntry(__ubuf__ scaleT* offsetAddr, const AscendQuantParam& para,
                                              int32_t start, MicroAPI::MaskReg& preg,
                                              MicroAPI::RegTensor<float>& f32OffsetVreg)
{
    MicroAPI::RegTensor<scaleT> zeroVreg;
    GenerateZeroVreg<scaleT>(zeroVreg);
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::RegTensor<scaleT> oriOffsetVreg;
        MicroAPI::RegTensor<scaleT> tempVreg;
        MicroAPI::RegTensor<scaleT> offsetVreg;
    }
}

template <typename scaleT>
__simd_callee__ inline void GetPerGroupKRowScaleEntry(__ubuf__ scaleT* scaleAddr,
                                                 MicroAPI::RegTensor<float>& f32ScaleVreg)
{
    MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<scaleT> tempVreg;
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempVreg, scaleAddr);
        MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32ScaleVreg, tempVreg, b32FullPreg);
    } else {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(f32ScaleVreg, scaleAddr);
    }
}

template <typename scaleT, const AscendQuantConfig& config>
__simd_callee__ inline void GetPerGroupKRowOffsetEntry(__ubuf__ scaleT* offsetAddr,
                                                  MicroAPI::RegTensor<float>& f32OffsetVreg)
{
    MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<scaleT> tempVreg;
}

template <typename dstT, typename scaleT, const MicroAPI::CastTrait& castTrait>
__simd_callee__ inline void TransRegForS8(MicroAPI::RegTensor<scaleT>& srcVreg, MicroAPI::RegTensor<dstT>& dstVreg,
                                     MicroAPI::MaskReg& preg)
{
    if constexpr (SupportType<scaleT, float>()) {
        // fp32->s16->fp16->s8
        MicroAPI::RegTensor<half> f16Vreg;
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND ||
                      castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR ||
                      castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<int16_t, scaleT, castTrait>((MicroAPI::RegTensor<int16_t> &)f16Vreg, srcVreg, preg);
        } else {
            MicroAPI::Cast<int16_t, scaleT, LayoutZMrgZRndRSatS>((MicroAPI::RegTensor<int16_t> &)f16Vreg, srcVreg, preg);
        }
        MicroAPI::Cast<half, int16_t, LayoutZMrgZRndRSatS>(f16Vreg, (MicroAPI::RegTensor<int16_t> &)f16Vreg, preg);
        MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(dstVreg, f16Vreg, preg);
    } else if constexpr (SupportType<scaleT, half>()) {
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND ||
                      castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR ||
                      castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<dstT, scaleT, castTrait>(dstVreg, srcVreg, preg);
        } else {
            MicroAPI::Cast<dstT, scaleT, LayoutZMrgZRndRSatS>(dstVreg, srcVreg, preg);
        }
    }
}

template <typename scaleT, const AscendQuantConfig& config>
__simd_callee__ inline void LoadContinousScaleAndOffset(__ubuf__ scaleT* scaleAddr, __ubuf__ scaleT* offsetAddr,
                                                   MicroAPI::RegTensor<scaleT>& scaleVreg,
                                                   MicroAPI::RegTensor<scaleT>& offsetVreg)
{
    MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleAddr);
}

template <typename srcT>
__simd_callee__ inline void LoadSrc(__ubuf__ srcT* srcAddr, MicroAPI::MaskReg& preg,
                               MicroAPI::RegTensor<float>& vreg)
{
    if constexpr (SupportType<srcT, half>()) {
        MicroAPI::RegTensor<srcT> srcVreg;
        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, srcAddr);
        MicroAPI::Cast<float, srcT, layoutZMrgZ>(vreg, srcVreg, preg);
    } else {
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(vreg, srcAddr);
    }
}

template <typename scaleT, const AscendQuantConfig& config>
__simd_callee__ inline void AddQuantOffsetIfExist(MicroAPI::RegTensor<float>& vreg, MicroAPI::RegTensor<float>& offsetVreg,
                                             MicroAPI::MaskReg& preg)
{
}


}  //  namespace AscendC
#endif  // LIB_ASCEND_QUANT_ASCEND_QUANT_L300_IMPL_H
