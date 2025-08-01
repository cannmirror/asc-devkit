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
 * \file ascend_quant_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_QUANTIZATION_QUANT_ASCEND_QUANT_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_QUANTIZATION_QUANT_ASCEND_QUANT_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quantization/ascend_quant_utils.h"

namespace AscendC {
constexpr uint32_t ASCENDC_QUANT_B16_VF_LEN = VECTOR_REG_WIDTH / sizeof(uint16_t);
constexpr uint32_t ASCENDC_QUANT_B32_VF_LEN = VECTOR_REG_WIDTH / sizeof(uint32_t);

/* **************************************************************************************************
 * pertensor process for int8/hif8 output                                             *
 * ************************************************************************************************* */
template <typename dstT, typename srcT>
__aicore__ inline void QuantPertensorForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const float scale, const float offset, const uint32_t calCount)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<bfloat16_t> b16vreg;
        MicroAPI::RegTensor<half> f16Vreg;
        MicroAPI::RegTensor<dstT> s8vreg;

        uint32_t sregLower = (uint32_t)ASCENDC_QUANT_B16_VF_LEN;
        uint32_t sreg = (uint32_t)calCount;
        uint16_t repeat = CeilDivision(calCount, sregLower);

        for (uint16_t i = 0; i < (uint16_t)repeat; ++i) {
            preg = MicroAPI::UpdateMask<uint16_t>(sreg);

            if constexpr (SupportType<srcT, bfloat16_t>()) {
                MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_NORM>(b16vreg, srcUb + i * sregLower);
                MicroAPI::Cast<half, bfloat16_t, MrgZRndRSatS>(f16Vreg, b16vreg, preg);
            } else {
                MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(f16Vreg, srcUb + i * sregLower);
            }

            MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(
                f16Vreg, f16Vreg, static_cast<half>(scale), preg);
            MicroAPI::Adds<half, half, MicroAPI::MaskMergeMode::ZEROING>(
                f16Vreg, f16Vreg, static_cast<half>(offset), preg);
            if constexpr (SupportType<dstT, int8_t>()) {
                MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(s8vreg, f16Vreg, preg);
            } else {
                MicroAPI::Cast<dstT, half, LayoutZMrgZRndASatS>(s8vreg, f16Vreg, preg);
            }
            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstUb + i * sregLower, s8vreg, preg);
        }
    }
}

template <typename dstT, typename srcT>
__aicore__ inline void QuantPertensorForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<float>& srcTensor,
    const float scale, const float offset, const uint32_t calCount)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ float* srcUb = (__local_mem__ float*)srcTensor.GetPhyAddr();

    __VEC_SCOPE__
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

            MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(
                f16Vreg, f16Vreg, static_cast<half>(scale), preg);
            MicroAPI::Adds<half, half, MicroAPI::MaskMergeMode::ZEROING>(
                f16Vreg, f16Vreg, static_cast<half>(offset), preg);

            if constexpr (SupportType<dstT, int8_t>()) {
                MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(s8vreg, f16Vreg, preg);
            } else {
                MicroAPI::Cast<dstT, half, LayoutZMrgZRndASatS>(s8vreg, f16Vreg, preg);
            }
            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(dstUb + i * sregLower, s8vreg, preg);
        }
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuantImpl(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(
        SupportType<T, half, float, bfloat16_t>(), "This AscendQuant only support half/float/bfloat16_t input dtype");
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize() && calCount <= dstTensor.GetSize() && calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]", calCount, srcTensor.GetSize(),
            dstTensor.GetSize());
    });
    QuantPertensorForB8<int8_t, T>(dstTensor, srcTensor, scale, offset, calCount);
}

/* **************************************************************************************************
 * pertensor process for fp8 output                                             *
 * ************************************************************************************************* */
template <typename dstT, typename srcT>
__aicore__ inline void QuantPertensorForFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const float scale, const float offset, const uint32_t calCount)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();

    __VEC_SCOPE__
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
            if constexpr (SupportType<srcT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(b16vreg, srcUb + i * sregLower);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32vreg, b16vreg, preg);
            } else {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(f32vreg, srcUb + i * sregLower);
            }

            MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(
                f32vreg, f32vreg, static_cast<float>(scale), preg);
            MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(
                f32vreg, f32vreg, static_cast<float>(offset), preg);

            MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(b8vreg, f32vreg, preg);
            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(dstUb + i * sregLower, b8vreg, preg);
        }
    }
}

template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuantImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<srcT, half, float, bfloat16_t>(),
        "This AscendQuant only support half/float/bfloat16_t input dtype");
    static_assert(SupportType<dstT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t>(),
        "This AscendQuant only support int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t output dtype");
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize() && calCount <= dstTensor.GetSize() && calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]", calCount, srcTensor.GetSize(),
            dstTensor.GetSize());
    });
    if constexpr (SupportType<dstT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        QuantPertensorForFp8<dstT, srcT>(dstTensor, srcTensor, scale, offset, calCount);
    } else {
        QuantPertensorForB8<dstT, srcT>(dstTensor, srcTensor, scale, offset, calCount); // for int8/hif8 output
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
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ srcT* scaleUb = (__local_mem__ srcT*)scaleTensor.GetPhyAddr();
    __local_mem__ srcT* offsetUb = (__local_mem__ srcT*)offsetTensor.GetPhyAddr();

    __VEC_SCOPE__
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
                if constexpr (SupportType<srcT, half, bfloat16_t>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        b16vreg, srcUb + i * scaleCount + j * sregLower);
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        scaleB16Vreg, scaleUb + j * sregLower);
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        offsetB16Vreg, offsetUb + j * sregLower);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32vreg, b16vreg, preg);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(scalef32vreg, scaleB16Vreg, preg);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(offsetf32vreg, offsetB16Vreg, preg);
                } else {
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                        f32vreg, srcUb + i * scaleCount + j * sregLower);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(scalef32vreg, scaleUb + j * sregLower);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(offsetf32vreg, offsetUb + j * sregLower);
                }

                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, scalef32vreg, preg);
                MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, offsetf32vreg, preg);

                MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(b8vreg, f32vreg, preg);
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    dstUb + i * scaleCount + j * sregLower, b8vreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT>
__aicore__ inline void QuantPerchannelForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<srcT>& scaleTensor, const LocalTensor<srcT>& offsetTensor, const uint32_t scaleCount,
    const uint32_t rowNum)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ srcT* scaleUb = (__local_mem__ srcT*)scaleTensor.GetPhyAddr();
    __local_mem__ srcT* offsetUb = (__local_mem__ srcT*)offsetTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<bfloat16_t> b16vreg;
        MicroAPI::RegTensor<half> f16Vreg;
        MicroAPI::RegTensor<dstT> s8vreg;
        MicroAPI::RegTensor<half> scaleVreg;
        MicroAPI::RegTensor<half> offsetVreg;
        MicroAPI::RegTensor<bfloat16_t> scaleBfVreg;
        MicroAPI::RegTensor<bfloat16_t> offsetB16Vreg;

        uint32_t sregLower = (uint32_t)ASCENDC_QUANT_B16_VF_LEN;

        for (uint16_t i = 0; i < (uint16_t)rowNum; ++i) {
            uint32_t sreg = (uint32_t)scaleCount;
            uint16_t repeat = CeilDivision(scaleCount, sregLower);
            for (uint16_t j = 0; j < (uint16_t)repeat; ++j) {
                preg = MicroAPI::UpdateMask<uint16_t>(sreg);
                uint32_t srcOffset = i * scaleCount + j * sregLower;

                if constexpr (SupportType<srcT, bfloat16_t>()) {
                    MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_NORM>(b16vreg, srcUb + srcOffset);
                    MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_NORM>(
                        offsetB16Vreg, offsetUb + j * sregLower);
                    MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_NORM>(scaleBfVreg, scaleUb + j * sregLower);
                    MicroAPI::Cast<half, bfloat16_t, MrgZRndRSatS>(f16Vreg, b16vreg, preg);
                    MicroAPI::Cast<half, bfloat16_t, MrgZRndRSatS>(offsetVreg, offsetB16Vreg, preg);
                    MicroAPI::Cast<half, bfloat16_t, MrgZRndRSatS>(scaleVreg, scaleBfVreg, preg);
                } else { // half
                    MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(f16Vreg, srcUb + srcOffset);
                    MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(offsetVreg, offsetUb + j * sregLower);
                    MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb + j * sregLower);
                }

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
}

template <typename dstT, typename srcT>
__aicore__ inline void QuantPerchannelForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& scaleTensor, const LocalTensor<float>& offsetTensor, const uint32_t scaleCount,
    const uint32_t rowNum)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ float* srcUb = (__local_mem__ float*)srcTensor.GetPhyAddr();
    __local_mem__ float* scaleUb = (__local_mem__ float*)scaleTensor.GetPhyAddr();
    __local_mem__ float* offsetUb = (__local_mem__ float*)offsetTensor.GetPhyAddr();

    __VEC_SCOPE__
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

                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                    f32vreg, srcUb + i * scaleCount + j * sregLower);
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
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    dstUb + i * scaleCount + j * sregLower, b8vreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT>
__aicore__ inline void QuantPerchannelForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<srcT>& scaleTensor, const srcT offset, const uint32_t scaleCount, const uint32_t rowNum)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ srcT* scaleUb = (__local_mem__ srcT*)scaleTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<bfloat16_t> b16vreg;
        MicroAPI::RegTensor<half> f16Vreg;
        MicroAPI::RegTensor<dstT> s8vreg;
        MicroAPI::RegTensor<half> scaleVreg;
        MicroAPI::RegTensor<bfloat16_t> scaleB16Vreg;
        uint32_t sregLower = (uint32_t)ASCENDC_QUANT_B16_VF_LEN;

        for (uint16_t i = 0; i < (uint16_t)rowNum; ++i) {
            uint32_t sreg = (uint32_t)scaleCount;
            uint16_t repeat = CeilDivision(scaleCount, sregLower);
            for (uint16_t j = 0; j < (uint16_t)repeat; ++j) {
                preg = MicroAPI::UpdateMask<uint16_t>(sreg);

                if constexpr (SupportType<srcT, bfloat16_t>()) {
                    MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_NORM>(
                        b16vreg, srcUb + i * scaleCount + j * sregLower);
                    MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_NORM>(
                        scaleB16Vreg, scaleUb + j * sregLower);
                    MicroAPI::Cast<half, bfloat16_t, MrgZRndRSatS>(f16Vreg, b16vreg, preg);
                    MicroAPI::Cast<half, bfloat16_t, MrgZRndRSatS>(scaleVreg, scaleB16Vreg, preg);
                    MicroAPI::Mul<half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, scaleVreg, preg);
                    MicroAPI::Adds<half, half, MicroAPI::MaskMergeMode::ZEROING>(
                        f16Vreg, f16Vreg, static_cast<half>(ToFloat(offset)), preg);
                } else { // half
                    MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(
                        f16Vreg, srcUb + i * scaleCount + j * sregLower);
                    MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb + j * sregLower);
                    MicroAPI::Mul<half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, scaleVreg, preg);
                    MicroAPI::Adds<half, half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, offset, preg);
                }

                if constexpr (SupportType<dstT, int8_t>()) {
                    MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(s8vreg, f16Vreg, preg);
                } else {
                    MicroAPI::Cast<dstT, half, LayoutZMrgZRndASatS>(s8vreg, f16Vreg, preg);
                }

                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(
                    dstUb + i * scaleCount + j * sregLower, s8vreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT>
__aicore__ inline void QuantPerchannelForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& scaleTensor, const float offset, const uint32_t scaleCount, const uint32_t rowNum)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ float* srcUb = (__local_mem__ float*)srcTensor.GetPhyAddr();
    __local_mem__ float* scaleUb = (__local_mem__ float*)scaleTensor.GetPhyAddr();

    __VEC_SCOPE__
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

                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                    f32vreg, srcUb + i * scaleCount + j * sregLower);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(scaleB32Vreg, scaleUb + j * sregLower);

                MicroAPI::Cast<half, float, LayoutZMrgZRndRSatS>(f16Vreg, f32vreg, preg);
                MicroAPI::Cast<half, float, LayoutZMrgZRndRSatS>(scaleVreg, scaleB32Vreg, preg);

                MicroAPI::Mul<half, MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, scaleVreg, preg);
                MicroAPI::Adds<half, half, MicroAPI::MaskMergeMode::ZEROING>(
                    f16Vreg, f16Vreg, static_cast<half>(offset), preg);

                if constexpr (SupportType<dstT, int8_t>()) {
                    MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(b8vreg, f16Vreg, preg);
                } else {
                    MicroAPI::Cast<dstT, half, LayoutZMrgZRndASatS>(b8vreg, f16Vreg, preg);
                }

                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    dstUb + i * scaleCount + j * sregLower, b8vreg, preg);
            }
        }
    }
}
template <typename dstT, typename srcT>
__aicore__ inline void QuantPerchannelForFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<srcT>& scaleTensor, const srcT offset, const uint32_t scaleCount, const uint32_t rowNum)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ srcT* scaleUb = (__local_mem__ srcT*)scaleTensor.GetPhyAddr();

    __VEC_SCOPE__
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
                if constexpr (SupportType<srcT, half, bfloat16_t>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        b16vreg, srcUb + i * scaleCount + j * sregLower);
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        scaleB16Vreg, scaleUb + j * sregLower);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32vreg, b16vreg, preg);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(scalef32vreg, scaleB16Vreg, preg);
                } else {
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                        f32vreg, srcUb + i * scaleCount + j * sregLower);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(scalef32vreg, scaleUb + j * sregLower);
                }

                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, scalef32vreg, preg);
                if constexpr (SupportType<srcT, bfloat16_t>()) {
                    MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(
                        f32vreg, f32vreg, ToFloat(offset), preg);
                } else {
                    MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(
                        f32vreg, f32vreg, static_cast<float>(offset), preg);
                }

                MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(b8vreg, f32vreg, preg);
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    dstUb + i * scaleCount + j * sregLower, b8vreg, preg);
            }
        }
    }
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
    static_assert(SupportType<srcT, half, float, bfloat16_t>(),
        "This AscendQuant only support half/float/bfloat16_t input dtype");
    static_assert(SupportType<dstT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t>(),
        "This AscendQuant only support int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t output dtype");
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize() && calCount <= dstTensor.GetSize() && calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]", calCount, srcTensor.GetSize(),
            dstTensor.GetSize());
    });
    ASCENDC_ASSERT((scaleCount > 0 && scaleCount == offsetCount),
        { KERNEL_LOG(KERNEL_ERROR, "scaleCount must be greater than 0 and equal to offsetCount!"); });
    ASCENDC_ASSERT((calCount % 32 == 0 && calCount % scaleCount == 0),
        { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of 32 and scaleCount!"); });
    ASCENDC_ASSERT((scaleCount == offsetCount), { KERNEL_LOG(KERNEL_ERROR, "scaleCount equal to offsetCount!"); });
    const uint32_t rowNum = calCount / scaleCount;
    if constexpr (SupportType<dstT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        QuantPerchannelForFp8<dstT, srcT>(dstTensor, srcTensor, scaleTensor, offsetTensor, scaleCount, rowNum);
    } else {
        QuantPerchannelForB8<dstT, srcT>(dstTensor, srcTensor, scaleTensor, offsetTensor, scaleCount,
            rowNum); // for int8/hif8 output
    }
}

template <typename dstT, typename srcT, bool isReuseSource = false>
__aicore__ inline void AscendQuantImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<srcT>& scaleTensor, const srcT offset,
    const uint32_t scaleCount, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<srcT, half, float, bfloat16_t>(),
        "This AscendQuant only support half/float/bfloat16_t input dtype");
    static_assert(SupportType<dstT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t>(),
        "This AscendQuant only support int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t output dtype");
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize() && calCount <= dstTensor.GetSize() && calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]", calCount, srcTensor.GetSize(),
            dstTensor.GetSize());
    });
    ASCENDC_ASSERT((scaleCount > 0), { KERNEL_LOG(KERNEL_ERROR, "scaleCount must be greater than 0"); });
    ASCENDC_ASSERT((calCount % 32 == 0 && calCount % scaleCount == 0),
        { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of 32 and scaleCount!"); });
    const uint32_t rowNum = calCount / scaleCount;
    if constexpr (SupportType<dstT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        QuantPerchannelForFp8<dstT, srcT>(dstTensor, srcTensor, scaleTensor, offset, scaleCount, rowNum);
    } else {
        QuantPerchannelForB8<dstT, srcT>(dstTensor, srcTensor, scaleTensor, offset, scaleCount,
            rowNum); // for int8/hif8 output
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuantImpl(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const T offset,
    const uint32_t scaleCount, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(
        SupportType<T, half, float, bfloat16_t>(), "This AscendQuant only support half/float/bfloat16_t input dtype");
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize() && calCount <= dstTensor.GetSize() && calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]", calCount, srcTensor.GetSize(),
            dstTensor.GetSize());
    });
    ASCENDC_ASSERT((scaleCount > 0), { KERNEL_LOG(KERNEL_ERROR, "scaleCount must be greater than 0"); });
    ASCENDC_ASSERT((calCount % 32 == 0 && calCount % scaleCount == 0),
        { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of 32 and scaleCount!"); });
    const uint32_t rowNum = calCount / scaleCount;
    QuantPerchannelForB8<int8_t, T>(dstTensor, srcTensor, scaleTensor, offset, scaleCount,
        rowNum); // for int8/hif8 output
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AscendQuantImpl(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor,
    const uint32_t scaleCount, const uint32_t offsetCount, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(
        SupportType<T, half, float, bfloat16_t>(), "This AscendQuant only support half/float/bfloat16_t input dtype");
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize() && calCount <= dstTensor.GetSize() && calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]", calCount, srcTensor.GetSize(),
            dstTensor.GetSize());
    });
    ASCENDC_ASSERT((scaleCount > 0 && scaleCount == offsetCount),
        { KERNEL_LOG(KERNEL_ERROR, "scaleCount must be greater than 0 and equal to offsetCount!"); });
    ASCENDC_ASSERT((calCount % 32 == 0 && calCount % scaleCount == 0),
        { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of 32 and scaleCount!"); });
    const uint32_t rowNum = calCount / scaleCount;
    QuantPerchannelForB8<int8_t, T>(dstTensor, srcTensor, scaleTensor, offsetTensor, scaleCount,
        rowNum); // for int8/hif8 output
}

template <typename scaleT>
__aicore__ inline float ConvertToFloat(const scaleT& offset)
{
    if constexpr (SupportType<scaleT, bfloat16_t>()) {
        return ToFloat(offset);
    }
    return static_cast<float>(offset);
}

template <typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void GetPerTokenScaleAndOffset(__local_mem__ scaleT* scaleAddr, __local_mem__ scaleT* offsetAddr,
    MicroAPI::RegTensor<scaleT>& scaleVreg, MicroAPI::RegTensor<scaleT>& offsetVreg)
{
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B16>(scaleVreg, scaleAddr);
        if constexpr (config.hasOffset) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B16>(offsetVreg, offsetAddr);
        }
    } else {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B32>(scaleVreg, scaleAddr);
        if constexpr (config.hasOffset) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B32>(offsetVreg, offsetAddr);
        }
    }
}

template <typename scaleT>
__aicore__ inline void GetPerTokenScale(__local_mem__ scaleT* scaleAddr, MicroAPI::RegTensor<scaleT>& scaleVreg)
{
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B16>(scaleVreg, scaleAddr);
    } else {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B32>(scaleVreg, scaleAddr);
    }
}

template <typename dstT, typename scaleT>
__aicore__ inline void StoreRes(__local_mem__ dstT* dstAddr, MicroAPI::RegTensor<dstT>& vreg, MicroAPI::MaskReg& preg)
{
    if (SupportType<scaleT, float>()) {
        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(dstAddr, vreg, preg);
    } else {
        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstAddr, vreg, preg);
    }
}

template <typename T>
__aicore__ inline void GetPerGroupScale(__local_mem__ T* scaleUb, const int32_t start, const AscendQuantParam& para,
    const AscendQuantConfig& config, MicroAPI::RegTensor<T>& scaleReg)
{
    // use vgather to get perGroup scale/offset
    uint32_t groupSize = para.groupSize;
    if constexpr (SupportType<T, half, bfloat16_t>()) {
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int16_t> vci_vreg;
        MicroAPI::RegTensor<uint16_t> index_vreg;
        MicroAPI::RegTensor<uint16_t> gsize_vreg;
        MicroAPI::Duplicate(gsize_vreg, static_cast<uint16_t>(groupSize));
        MicroAPI::Arange(vci_vreg, static_cast<int16_t>(start));
        MicroAPI::Div(index_vreg, (MicroAPI::RegTensor<uint16_t>&)vci_vreg, gsize_vreg, preg);
        MicroAPI::DataCopyGather(scaleReg, scaleUb, index_vreg, preg);
    } else {
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int32_t> vci_vreg;
        MicroAPI::RegTensor<uint32_t> index_vreg;
        MicroAPI::RegTensor<uint32_t> gsize_vreg;
        MicroAPI::Duplicate(gsize_vreg, static_cast<uint32_t>(groupSize));
        MicroAPI::Arange(vci_vreg, static_cast<int32_t>(start));
        MicroAPI::Div(index_vreg, (MicroAPI::RegTensor<uint32_t>&)vci_vreg, gsize_vreg, preg);
        MicroAPI::DataCopyGather(scaleReg, scaleUb, index_vreg, preg);
    }
}

template <typename T>
__aicore__ inline void GetPerGroupOffset(__local_mem__ T* offsetUb, const int32_t start, const AscendQuantParam& para,
    const AscendQuantConfig& config, MicroAPI::RegTensor<T>& offsetReg)
{
    // use vgather to get perGroup scale/offset
    uint32_t groupSize = para.groupSize;
    if constexpr (SupportType<T, half, bfloat16_t>()) {
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int16_t> vci_vreg;
        MicroAPI::RegTensor<uint16_t> index_vreg;
        MicroAPI::RegTensor<uint16_t> gsize_vreg;
        MicroAPI::Duplicate(gsize_vreg, static_cast<uint16_t>(groupSize));
        MicroAPI::Arange(vci_vreg, static_cast<int16_t>(start));
        MicroAPI::Div(index_vreg, (MicroAPI::RegTensor<uint16_t>&)vci_vreg, gsize_vreg, preg);
        MicroAPI::DataCopyGather(offsetReg, offsetUb, index_vreg, preg);
    } else {
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int32_t> vci_vreg;
        MicroAPI::RegTensor<uint32_t> index_vreg;
        MicroAPI::RegTensor<uint32_t> gsize_vreg;
        MicroAPI::Duplicate(gsize_vreg, static_cast<uint32_t>(groupSize));
        MicroAPI::Arange(vci_vreg, static_cast<int32_t>(start));
        MicroAPI::Div(index_vreg, (MicroAPI::RegTensor<uint32_t>&)vci_vreg, gsize_vreg, preg);
        MicroAPI::DataCopyGather(offsetReg, offsetUb, index_vreg, preg);
    }
}

template <typename scaleT>
__aicore__ inline void GenerateZeroVreg(MicroAPI::RegTensor<scaleT>& zeroVreg)
{
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
        MicroAPI::MaskReg b16FullPreg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(zeroVreg, static_cast<scaleT>(0), b16FullPreg);
    } else {
        MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(zeroVreg, static_cast<scaleT>(0), b32FullPreg);
    }
}

template <typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void GetPerGroupScaleEntry(__local_mem__ scaleT* scaleAddr, const AscendQuantParam& para,
    int32_t start, MicroAPI::MaskReg& preg, MicroAPI::RegTensor<float>& f32ScaleVreg)
{
    MicroAPI::RegTensor<scaleT> zeroVreg;
    GenerateZeroVreg<scaleT>(zeroVreg);
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
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
__aicore__ inline void GetPerGroupOffsetEntry(__local_mem__ scaleT* offsetAddr, const AscendQuantParam& para,
    int32_t start, MicroAPI::MaskReg& preg, MicroAPI::RegTensor<float>& f32OffsetVreg)
{
    MicroAPI::RegTensor<scaleT> zeroVreg;
    GenerateZeroVreg<scaleT>(zeroVreg);
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
        MicroAPI::RegTensor<scaleT> oriOffsetVreg;
        MicroAPI::RegTensor<scaleT> tempVreg;
        MicroAPI::RegTensor<scaleT> offsetVreg;
        if constexpr (config.hasOffset) {
            GetPerGroupOffset(offsetAddr, start, para, config, oriOffsetVreg);
            MicroAPI::Interleave(offsetVreg, tempVreg, oriOffsetVreg, zeroVreg);
            MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32OffsetVreg, offsetVreg, preg);
        }
    } else {
        if constexpr (config.hasOffset) {
            GetPerGroupOffset(offsetAddr, start, para, config, f32OffsetVreg);
        }
    }
}

template <typename scaleT>
__aicore__ inline void GetPerGroupKRowScaleEntry(
    __local_mem__ scaleT* scaleAddr, MicroAPI::RegTensor<float>& f32ScaleVreg)
{
    MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<scaleT> tempVreg;
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempVreg, scaleAddr);
        MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32ScaleVreg, tempVreg, b32FullPreg);
    } else {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(f32ScaleVreg, scaleAddr);
    }
}

template <typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void GetPerGroupKRowOffsetEntry(
    __local_mem__ scaleT* offsetAddr, MicroAPI::RegTensor<float>& f32OffsetVreg)
{
    MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<scaleT> tempVreg;
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
        if constexpr (config.hasOffset) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempVreg, offsetAddr);
            MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32OffsetVreg, tempVreg, b32FullPreg);
        }
    } else {
        if constexpr (config.hasOffset) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(f32OffsetVreg, offsetAddr);
        }
    }
}

template <typename dstT, typename scaleT, const MicroAPI::CastTrait& castTrait>
__aicore__ inline void TransRegForFp8(
    MicroAPI::RegTensor<scaleT>& srcVreg, MicroAPI::RegTensor<dstT>& dstVreg, MicroAPI::MaskReg& preg)
{
    if constexpr (castTrait.roundMode == RoundMode::CAST_RINT) {
        MicroAPI::Cast<dstT, scaleT, castTrait>(dstVreg, srcVreg, preg);
    } else {
        MicroAPI::Cast<dstT, scaleT, LayoutZMrgZRndRSatS>(dstVreg, srcVreg, preg);
    }
}

template <typename dstT, typename scaleT, const MicroAPI::CastTrait& castTrait>
__aicore__ inline void TransRegForHif8(
    MicroAPI::RegTensor<scaleT>& srcVreg, MicroAPI::RegTensor<dstT>& dstVreg, MicroAPI::MaskReg& preg)
{
    if constexpr (SupportType<scaleT, bfloat16_t>()) {
        // bf16->fp32->hif8
        MicroAPI::MaskReg preg1;
        MicroAPI::MaskReg preg2 = MicroAPI::CreateMask<scaleT, MicroAPI::MaskPattern::ALLF>();
        MicroAPI::RegTensor<float> f32Vreg;
        MicroAPI::RegTensor<float> f32Vreg2;
        MicroAPI::RegTensor<scaleT> srcVreg2;
        MicroAPI::RegTensor<dstT> dstVreg2;
        MicroAPI::MaskInterleave<scaleT>(preg1, preg2, preg, preg2);
        MicroAPI::Interleave(srcVreg, srcVreg2, srcVreg, srcVreg2);
        MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32Vreg, srcVreg, preg1);
        MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32Vreg2, srcVreg2, preg2);
        if constexpr (castTrait.roundMode == RoundMode::CAST_ROUND || castTrait.roundMode == RoundMode::CAST_HYBRID) {
            MicroAPI::Cast<dstT, float, castTrait>(dstVreg, f32Vreg, preg1);
            MicroAPI::Cast<dstT, float, castTrait>(dstVreg2, f32Vreg2, preg2);
        } else {
            MicroAPI::Cast<dstT, float, LayoutZMrgZRndASatS>(dstVreg, f32Vreg, preg1);
            MicroAPI::Cast<dstT, float, LayoutZMrgZRndASatS>(dstVreg2, f32Vreg2, preg2);
        }
        MicroAPI::DeInterleave((MicroAPI::RegTensor<scaleT>&)dstVreg, (MicroAPI::RegTensor<scaleT>&)dstVreg2,
            (MicroAPI::RegTensor<scaleT>&)dstVreg, (MicroAPI::RegTensor<scaleT>&)dstVreg2);
    } else if constexpr (SupportType<scaleT, half, float>()) {
        if constexpr (castTrait.roundMode == RoundMode::CAST_ROUND || castTrait.roundMode == RoundMode::CAST_HYBRID) {
            MicroAPI::Cast<dstT, scaleT, castTrait>(dstVreg, srcVreg, preg);
        } else {
            MicroAPI::Cast<dstT, scaleT, LayoutZMrgZRndASatS>(dstVreg, srcVreg, preg);
        }
    }
}

template <typename dstT, typename scaleT, const MicroAPI::CastTrait& castTrait>
__aicore__ inline void TransRegForS8(
    MicroAPI::RegTensor<scaleT>& srcVreg, MicroAPI::RegTensor<dstT>& dstVreg, MicroAPI::MaskReg& preg)
{
    if constexpr (SupportType<scaleT, bfloat16_t>()) {
        // bf16->fp32->s16->fp16->s8
        MicroAPI::MaskReg preg1;
        MicroAPI::MaskReg preg2 = MicroAPI::CreateMask<scaleT, MicroAPI::MaskPattern::ALLF>();
        MicroAPI::RegTensor<float> f32Vreg;
        MicroAPI::RegTensor<float> f32Vreg2;
        MicroAPI::RegTensor<scaleT> srcVreg2;
        MicroAPI::RegTensor<dstT> dstVreg2;
        MicroAPI::MaskInterleave<scaleT>(preg1, preg2, preg, preg2);
        MicroAPI::Interleave(srcVreg, srcVreg2, srcVreg, srcVreg2);
        MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32Vreg, srcVreg, preg1);
        MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32Vreg2, srcVreg2, preg2);
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND
                      || castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR
                      || castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<int16_t, float, castTrait>((MicroAPI::RegTensor<int16_t>&)f32Vreg, f32Vreg, preg1);
            MicroAPI::Cast<int16_t, float, castTrait>((MicroAPI::RegTensor<int16_t>&)f32Vreg2, f32Vreg2, preg2);
        } else {
            MicroAPI::Cast<int16_t, float, LayoutZMrgZRndRSatS>((MicroAPI::RegTensor<int16_t>&)f32Vreg, f32Vreg, preg1);
            MicroAPI::Cast<int16_t, float, LayoutZMrgZRndRSatS>(
                (MicroAPI::RegTensor<int16_t>&)f32Vreg2, f32Vreg2, preg2);
        }
        MicroAPI::Cast<half, int16_t, LayoutZMrgZRndRSatS>(
            (MicroAPI::RegTensor<half>&)f32Vreg, (MicroAPI::RegTensor<int16_t>&)f32Vreg, preg1);
        MicroAPI::Cast<half, int16_t, LayoutZMrgZRndRSatS>(
            (MicroAPI::RegTensor<half>&)f32Vreg2, (MicroAPI::RegTensor<int16_t>&)f32Vreg2, preg2);
        MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(dstVreg, (MicroAPI::RegTensor<half>&)f32Vreg, preg1);
        MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(dstVreg2, (MicroAPI::RegTensor<half>&)f32Vreg2, preg2);
        MicroAPI::DeInterleave((MicroAPI::RegTensor<scaleT>&)dstVreg, (MicroAPI::RegTensor<scaleT>&)dstVreg2,
            (MicroAPI::RegTensor<scaleT>&)dstVreg, (MicroAPI::RegTensor<scaleT>&)dstVreg2);
    } else if constexpr (SupportType<scaleT, float>()) {
        // fp32->s16->fp16->s8
        MicroAPI::RegTensor<half> f16Vreg;
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND
                      || castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR
                      || castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<int16_t, scaleT, castTrait>((MicroAPI::RegTensor<int16_t>&)f16Vreg, srcVreg, preg);
        } else {
            MicroAPI::Cast<int16_t, scaleT, LayoutZMrgZRndRSatS>((MicroAPI::RegTensor<int16_t>&)f16Vreg, srcVreg, preg);
        }
        MicroAPI::Cast<half, int16_t, LayoutZMrgZRndRSatS>(f16Vreg, (MicroAPI::RegTensor<int16_t>&)f16Vreg, preg);
        MicroAPI::Cast<dstT, half, LayoutZMrgZRndRSatS>(dstVreg, f16Vreg, preg);
    } else if constexpr (SupportType<scaleT, half>()) {
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND
                      || castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR
                      || castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<dstT, scaleT, castTrait>(dstVreg, srcVreg, preg);
        } else {
            MicroAPI::Cast<dstT, scaleT, LayoutZMrgZRndRSatS>(dstVreg, srcVreg, preg);
        }
    }
}

template <typename dstT, typename scaleT, const MicroAPI::CastTrait& castTrait>
__aicore__ inline void TransRegForFp4(
    MicroAPI::RegTensor<scaleT>& vreg, MicroAPI::RegTensor<dstT>& dstVreg, MicroAPI::MaskReg& preg)
{
    MicroAPI::RegTensor<bfloat16_t> bf16Vreg;
    if constexpr (SupportType<scaleT, float>()) {
        MicroAPI::Cast<bfloat16_t, scaleT, LayoutZMrgZRndRSatS>(bf16Vreg, vreg, preg);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
            (MicroAPI::RegTensor<uint16_t>&)bf16Vreg, (MicroAPI::RegTensor<uint32_t>&)bf16Vreg);
        MicroAPI::MaskPack(preg, preg);
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND
                      || castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR
                      || castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<dstT, bfloat16_t, castTrait>(dstVreg, bf16Vreg, preg);
        } else {
            MicroAPI::Cast<dstT, bfloat16_t, LayoutZMrgZRndRSatS>(dstVreg, bf16Vreg, preg);
        }
    } else if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::Cast<bfloat16_t, scaleT, LayoutZMrgZRndRSatS>(bf16Vreg, vreg, preg);
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND
                      || castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR
                      || castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<dstT, bfloat16_t, castTrait>(dstVreg, bf16Vreg, preg);
        } else {
            MicroAPI::Cast<dstT, bfloat16_t, LayoutZMrgZRndRSatS>(dstVreg, bf16Vreg, preg);
        }
    } else if constexpr (SupportType<scaleT, bfloat16_t>()) {
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND
                      || castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR
                      || castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<dstT, bfloat16_t, castTrait>(dstVreg, vreg, preg);
        } else {
            MicroAPI::Cast<dstT, bfloat16_t, LayoutZMrgZRndRSatS>(dstVreg, vreg, preg);
        }
    }
}

template <typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void LoadContinousScaleAndOffset(__local_mem__ scaleT* scaleAddr, __local_mem__ scaleT* offsetAddr,
    MicroAPI::RegTensor<scaleT>& scaleVreg, MicroAPI::RegTensor<scaleT>& offsetVreg)
{
    MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleAddr);
    if constexpr (config.hasOffset) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(offsetVreg, offsetAddr);
    }
}

template <typename srcT>
__aicore__ inline void LoadSrc(__local_mem__ srcT* srcAddr, MicroAPI::MaskReg& preg, MicroAPI::RegTensor<float>& vreg)
{
    if constexpr (SupportType<srcT, half, bfloat16_t>()) {
        MicroAPI::RegTensor<srcT> srcVreg;
        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, srcAddr);
        MicroAPI::Cast<float, srcT, layoutZMrgZ>(vreg, srcVreg, preg);
    } else {
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(vreg, srcAddr);
    }
}

template <typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void AddQuantOffsetIfExist(
    MicroAPI::RegTensor<float>& vreg, MicroAPI::RegTensor<float>& offsetVreg, MicroAPI::MaskReg& preg)
{
    if constexpr (config.hasOffset) {
        MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, offsetVreg, preg);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerTokenForFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> offsetVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<float> f32ScaleVreg;
        MicroAPI::RegTensor<float> f32OffsetVreg;
        MicroAPI::RegTensor<srcT> srcVreg;
        MicroAPI::RegTensor<float> f32Vreg;
        MicroAPI::RegTensor<dstT> b8Vreg;
        MicroAPI::MaskReg b16FullPreg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < rowNum; ++i) {
            if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                GetPerTokenScaleAndOffset<scaleT, config>(scaleUb + i, offsetUb + i, scaleVreg, offsetVreg);
                MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32ScaleVreg, scaleVreg, b16FullPreg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32OffsetVreg, offsetVreg, b16FullPreg);
                }
            } else {
                GetPerTokenScaleAndOffset<scaleT, config>(scaleUb + i, offsetUb + i, f32ScaleVreg, f32OffsetVreg);
            }
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                if constexpr (SupportType<srcT, half, bfloat16_t>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        srcVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32Vreg, srcVreg, preg);
                } else {
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(f32Vreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32ScaleVreg, preg);
                AddQuantOffsetIfExist<float, config>(f32Vreg, f32OffsetVreg, preg);
                TransRegForFp8<dstT, float, castTrait>(f32Vreg, b8Vreg, preg);
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    dstUb + i * para.n + j * vecLen, b8Vreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerTokenForHif8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<scaleT> offsetVreg;
        MicroAPI::RegTensor<srcT> tempVreg;
        for (uint16_t i = 0; i < rowNum; ++i) {
            GetPerTokenScaleAndOffset<scaleT, config>(scaleUb + i, offsetUb + i, scaleVreg, offsetVreg);
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<scaleT>(sreg);
                if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        tempVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
                } else {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, offsetVreg, preg);
                }
                TransRegForHif8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                StoreRes<dstT, scaleT>(dstUb + i * para.n + j * vecLen, dstVreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerTokenForS8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<scaleT> offsetVreg;
        MicroAPI::RegTensor<srcT> tempVreg;
        for (uint16_t i = 0; i < rowNum; ++i) {
            GetPerTokenScaleAndOffset<scaleT, config>(scaleUb + i, offsetUb + i, scaleVreg, offsetVreg);
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<scaleT>(sreg);
                if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        tempVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
                } else {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, offsetVreg, preg);
                }
                TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                StoreRes<dstT, scaleT>(dstUb + i * para.n + j * vecLen, dstVreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerTokenForFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const scaleT& offset, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);
    float fp32_offset = ConvertToFloat<scaleT>(offset);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<float> f32ScaleVreg;
        MicroAPI::RegTensor<srcT> srcVreg;
        MicroAPI::RegTensor<float> f32Vreg;
        MicroAPI::RegTensor<dstT> b8Vreg;
        MicroAPI::MaskReg b16FullPreg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < rowNum; ++i) {
            if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B16>(scaleVreg, scaleUb + i);
                MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32ScaleVreg, scaleVreg, b16FullPreg);
            } else {
                MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B32>(f32ScaleVreg, scaleUb + i);
            }
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                if constexpr (SupportType<srcT, half, bfloat16_t>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        srcVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32Vreg, srcVreg, preg);
                } else {
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(f32Vreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32ScaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, fp32_offset, preg);
                }
                TransRegForFp8<dstT, float, castTrait>(f32Vreg, b8Vreg, preg);
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    dstUb + i * para.n + j * vecLen, b8Vreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerTokenForHif8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const scaleT& offset, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<srcT> tempVreg;
        for (uint16_t i = 0; i < rowNum; ++i) {
            GetPerTokenScale<scaleT>(scaleUb + i, scaleVreg);
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<scaleT>(sreg);
                if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        tempVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
                } else {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Adds<scaleT, scaleT, MicroAPI::MaskMergeMode::ZEROING>(
                        srcVreg, srcVreg, static_cast<scaleT>(offset), preg);
                }
                TransRegForHif8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                StoreRes<dstT, scaleT>(dstUb + i * para.n + j * vecLen, dstVreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerTokenForS8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const scaleT offset, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<srcT> tempVreg;
        for (uint16_t i = 0; i < rowNum; ++i) {
            GetPerTokenScale<scaleT>(scaleUb + i, scaleVreg);
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<scaleT>(sreg);
                if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        tempVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
                } else {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Adds<scaleT, scaleT, MicroAPI::MaskMergeMode::ZEROING>(
                        srcVreg, srcVreg, static_cast<scaleT>(offset), preg);
                }
                TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                StoreRes<dstT, scaleT>(dstUb + i * para.n + j * vecLen, dstVreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKColFp4(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<srcT> tempVreg;
        for (uint16_t i = 0; i < rowNum; ++i) {
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<scaleT>(sreg);
                GetPerGroupScale(scaleUb + i * scaleK, j * vecLen, para, config, scaleVreg);
                if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        tempVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
                } else {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                TransRegForFp4<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    (__local_mem__ uint8_t*)dstUb + (i * para.n + j * vecLen) / 2,
                    (MicroAPI::RegTensor<uint8_t>&)dstVreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKColFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<float> f32ScaleVreg;
        MicroAPI::RegTensor<float> f32OffsetVreg;
        MicroAPI::RegTensor<srcT> srcVreg;
        MicroAPI::RegTensor<float> f32Vreg;
        MicroAPI::RegTensor<dstT> b8Vreg;
        for (uint16_t i = 0; i < rowNum; ++i) {
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                GetPerGroupScaleEntry<scaleT, config>(scaleUb + i * scaleK, para, j * vecLen, preg, f32ScaleVreg);
                GetPerGroupScaleEntry<scaleT, config>(offsetUb + i * scaleK, para, j * vecLen, preg, f32OffsetVreg);
                if constexpr (SupportType<srcT, half, bfloat16_t>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        srcVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32Vreg, srcVreg, preg);
                } else {
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(f32Vreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32ScaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32OffsetVreg, preg);
                }
                TransRegForFp8<dstT, float, castTrait>(f32Vreg, b8Vreg, preg);
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    dstUb + i * para.n + j * vecLen, b8Vreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKColHif8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<scaleT> offsetVreg;
        MicroAPI::RegTensor<srcT> tempSrcVreg;
        for (uint16_t i = 0; i < rowNum; ++i) {
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<scaleT>(sreg);
                GetPerGroupScale(scaleUb + i * scaleK, j * vecLen, para, config, scaleVreg);
                if constexpr (config.hasOffset) {
                    GetPerGroupOffset(offsetUb + i * scaleK, j * vecLen, para, config, offsetVreg);
                }
                if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        tempSrcVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempSrcVreg, preg);
                } else {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, offsetVreg, preg);
                }
                TransRegForHif8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                StoreRes<dstT, scaleT>(dstUb + i * para.n + j * vecLen, dstVreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKColS8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<scaleT> offsetVreg;
        MicroAPI::RegTensor<srcT> tempVreg;
        for (uint16_t i = 0; i < rowNum; ++i) {
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<scaleT>(sreg);
                GetPerGroupScale(scaleUb + i * scaleK, j * vecLen, para, config, scaleVreg);
                if constexpr (config.hasOffset) {
                    GetPerGroupOffset(offsetUb + i * scaleK, j * vecLen, para, config, offsetVreg);
                }
                if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        tempVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
                } else {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, offsetVreg, preg);
                }
                TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                StoreRes<dstT, scaleT>(dstUb + i * para.n + j * vecLen, dstVreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKColFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const scaleT& offset, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint32_t sreg = para.n;
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);
    float fp32_offset = ConvertToFloat<scaleT>(offset);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<float> f32ScaleVreg;
        MicroAPI::RegTensor<float> f32OffsetVreg;
        MicroAPI::RegTensor<srcT> srcVreg;
        MicroAPI::RegTensor<float> f32Vreg;
        MicroAPI::RegTensor<dstT> b8Vreg;
        for (uint16_t i = 0; i < rowNum; ++i) {
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                GetPerGroupScaleEntry<scaleT, config>(scaleUb + i * scaleK, para, j * vecLen, preg, f32ScaleVreg);
                if constexpr (SupportType<srcT, half, bfloat16_t>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        srcVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32Vreg, srcVreg, preg);
                } else {
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(f32Vreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32ScaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, fp32_offset, preg);
                }
                TransRegForFp8<dstT, float, castTrait>(f32Vreg, b8Vreg, preg);
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    dstUb + i * para.n + j * vecLen, b8Vreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKColHif8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const scaleT& offset, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<srcT> tempSrcVreg;
        for (uint16_t i = 0; i < rowNum; ++i) {
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<scaleT>(sreg);
                GetPerGroupScale(scaleUb + i * scaleK, j * vecLen, para, config, scaleVreg);
                if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        tempSrcVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempSrcVreg, preg);
                } else {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Adds<scaleT, scaleT, MicroAPI::MaskMergeMode::ZEROING>(
                        srcVreg, srcVreg, static_cast<scaleT>(offset), preg);
                }
                TransRegForHif8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                StoreRes<dstT, scaleT>(dstUb + i * para.n + j * vecLen, dstVreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKColS8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const scaleT offset, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<srcT> tempVreg;
        for (uint16_t i = 0; i < rowNum; ++i) {
            uint32_t sreg = para.n;
            for (uint16_t j = 0; j < repeat; ++j) {
                preg = MicroAPI::UpdateMask<scaleT>(sreg);
                GetPerGroupScale(scaleUb + i * scaleK, j * vecLen, para, config, scaleVreg);
                if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        tempVreg, srcUb + i * para.n + j * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
                } else {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
                }
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Adds<scaleT, scaleT, MicroAPI::MaskMergeMode::ZEROING>(
                        srcVreg, srcVreg, static_cast<scaleT>(offset), preg);
                }
                TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                StoreRes<dstT, scaleT>(dstUb + i * para.n + j * vecLen, dstVreg, preg);
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const MicroAPI::CastTrait& castTrait>
__aicore__ inline void QuantPerGroupForKRowFp4OneRow(__local_mem__ dstT* dstAddr, __local_mem__ srcT* srcAddr,
    __local_mem__ scaleT* scaleAddr, MicroAPI::RegTensor<dstT>& dstVreg, MicroAPI::RegTensor<scaleT>& srcVreg,
    MicroAPI::RegTensor<scaleT>& scaleVreg, MicroAPI::RegTensor<srcT>& tempVreg, MicroAPI::MaskReg& preg,
    uint16_t repeat, uint32_t n, uint32_t vecLen)
{
    for (uint16_t j = 0; j < repeat; ++j) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleAddr + j * vecLen);
        preg = MicroAPI::UpdateMask<scaleT>(n);
        if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
            MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempVreg, srcAddr + j * vecLen);
            MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
        } else {
            MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcAddr + j * vecLen);
        }
        MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
        TransRegForFp4<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
            (__local_mem__ uint8_t*)dstAddr + (j * vecLen) / 2, (MicroAPI::RegTensor<uint8_t>&)dstVreg, preg);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowFp4(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint16_t tailRow = rowNum % para.groupSize;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<srcT> tempVreg;
        for (uint16_t i0 = 0; i0 < mainRowGroup; ++i0) {
            for (uint16_t i1 = 0; i1 < static_cast<uint16_t>(para.groupSize); ++i1) {
                QuantPerGroupForKRowFp4OneRow<dstT, srcT, scaleT, castTrait>(
                    dstUb + ((i0 * para.groupSize + i1) * para.n) / 2, srcUb + (i0 * para.groupSize + i1) * para.n,
                    scaleUb + i0 * para.n, dstVreg, srcVreg, scaleVreg, tempVreg, preg, repeat, para.n, vecLen);
            }
        }
        for (uint16_t i = 0; i < tailRow; ++i) {
            QuantPerGroupForKRowFp4OneRow<dstT, srcT, scaleT, castTrait>(
                dstUb + ((mainRowGroup * para.groupSize + i) * para.n) / 2,
                srcUb + (mainRowGroup * para.groupSize + i) * para.n, scaleUb + mainRowGroup * para.n, dstVreg, srcVreg,
                scaleVreg, tempVreg, preg, repeat, para.n, vecLen);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowFp8TailBlock(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb, uint16_t repeat, uint16_t tailRow, uint32_t n,
    uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<float> f32ScaleVreg;
    MicroAPI::RegTensor<float> f32OffsetVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::RegTensor<dstT> b8Vreg;
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            GetPerGroupKRowScaleEntry<scaleT>(scaleUb + j * vecLen, f32ScaleVreg);
            GetPerGroupKRowOffsetEntry<scaleT, config>(offsetUb + j * vecLen, f32OffsetVreg);
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            if constexpr (SupportType<srcT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, srcUb + i * n + j * vecLen);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32Vreg, srcVreg, preg);
            } else {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(f32Vreg, srcUb + i * n + j * vecLen);
            }
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32ScaleVreg, preg);
            if constexpr (config.hasOffset) {
                MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32OffsetVreg, preg);
            }
            TransRegForFp8<dstT, float, castTrait>(f32Vreg, b8Vreg, preg);
            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(dstUb + i * n + j * vecLen, b8Vreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint16_t tailRow = rowNum % para.groupSize;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<float> f32ScaleVreg;
        MicroAPI::RegTensor<float> f32OffsetVreg;
        MicroAPI::RegTensor<srcT> srcVreg;
        MicroAPI::RegTensor<float> f32Vreg;
        MicroAPI::RegTensor<dstT> b8Vreg;
        for (uint16_t i = 0; i < mainRowGroup; ++i) {
            for (uint16_t j = 0; j < static_cast<uint16_t>(para.groupSize); ++j) {
                uint32_t sreg = para.n;
                for (uint16_t k = 0; k < repeat; ++k) {
                    GetPerGroupKRowScaleEntry<scaleT>(scaleUb + i * para.n + k * vecLen, f32ScaleVreg);
                    GetPerGroupKRowOffsetEntry<scaleT, config>(offsetUb + i * para.n + k * vecLen, f32OffsetVreg);
                    preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                    if constexpr (SupportType<srcT, half, bfloat16_t>()) {
                        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            srcVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                        MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32Vreg, srcVreg, preg);
                    } else {
                        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                            f32Vreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                    }
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32ScaleVreg, preg);
                    if constexpr (config.hasOffset) {
                        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32OffsetVreg, preg);
                    }
                    TransRegForFp8<dstT, float, castTrait>(f32Vreg, b8Vreg, preg);
                    MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(
                        dstUb + (i * para.groupSize + j) * para.n + k * vecLen, b8Vreg, preg);
                }
            }
        }
        QuantPerGroupForKRowFp8TailBlock<dstT, srcT, scaleT, config>(dstUb + mainRowGroup * para.groupSize * para.n,
            srcUb + mainRowGroup * para.groupSize * para.n, scaleUb + mainRowGroup * para.n,
            offsetUb + mainRowGroup * para.n, repeat, tailRow, para.n, vecLen);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowHif8TailBlock(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb, uint16_t repeat, uint16_t tailRow, uint32_t n,
    uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> srcVreg;
    MicroAPI::RegTensor<srcT> tempSrcVreg;
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb + j * vecLen);
            if constexpr (config.hasOffset) {
                MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(offsetVreg, offsetUb + j * vecLen);
            }
            preg = MicroAPI::UpdateMask<scaleT>(sreg);
            if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempSrcVreg, srcUb + i * n + j * vecLen);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempSrcVreg, preg);
            } else {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * n + j * vecLen);
            }
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
            if constexpr (config.hasOffset) {
                MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, offsetVreg, preg);
            }
            TransRegForHif8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
            StoreRes<dstT, scaleT>(dstUb + i * n + j * vecLen, dstVreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowHif8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint16_t tailRow = rowNum % para.groupSize;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> offsetVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        MicroAPI::RegTensor<srcT> tempSrcVreg;
        for (uint16_t i = 0; i < mainRowGroup; ++i) {
            for (uint16_t j = 0; j < static_cast<uint16_t>(para.groupSize); ++j) {
                uint32_t sreg = para.n;
                for (uint16_t k = 0; k < repeat; ++k) {
                    LoadContinousScaleAndOffset<scaleT, config>(
                        scaleUb + i * para.n + k * vecLen, offsetUb + i * para.n + k * vecLen, scaleVreg, offsetVreg);
                    preg = MicroAPI::UpdateMask<scaleT>(sreg);
                    if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            tempSrcVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                        MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempSrcVreg, preg);
                    } else {
                        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(
                            srcVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                    }
                    MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                    if constexpr (config.hasOffset) {
                        MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, offsetVreg, preg);
                    }
                    TransRegForHif8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                    StoreRes<dstT, scaleT>(dstUb + (i * para.groupSize + j) * para.n + k * vecLen, dstVreg, preg);
                }
            }
        }
        QuantPerGroupForKRowHif8TailBlock<dstT, srcT, scaleT, config>(dstUb + mainRowGroup * para.groupSize * para.n,
            srcUb + mainRowGroup * para.groupSize * para.n, scaleUb + mainRowGroup * para.n,
            offsetUb + mainRowGroup * para.n, repeat, tailRow, para.n, vecLen);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowS8TailBlock(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb, uint16_t repeat, uint16_t tailRow, uint32_t n,
    uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> tempVreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> srcVreg;
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb + j * vecLen);
            if constexpr (config.hasOffset) {
                MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(offsetVreg, offsetUb + j * vecLen);
            }
            preg = MicroAPI::UpdateMask<scaleT>(sreg);
            if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempVreg, srcUb + i * n + j * vecLen);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
            } else {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * n + j * vecLen);
            }
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
            if constexpr (config.hasOffset) {
                MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, offsetVreg, preg);
            }
            TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
            StoreRes<dstT, scaleT>(dstUb + i * n + j * vecLen, dstVreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowS8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint16_t tailRow = rowNum % para.groupSize;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> offsetVreg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<srcT> tempVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        for (uint16_t i = 0; i < mainRowGroup; ++i) {
            for (uint16_t j = 0; j < static_cast<uint16_t>(para.groupSize); ++j) {
                uint32_t sreg = para.n;
                for (uint16_t k = 0; k < repeat; ++k) {
                    LoadContinousScaleAndOffset<scaleT, config>(
                        scaleUb + i * para.n + k * vecLen, offsetUb + i * para.n + k * vecLen, scaleVreg, offsetVreg);
                    preg = MicroAPI::UpdateMask<scaleT>(sreg);
                    if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            tempVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                        MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
                    } else {
                        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(
                            srcVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                    }
                    MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                    if constexpr (config.hasOffset) {
                        MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, offsetVreg, preg);
                    }
                    TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                    StoreRes<dstT, scaleT>(dstUb + (i * para.groupSize + j) * para.n + k * vecLen, dstVreg, preg);
                }
            }
        }
        QuantPerGroupForKRowS8TailBlock<dstT, srcT, scaleT, config>(dstUb + mainRowGroup * para.groupSize * para.n,
            srcUb + mainRowGroup * para.groupSize * para.n, scaleUb + mainRowGroup * para.n,
            offsetUb + mainRowGroup * para.n, repeat, tailRow, para.n, vecLen);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowFp8TailBlock(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, const scaleT& offset, uint16_t repeat, uint16_t tailRow, uint32_t n, uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> oriScaleVreg;
    MicroAPI::RegTensor<float> f32ScaleVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::RegTensor<dstT> b8Vreg;
    MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    float fp32_offset = ConvertToFloat<scaleT>(offset);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_UNPACK_B16>(oriScaleVreg, scaleUb + j * vecLen);
                MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32ScaleVreg, oriScaleVreg, b32FullPreg);
            } else {
                MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(f32ScaleVreg, scaleUb + j * vecLen);
            }
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadSrc<srcT>(srcUb + i * n + j * vecLen, preg, f32Vreg);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32ScaleVreg, preg);
            if constexpr (config.hasOffset) {
                MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, fp32_offset, preg);
            }
            TransRegForFp8<dstT, float, castTrait>(f32Vreg, b8Vreg, preg);
            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(dstUb + i * n + j * vecLen, b8Vreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const scaleT& offset, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint16_t tailRow = rowNum % para.groupSize;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint32_t sreg = para.n;
    float fp32_offset = ConvertToFloat<scaleT>(offset);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> oriScaleVreg;
        MicroAPI::RegTensor<float> f32ScaleVreg;
        MicroAPI::RegTensor<float> f32Vreg;
        MicroAPI::RegTensor<dstT> b8Vreg;
        MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < mainRowGroup; ++i) {
            for (uint16_t j = 0; j < static_cast<uint16_t>(para.groupSize); ++j) {
                sreg = para.n;
                for (uint16_t k = 0; k < repeat; ++k) {
                    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            oriScaleVreg, scaleUb + i * para.n + k * vecLen);
                        MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32ScaleVreg, oriScaleVreg, b32FullPreg);
                    } else {
                        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(
                            f32ScaleVreg, scaleUb + i * para.n + k * vecLen);
                    }
                    preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                    LoadSrc<srcT>(srcUb + (i * para.groupSize + j) * para.n + k * vecLen, preg, f32Vreg);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32ScaleVreg, preg);
                    if constexpr (config.hasOffset) {
                        MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(
                            f32Vreg, f32Vreg, fp32_offset, preg);
                    }
                    TransRegForFp8<dstT, float, castTrait>(f32Vreg, b8Vreg, preg);
                    MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK4_B32>(
                        dstUb + (i * para.groupSize + j) * para.n + k * vecLen, b8Vreg, preg);
                }
            }
        }
        QuantPerGroupForKRowFp8TailBlock<dstT, srcT, scaleT, config>(dstUb + mainRowGroup * para.groupSize * para.n,
            srcUb + mainRowGroup * para.groupSize * para.n, scaleUb + mainRowGroup * para.n, offset, repeat, tailRow,
            para.n, vecLen);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowHif8TailBlock(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, const scaleT& offset, uint16_t repeat, uint16_t tailRow, uint32_t n, uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> srcVreg;
    MicroAPI::RegTensor<srcT> tempVreg;
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb + j * vecLen);
            preg = MicroAPI::UpdateMask<scaleT>(sreg);
            if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempVreg, srcUb + i * n + j * vecLen);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
            } else {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * n + j * vecLen);
            }
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
            if constexpr (config.hasOffset) {
                MicroAPI::Adds<scaleT, scaleT, MicroAPI::MaskMergeMode::ZEROING>(
                    srcVreg, srcVreg, static_cast<scaleT>(offset), preg);
            }
            TransRegForHif8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
            StoreRes<dstT, scaleT>(dstUb + i * n + j * vecLen, dstVreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowHif8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const scaleT& offset, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint16_t tailRow = rowNum % para.groupSize;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        MicroAPI::RegTensor<srcT> tempVreg;
        for (uint16_t i = 0; i < mainRowGroup; ++i) {
            for (uint16_t j = 0; j < static_cast<uint16_t>(para.groupSize); ++j) {
                uint32_t sreg = para.n;
                for (uint16_t k = 0; k < repeat; ++k) {
                    MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(
                        scaleVreg, scaleUb + i * para.n + k * vecLen);
                    preg = MicroAPI::UpdateMask<scaleT>(sreg);
                    if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            tempVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                        MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
                    } else {
                        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(
                            srcVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                    }
                    MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                    if constexpr (config.hasOffset) {
                        MicroAPI::Adds<scaleT, scaleT, MicroAPI::MaskMergeMode::ZEROING>(
                            srcVreg, srcVreg, static_cast<scaleT>(offset), preg);
                    }
                    TransRegForHif8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                    StoreRes<dstT, scaleT>(dstUb + (i * para.groupSize + j) * para.n + k * vecLen, dstVreg, preg);
                }
            }
        }
        QuantPerGroupForKRowHif8TailBlock<dstT, srcT, scaleT, config>(dstUb + mainRowGroup * para.groupSize * para.n,
            srcUb + mainRowGroup * para.groupSize * para.n, scaleUb + mainRowGroup * para.n, offset, repeat, tailRow,
            para.n, vecLen);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowS8TailBlock(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, const scaleT& offset, uint16_t repeat, uint16_t tailRow, uint32_t n, uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> tempVreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> srcVreg;
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb + j * vecLen);
            preg = MicroAPI::UpdateMask<scaleT>(sreg);
            if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempVreg, srcUb + i * n + j * vecLen);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
            } else {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * n + j * vecLen);
            }
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
            if constexpr (config.hasOffset) {
                MicroAPI::Adds<scaleT, scaleT, MicroAPI::MaskMergeMode::ZEROING>(
                    srcVreg, srcVreg, static_cast<scaleT>(offset), preg);
            }
            TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
            StoreRes<dstT, scaleT>(dstUb + i * n + j * vecLen, dstVreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowS8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const scaleT offset, const AscendQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint16_t tailRow = rowNum % para.groupSize;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<scaleT> scaleVreg;
        MicroAPI::RegTensor<srcT> tempVreg;
        MicroAPI::RegTensor<dstT> dstVreg;
        MicroAPI::RegTensor<scaleT> srcVreg;
        for (uint16_t i = 0; i < mainRowGroup; ++i) {
            for (uint16_t j = 0; j < static_cast<uint16_t>(para.groupSize); ++j) {
                uint32_t sreg = para.n;
                for (uint16_t k = 0; k < repeat; ++k) {
                    MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(
                        scaleVreg, scaleUb + i * para.n + k * vecLen);
                    preg = MicroAPI::UpdateMask<scaleT>(sreg);
                    if constexpr (SupportType<srcT, half, bfloat16_t>() && SupportType<scaleT, float>()) {
                        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            tempVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                        MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
                    } else {
                        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(
                            srcVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                    }
                    MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                    if constexpr (config.hasOffset) {
                        MicroAPI::Adds<scaleT, scaleT, MicroAPI::MaskMergeMode::ZEROING>(
                            srcVreg, srcVreg, static_cast<scaleT>(offset), preg);
                    }
                    TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                    StoreRes<dstT, scaleT>(dstUb + (i * para.groupSize + j) * para.n + k * vecLen, dstVreg, preg);
                }
            }
        }
        QuantPerGroupForKRowS8TailBlock<dstT, srcT, scaleT, config>(dstUb + mainRowGroup * para.groupSize * para.n,
            srcUb + mainRowGroup * para.groupSize * para.n, scaleUb + mainRowGroup * para.n, offset, repeat, tailRow,
            para.n, vecLen);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void AscendQuantPerToken(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
    const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    if constexpr (SupportType<dstT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        QuantPerTokenForFp8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else if constexpr (SupportType<dstT, hifloat8_t>()) {
        QuantPerTokenForHif8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else if constexpr (SupportType<dstT, int8_t>()) {
        QuantPerTokenForS8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for AscendQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void AscendQuantPerToken(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor, const scaleT offset,
    const AscendQuantParam& para)
{
    if constexpr (SupportType<dstT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        QuantPerTokenForFp8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offset, para);
    } else if constexpr (SupportType<dstT, hifloat8_t>()) {
        QuantPerTokenForHif8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offset, para);
    } else if constexpr (SupportType<dstT, int8_t>()) {
        QuantPerTokenForS8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offset, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for AscendQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config>
__aicore__ inline void AscendQuantPerGroupForKCol(const LocalTensor<dstT>& dstTensor,
    const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    if constexpr (SupportType<dstT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        QuantPerGroupForKColFp8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else if constexpr (SupportType<dstT, hifloat8_t>()) {
        QuantPerGroupForKColHif8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else if constexpr (SupportType<dstT, int8_t>()) {
        QuantPerGroupForKColS8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else if constexpr (SupportType<dstT, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
        // fp4 doesn't count offset
        QuantPerGroupForKColFp4<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for AscendQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config>
__aicore__ inline void AscendQuantPerGroupForKCol(const LocalTensor<dstT>& dstTensor,
    const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LocalTensor<scaleT>& scaleTensor, const scaleT offset, const AscendQuantParam& para)
{
    if constexpr (SupportType<dstT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        QuantPerGroupForKColFp8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offset, para);
    } else if constexpr (SupportType<dstT, hifloat8_t>()) {
        QuantPerGroupForKColHif8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offset, para);
    } else if constexpr (SupportType<dstT, int8_t>()) {
        QuantPerGroupForKColS8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offset, para);
    } else if constexpr (SupportType<dstT, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
        // fp4 doesn't count offset
        QuantPerGroupForKColFp4<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for AscendQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config>
__aicore__ inline void AscendQuantPerGroupForKRow(const LocalTensor<dstT>& dstTensor,
    const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    if constexpr (SupportType<dstT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        QuantPerGroupForKRowFp8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else if constexpr (SupportType<dstT, hifloat8_t>()) {
        QuantPerGroupForKRowHif8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else if constexpr (SupportType<dstT, int8_t>()) {
        QuantPerGroupForKRowS8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else if constexpr (SupportType<dstT, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
        // fp4 doesn't count offset
        QuantPerGroupForKRowFp4<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for AscendQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config>
__aicore__ inline void AscendQuantPerGroupForKRow(const LocalTensor<dstT>& dstTensor,
    const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LocalTensor<scaleT>& scaleTensor, const scaleT offset, const AscendQuantParam& para)
{
    if constexpr (SupportType<dstT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        QuantPerGroupForKRowFp8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offset, para);
    } else if constexpr (SupportType<dstT, hifloat8_t>()) {
        QuantPerGroupForKRowHif8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offset, para);
    } else if constexpr (SupportType<dstT, int8_t>()) {
        QuantPerGroupForKRowS8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offset, para);
    } else if constexpr (SupportType<dstT, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
        // fp4 doesn't count offset
        QuantPerGroupForKRowFp4<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for AscendQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config,
    const AscendQuantPolicy& policy>
__aicore__ inline void AscendQuantImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
    const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(
        SupportType<srcT, half, float, bfloat16_t>(), "AscendQuant only support half/float/bfloat16_t input dtype");
    static_assert(
        SupportType<scaleT, half, float, bfloat16_t>(), "AscendQuant only support half/float/bfloat16_t scale dtype");
    static_assert((policy == AscendQuantPolicy::PER_TOKEN || policy == AscendQuantPolicy::PER_GROUP),
        "unsupported policy for AscendQuant in current device!");
    ASCENDC_ASSERT(
        (para.calCount <= srcTensor.GetSize() && para.calCount <= dstTensor.GetSize() && para.calCount >= 0), {
            KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]", para.calCount,
                srcTensor.GetSize(), dstTensor.GetSize());
        });
    ASCENDC_ASSERT(
        (para.calCount % para.n == 0), { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of n!"); });
    if constexpr (policy == AscendQuantPolicy::PER_TOKEN) {
        static_assert(SupportType<dstT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t>(),
            "AscendQuant PerToken only support int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t output dtype");
        AscendQuantPerToken<dstT, srcT, scaleT, config>(
            dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
    } else if constexpr (policy == AscendQuantPolicy::PER_GROUP) {
        static_assert(SupportType<dstT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t, fp4x2_e2m1_t, fp4x2_e1m2_t>(),
            "AscendQuant PerGroup only support "
            "int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t/fp4x2_e2m1_t/fp4x2_e1m2_t output dtype");
        static_assert(
            ((config.kDim == 1) || (config.kDim == 0)), "AscendAntiQuant PerGroup only support K is axis 0/1!");
        ASCENDC_ASSERT((para.groupSize > 0 && para.groupSize % 32 == 0),
            { KERNEL_LOG(KERNEL_ERROR, "groupSize must be an integer multiple of 32 and greater than 0 !"); });
        if constexpr (config.kDim == 1) {
            AscendQuantPerGroupForKCol<dstT, srcT, scaleT, isReuseSource, config>(
                dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
        } else {
            AscendQuantPerGroupForKRow<dstT, srcT, scaleT, isReuseSource, config>(
                dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config,
    const AscendQuantPolicy& policy>
__aicore__ inline void AscendQuantImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor, const scaleT offset,
    const AscendQuantParam& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(
        SupportType<srcT, half, float, bfloat16_t>(), "AscendQuant only support half/float/bfloat16_t input dtype");
    static_assert(
        SupportType<scaleT, half, float, bfloat16_t>(), "AscendQuant only support half/float/bfloat16_t scale dtype");
    static_assert((policy == AscendQuantPolicy::PER_TOKEN || policy == AscendQuantPolicy::PER_GROUP),
        "unsupported policy for AscendQuant in current device!");
    ASCENDC_ASSERT(
        (para.calCount <= srcTensor.GetSize() && para.calCount <= dstTensor.GetSize() && para.calCount >= 0), {
            KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]", para.calCount,
                srcTensor.GetSize(), dstTensor.GetSize());
        });
    ASCENDC_ASSERT(
        (para.calCount % para.n == 0), { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of n!"); });
    if constexpr (policy == AscendQuantPolicy::PER_TOKEN) {
        static_assert(SupportType<dstT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t>(),
            "AscendQuant PerToken only support int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t output dtype");
        AscendQuantPerToken<dstT, srcT, scaleT, config>(
            dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset, para);
    } else if constexpr (policy == AscendQuantPolicy::PER_GROUP) {
        static_assert(SupportType<dstT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t, fp4x2_e2m1_t, fp4x2_e1m2_t>(),
            "AscendQuant PerGroup only support "
            "int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t/fp4x2_e2m1_t/fp4x2_e1m2_t output dtype");
        static_assert(
            ((config.kDim == 1) || (config.kDim == 0)), "AscendAntiQuant PerGroup only support K is axis 0/1!");
        ASCENDC_ASSERT((para.groupSize > 0 && para.groupSize % 32 == 0),
            { KERNEL_LOG(KERNEL_ERROR, "groupSize must be an integer multiple of 32 and greater than 0 !"); });
        if constexpr (config.kDim == 1) {
            AscendQuantPerGroupForKCol<dstT, srcT, scaleT, isReuseSource, config>(
                dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset, para);
        } else {
            AscendQuantPerGroupForKRow<dstT, srcT, scaleT, isReuseSource, config>(
                dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset, para);
        }
    }
}
} //  namespace AscendC
#endif // AICORE_ADV_API_DETAIL_QUANTIZATION_QUANT_ASCEND_QUANT_C310_IMPL_H
