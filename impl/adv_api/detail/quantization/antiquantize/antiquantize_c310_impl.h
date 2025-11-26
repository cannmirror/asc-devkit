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
 * \file antiquantize_c310_impl.h
 * \brief
 */
#ifndef IMPL_QUANTIZATION_ANTIQUANTIZE_C310_IMPL_H
#define IMPL_QUANTIZATION_ANTIQUANTIZE_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "antiquantize_common.h"
#include "../antiquant/ascend_antiquant_c310_impl.h"

namespace AscendC {
namespace AntiQuantizeUtils {
template <typename SrcT>
__simd_callee__ inline void CastAntiQuantizeSrcToFp32(MicroAPI::RegTensor<float>& f32SrcVreg, MicroAPI::RegTensor<SrcT>& srcVreg,
    MicroAPI::RegTensor<half>& f16Vreg, MicroAPI::MaskReg& preg)
{
    if constexpr (SupportType<SrcT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        MicroAPI::Cast<float, SrcT, layoutZMrgZ>(f32SrcVreg, srcVreg, preg);
    } else {
        MicroAPI::Cast<half, SrcT, layoutZMrgZ>(f16Vreg, srcVreg, preg);
        MicroAPI::Cast<float, half, layoutZMrgZ>(f32SrcVreg, f16Vreg, preg);
    }
}
}

template <typename dstT, typename SrcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantizePerGroupForColCommonVF(__ubuf__ dstT* dstUb, __ubuf__ SrcT* srcUb,
    __ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb, const AscendAntiQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint32_t sreg = para.n;
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<SrcT> srcVreg;
    MicroAPI::RegTensor<half> f16Vreg;
    MicroAPI::RegTensor<float> f32SrcVreg;
    MicroAPI::RegTensor<float> f32ScaleVreg;
    MicroAPI::RegTensor<float> f32OffsetVreg;
    for (uint16_t i = 0; i < rowNum; ++i) {
        sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            GetPerGroupScaleAndOffset<scaleT, config>(scaleUb + i * scaleK, offsetUb + i * scaleK,
                j * vecLen, para, scaleVreg, offsetVreg);
            MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(srcVreg, srcUb + i * para.n + j * vecLen);
            AntiQuantizeUtils::CastAntiQuantizeSrcToFp32<SrcT>(f32SrcVreg, srcVreg, f16Vreg, preg);
            ConvertToF32ScaleAndOffset<scaleT, config>(scaleVreg, offsetVreg, preg, f32ScaleVreg, f32OffsetVreg);
            if constexpr (SupportType<scaleT, float>()) {
                AddOffsetIfExist<float, config>(f32SrcVreg, offsetVreg, preg);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32SrcVreg, f32SrcVreg, scaleVreg, preg);
            } else {
                ConvertToF32ScaleAndOffset<scaleT, config>(scaleVreg, offsetVreg, preg, f32ScaleVreg, f32OffsetVreg);
                AddOffsetIfExist<float, config>(f32SrcVreg, f32OffsetVreg, preg);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32SrcVreg, f32SrcVreg, f32ScaleVreg, preg);
            }
            StoreF32Res<dstT>((dstUb + i * para.n + j * vecLen), f32SrcVreg, preg);
        }
    }
}
/* **************************************************************************************************
 * PER_GROUP for B8                                             *
 * ************************************************************************************************* */
template <typename dstT, typename SrcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantizePerGroupForColCommon(const LocalTensor<dstT>& dstTensor, const LocalTensor<SrcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ SrcT* srcUb = (__ubuf__ SrcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    __ubuf__ scaleT* offsetUb = (__ubuf__ scaleT*)offsetTensor.GetPhyAddr();
    AntiQuantizePerGroupForColCommonVF<dstT, SrcT, scaleT, config>(dstUb, srcUb,
        scaleUb, offsetUb, para);
}

template <typename dstT, typename SrcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_callee__ inline void AntiQuantizePerGroupForRowCommonTailBlock(__ubuf__ dstT* dstUb, __ubuf__ SrcT* srcUb,
    __ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb, uint16_t repeat, uint16_t tailRow, uint32_t n,
    uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<SrcT> srcVreg;
    MicroAPI::RegTensor<float> f32ScaleVreg;
    MicroAPI::RegTensor<float> f32OffsetVreg;
    MicroAPI::RegTensor<float> f32SrcVreg;
    MicroAPI::RegTensor<half> f16Vreg;
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            LoadNormScaleAndOffset<scaleT, config>((scaleUb + j * vecLen),
                (offsetUb + j * vecLen), scaleVreg, offsetVreg);
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                srcVreg, srcUb + i * n + j * vecLen);
            AntiQuantizeUtils::CastAntiQuantizeSrcToFp32<SrcT>(f32SrcVreg, srcVreg, f16Vreg, preg);
            if constexpr (SupportType<scaleT, float>()) {
                AddOffsetIfExist<float, config>(f32SrcVreg, offsetVreg, preg);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32SrcVreg, f32SrcVreg, scaleVreg, preg);
            } else {
                ConvertToF32ScaleAndOffset<scaleT, config>(scaleVreg, offsetVreg, preg,
                    f32ScaleVreg, f32OffsetVreg);
                AddOffsetIfExist<float, config>(f32SrcVreg, f32OffsetVreg, preg);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32SrcVreg, f32SrcVreg, f32ScaleVreg, preg);
            }
            StoreF32Res<dstT>((dstUb + i * n + j * vecLen), f32SrcVreg, preg);
        }
    }
}

template <typename dstT, typename SrcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantizePerGroupForRowCommonVF(__ubuf__ dstT* dstUb, __ubuf__ SrcT* srcUb,
    __ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb, const AscendAntiQuantParam para,
    uint16_t rowNum, uint16_t tailRow)
{
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<SrcT> srcVreg;
    MicroAPI::RegTensor<float> f32Svreg;
    MicroAPI::RegTensor<float> f32Ovreg;
    MicroAPI::RegTensor<float> f32SrcVreg;
    MicroAPI::RegTensor<half> f16Vreg;
    for (uint16_t i = 0; i < mainRowGroup; ++i) {
        for (uint16_t j = 0; j < static_cast<uint16_t>(para.groupSize); ++j) {
            uint32_t sreg = para.n;
            for (uint16_t k = 0; k < repeat; ++k) {
                preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                LoadNormScaleAndOffset<scaleT, config>((scaleUb + i * para.n + k * vecLen),
                    (offsetUb + i * para.n + k * vecLen), scaleVreg, offsetVreg);
                MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    srcVreg, (srcUb + (i * para.groupSize + j) * para.n + k * vecLen));
                AntiQuantizeUtils::CastAntiQuantizeSrcToFp32<SrcT>(f32SrcVreg, srcVreg, f16Vreg, preg);
                if constexpr (SupportType<scaleT, float>()) {
                    AddOffsetIfExist<float, config>(f32SrcVreg, offsetVreg, preg);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32SrcVreg, f32SrcVreg, scaleVreg, preg);
                } else {
                    ConvertToF32ScaleAndOffset<scaleT, config>(scaleVreg, offsetVreg, preg, f32Svreg, f32Ovreg);
                    AddOffsetIfExist<float, config>(f32SrcVreg, f32Ovreg, preg);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32SrcVreg, f32SrcVreg, f32Svreg, preg);
                }
                StoreF32Res<dstT>((dstUb + (i * para.groupSize + j) * para.n + k * vecLen), f32SrcVreg, preg);
            }
        }
    }
    AntiQuantizePerGroupForRowCommonTailBlock<dstT, SrcT, scaleT, config>(
        dstUb + mainRowGroup * para.groupSize * para.n, srcUb + mainRowGroup * para.groupSize * para.n,
        scaleUb + mainRowGroup * para.n, offsetUb + mainRowGroup * para.n, repeat, tailRow, para.n, vecLen);
}

template <typename dstT, typename SrcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantizePerGroupForRowCommon(const LocalTensor<dstT>& dstTensor, const LocalTensor<SrcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ SrcT* srcUb = (__ubuf__ SrcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    __ubuf__ scaleT* offsetUb = (__ubuf__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t tailRow = rowNum % para.groupSize;
    AntiQuantizePerGroupForRowCommonVF<dstT, SrcT, scaleT, config>(dstUb, srcUb,
        scaleUb, offsetUb, para, rowNum, tailRow);
}

template <typename dstT, typename SrcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantizePerTokenCommonVF(__ubuf__ dstT* dstUb, __ubuf__ SrcT* srcUb,
    __ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb, const AscendAntiQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint32_t sreg = para.n;

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<SrcT> srcVreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::RegTensor<float> f32ScaleVreg;
    MicroAPI::RegTensor<float> f32OffsetVreg;
    MicroAPI::RegTensor<half> f16Vreg;
    for (uint16_t i = 0; i < rowNum; ++i) {
        LoadPerTokenScaleAndOffset<scaleT, config>(scaleUb + i, offsetUb + i, scaleVreg, offsetVreg);
        sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                srcVreg, srcUb + i * para.n + j * vecLen);
            AntiQuantizeUtils::CastAntiQuantizeSrcToFp32<SrcT>(f32Vreg, srcVreg, f16Vreg, preg);
            
            if constexpr (SupportType<scaleT, float>()) {
                AddOffsetIfExist<float, config>(f32Vreg, offsetVreg, preg);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, scaleVreg, preg);
            } else {
                ConvertToF32ScaleAndOffset<scaleT, config>(
                    scaleVreg, offsetVreg, preg, f32ScaleVreg, f32OffsetVreg);
                AddOffsetIfExist<float, config>(f32Vreg, f32OffsetVreg, preg);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32ScaleVreg, preg);
            }
            StoreF32Res<dstT>((dstUb + i * para.n + j * vecLen), f32Vreg, preg);
        }
    }
}

/* **************************************************************************************************
 * PER_TOKEN for B8                                             *
 * ************************************************************************************************* */
template <typename dstT, typename SrcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantizePerTokenCommon(const LocalTensor<dstT>& dstTensor,
    const LocalTensor<SrcT>& srcTensor, const LocalTensor<scaleT>& scaleTensor,
    const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ SrcT* srcUb = (__ubuf__ SrcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    __ubuf__ scaleT* offsetUb = (__ubuf__ scaleT*)offsetTensor.GetPhyAddr();
    AntiQuantizePerTokenCommonVF<dstT, SrcT, scaleT, config>(dstUb, srcUb,
        scaleUb, offsetUb, para);
}

/* **************************************************************************************************
 * PER_TENSOR for B8                                             *
 * ************************************************************************************************* */
template <bool hasOffset, typename SrcT, typename DstT>
__simd_vf__ inline void PerTensorProcessCommon(__ubuf__ DstT* dst, __ubuf__ SrcT* src,
    const DstT offset, const DstT scale, const uint32_t srcCalCount)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SrcT> vreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::RegTensor<DstT> outReg;
    MicroAPI::RegTensor<half> f16Vreg;

    uint32_t sregLower = (uint32_t)ANTIQUANT_B32_VF_LEN;
    uint32_t sreg = (uint32_t)srcCalCount;
    uint16_t repeat = CeilDivision(srcCalCount, sregLower);

    for (uint16_t i = 0; i < (uint16_t)repeat; ++i) {
        preg = MicroAPI::UpdateMask<uint32_t>(sreg);
        MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vreg, src + i * sregLower);
        AntiQuantizeUtils::CastAntiQuantizeSrcToFp32<SrcT>(f32Vreg, vreg, f16Vreg, preg);
        if constexpr (SupportType<DstT, bfloat16_t>()) {
            if constexpr (hasOffset) {
                MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, ToFloat(offset), preg);
            }
            MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, ToFloat(scale), preg);
        } else {
            if constexpr (hasOffset) {
                MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, static_cast<float>(offset), preg);
            }
            MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, static_cast<float>(scale), preg);
        }

        MicroAPI::Cast<DstT, float, LayoutZMrgZRndRSatS>(outReg, f32Vreg, preg);
        MicroAPI::DataCopy<DstT, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * sregLower, outReg, preg);
    }
}

template <const AntiQuantizeConfig& config, typename DstT, typename SrcT>
__aicore__ inline void AntiQuantizePerTensor(const LocalTensor<DstT>& dst,
    const LocalTensor<SrcT>& src, const DstT offset, const DstT scale,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const AntiQuantizeParams& params)
{
    static_assert(SupportType<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t, int8_t>(),
        "This AntiQuantize only support fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t/int8_t input dtype");
    static_assert(SupportType<DstT, half, bfloat16_t>(),
        "This AntiQuantize only support half/bfloat16_t output dtype");
    __ubuf__ DstT* dstUb = (__ubuf__ DstT*)dst.GetPhyAddr();
    __ubuf__ SrcT* srcUb = (__ubuf__ SrcT*)src.GetPhyAddr();
    auto tmpbuffer = sharedTmpBuffer.ReinterpretCast<DstT>();
    __ubuf__ DstT* tmpbufferUb = (__ubuf__ DstT*)tmpbuffer.GetPhyAddr();

    PerTensorProcessCommon<config.hasOffset, SrcT, DstT>(dstUb, srcUb, offset, scale, params.m * params.n);
}

/* **************************************************************************************************
 * PER_CHANNEL for B8                                             *
 * ************************************************************************************************* */
template <bool hasOffset, typename SrcT, typename DstT>
__simd_vf__ inline void PerChannelNoTransposeCommon(__ubuf__ DstT* dst, __ubuf__ SrcT* src,
    __ubuf__ DstT* offset, __ubuf__ DstT* scale, const uint32_t M, const uint32_t N)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SrcT> vreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::RegTensor<DstT> outReg;
    MicroAPI::RegTensor<DstT> scaleB16Vreg;
    MicroAPI::RegTensor<DstT> offsetB16Vreg;
    MicroAPI::RegTensor<float> scaleB32Vreg;
    MicroAPI::RegTensor<float> offsetB32Vreg;
    MicroAPI::RegTensor<half> f16Vreg;
    uint32_t sregLower = ANTIQUANT_B32_VF_LEN;
    uint32_t sreg = N;
    uint16_t repeat = CeilDivision(N, sregLower);

    for (uint16_t i = 0; i < repeat; ++i) {
        preg = MicroAPI::UpdateMask<uint32_t>(sreg);
        // load offset and scale ,then cast to float to add &&mul
        if constexpr (hasOffset) {
            MicroAPI::DataCopy<DstT, MicroAPI::LoadDist::DIST_UNPACK_B16>(offsetB16Vreg, offset + i * sregLower);
            MicroAPI::Cast<float, DstT, layoutZMrgZ>(offsetB32Vreg, offsetB16Vreg, preg); // b16->fp32
        }
        MicroAPI::DataCopy<DstT, MicroAPI::LoadDist::DIST_UNPACK_B16>(scaleB16Vreg, scale + i * sregLower);
        MicroAPI::Cast<float, DstT, layoutZMrgZ>(scaleB32Vreg, scaleB16Vreg, preg);   // b16->fp32

        for (uint16_t j = 0; j < (uint16_t)M; ++j) {
            MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vreg, src + j * N + i * sregLower);
            AntiQuantizeUtils::CastAntiQuantizeSrcToFp32<SrcT>(f32Vreg, vreg, f16Vreg, preg);
            if constexpr (hasOffset) {
                MicroAPI::Add(f32Vreg, f32Vreg, offsetB32Vreg, preg);
            }
            MicroAPI::Mul(f32Vreg, f32Vreg, scaleB32Vreg, preg);

            MicroAPI::Cast<DstT, float, LayoutZMrgZRndRSatS>(outReg, f32Vreg, preg);
            MicroAPI::DataCopy<DstT, MicroAPI::StoreDist::DIST_PACK_B32>(dst + j * N + i * sregLower, outReg,
                preg);
        }
    }
}

template <bool hasOffset, typename SrcT, typename DstT>
__aicore__ inline void AntiQuantPerChannelNoTranspose(const LocalTensor<DstT>& dst,
    const LocalTensor<SrcT>& src, const LocalTensor<DstT>& offset,
    const LocalTensor<DstT>& scale, const uint32_t M, const uint32_t N)
{
    __ubuf__ DstT* scaleUb = (__ubuf__ DstT*)scale.GetPhyAddr();
    __ubuf__ DstT* offsetUb = (__ubuf__ DstT*)offset.GetPhyAddr();
    __ubuf__ DstT* dstUb = (__ubuf__ DstT*)dst.GetPhyAddr();
    __ubuf__ SrcT* srcUb = (__ubuf__ SrcT*)src.GetPhyAddr();
    PerChannelNoTransposeCommon<hasOffset, SrcT, DstT>(dstUb, srcUb, offsetUb, scaleUb, M, N);
}

/* **************************************************************************************************
 * PER_CHANNEL for B8                                             *
 * ************************************************************************************************* */
template <const AntiQuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void AntiQuantizePerChannel(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor,
    const ScaleT& scale, const OffsetT& offset, const LocalTensor<uint8_t>& sharedTmpBuffer, const AntiQuantizeParams& params)
{
    static_assert(SupportType<DstT, half, bfloat16_t>(),
        "This AntiQuantize only support half/bfloat16_6 output dtype");

    AntiQuantPerChannelNoTranspose<config.hasOffset, SrcT, DstT>(dstTensor, srcTensor, offset, scale, params.m, params.n);
}
}  // namespace AscendC
#endif  // IMPL_QUANTIZATION_ANTIQUANTIZE_C310_IMPL_H
