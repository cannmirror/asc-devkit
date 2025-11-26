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
 * \file quantize_impl.h
 * \brief
 */
#ifndef IMPL_QUANTIZATION_QUANTIZE_QUANTIZE_IMPL_H
#define IMPL_QUANTIZATION_QUANTIZE_QUANTIZE_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "include/adv_api/quantization/quantize_utils.h"

namespace AscendC {
namespace QuantizeUtils {
template <typename DstT, typename SrcT, typename ScaleT>
__aicore__ inline void CheckApiDtypeValid()
{
    static_assert(SupportType<Tuple<SrcT, ScaleT, DstT>,
        Tuple<half, half, fp8_e4m3fn_t>, Tuple<half, half, fp8_e5m2_t>,
        Tuple<bfloat16_t, bfloat16_t, fp8_e4m3fn_t>, Tuple<bfloat16_t, bfloat16_t, fp8_e5m2_t>,
        Tuple<float, float, fp8_e4m3fn_t>, Tuple<float, float, fp8_e5m2_t>,
        Tuple<half, float, fp8_e4m3fn_t>, Tuple<half, float, fp8_e5m2_t>,
        Tuple<bfloat16_t, float, fp8_e4m3fn_t>, Tuple<bfloat16_t, float, fp8_e5m2_t>,
        Tuple<half, half, hifloat8_t>, Tuple<half, half, int8_t>,
        Tuple<bfloat16_t, bfloat16_t, hifloat8_t>, Tuple<bfloat16_t, bfloat16_t, int8_t>,
        Tuple<float, float, hifloat8_t>, Tuple<float, float, int8_t>,
        Tuple<half, float, hifloat8_t>, Tuple<half, float, int8_t>,
        Tuple<bfloat16_t, float, hifloat8_t>, Tuple<bfloat16_t, float, int8_t>,
        Tuple<half, half, fp4x2_e1m2_t>, Tuple<half, half, fp4x2_e2m1_t>,
        Tuple<bfloat16_t, bfloat16_t, fp4x2_e1m2_t>, Tuple<bfloat16_t, bfloat16_t, fp4x2_e2m1_t>,
        Tuple<float, float, fp4x2_e1m2_t>, Tuple<float, float, fp4x2_e2m1_t>,
        Tuple<half, float, fp4x2_e1m2_t>, Tuple<half, float, fp4x2_e2m1_t>,
        Tuple<bfloat16_t, float, fp4x2_e1m2_t>, Tuple<bfloat16_t, float, fp4x2_e2m1_t>>(),
        "Failed to check data type for Quantize"
    );
}

template <typename DstT, typename ScaleT>
__simd_callee__ inline void StoreRes(__ubuf__ DstT* dstAddr, MicroAPI::RegTensor<DstT>& vreg,
                                MicroAPI::MaskReg& mask)
{
    if (SupportType<ScaleT, float>()) {
        MicroAPI::DataCopy<DstT, MicroAPI::StoreDist::DIST_PACK4_B32>(dstAddr, vreg, mask);
    } else {
        MicroAPI::DataCopy<DstT, MicroAPI::StoreDist::DIST_PACK_B16>(dstAddr, vreg, mask);
    }
}

template <typename DstT, typename ScaleT, const MicroAPI::CastTrait &castTrait>
__simd_callee__ inline void TransRegForFp8(
    MicroAPI::RegTensor<ScaleT> &srcVreg, MicroAPI::RegTensor<DstT> &dstVreg, MicroAPI::MaskReg &mask)
{
    if constexpr (castTrait.roundMode == RoundMode::CAST_RINT) {
        MicroAPI::Cast<DstT, ScaleT, castTrait>(dstVreg, srcVreg, mask);
    } else {
        MicroAPI::Cast<DstT, ScaleT, LayoutZMrgZRndRSatS>(dstVreg, srcVreg, mask);
    }
}

template <typename DstT, typename ScaleT, const MicroAPI::CastTrait& castTrait>
__simd_callee__ inline void TransRegForHif8(MicroAPI::RegTensor<ScaleT>& srcVreg, MicroAPI::RegTensor<DstT>& dstVreg,
                                       MicroAPI::MaskReg& mask)
{
    if constexpr (SupportType<ScaleT, bfloat16_t>()) {
        // bf16->fp32->hif8
        MicroAPI::MaskReg mask1;
        MicroAPI::MaskReg mask2 = MicroAPI::CreateMask<ScaleT, MicroAPI::MaskPattern::ALLF>();
        MicroAPI::RegTensor<float> f32Vreg;
        MicroAPI::RegTensor<float> f32Vreg2;
        MicroAPI::RegTensor<ScaleT> srcVreg2;
        MicroAPI::RegTensor<DstT> dstVreg2;
        MicroAPI::MaskInterleave<ScaleT>(mask1, mask2, mask, mask2);
        MicroAPI::Interleave(srcVreg, srcVreg2, srcVreg, srcVreg2);
        MicroAPI::Cast<float, ScaleT, layoutZMrgZ>(f32Vreg, srcVreg, mask1);
        MicroAPI::Cast<float, ScaleT, layoutZMrgZ>(f32Vreg2, srcVreg2, mask2);
        if constexpr (castTrait.roundMode == RoundMode::CAST_ROUND || castTrait.roundMode == RoundMode::CAST_HYBRID) {
            MicroAPI::Cast<DstT, float, castTrait>(dstVreg, f32Vreg, mask1);
            MicroAPI::Cast<DstT, float, castTrait>(dstVreg2, f32Vreg2, mask2);
        } else {
            MicroAPI::Cast<DstT, float, LayoutZMrgZRndASatS>(dstVreg, f32Vreg, mask1);
            MicroAPI::Cast<DstT, float, LayoutZMrgZRndASatS>(dstVreg2, f32Vreg2, mask2);
        }
        MicroAPI::DeInterleave((MicroAPI::RegTensor<ScaleT> &)dstVreg, (MicroAPI::RegTensor<ScaleT> &)dstVreg2,
            (MicroAPI::RegTensor<ScaleT> &)dstVreg, (MicroAPI::RegTensor<ScaleT> &)dstVreg2);
    } else if constexpr (SupportType<ScaleT, half, float>()) {
        if constexpr (castTrait.roundMode == RoundMode::CAST_ROUND || castTrait.roundMode == RoundMode::CAST_HYBRID) {
            MicroAPI::Cast<DstT, ScaleT, castTrait>(dstVreg, srcVreg, mask);
        } else {
            MicroAPI::Cast<DstT, ScaleT, LayoutZMrgZRndASatS>(dstVreg, srcVreg, mask);
        }
    }
}

template <typename DstT, typename ScaleT, const MicroAPI::CastTrait& castTrait>
__simd_callee__ inline void TransRegForS8(MicroAPI::RegTensor<ScaleT>& srcVreg, MicroAPI::RegTensor<DstT>& dstVreg,
                                     MicroAPI::MaskReg& mask)
{
    if constexpr (SupportType<ScaleT, bfloat16_t>()) {
        // bf16->fp32->s16->fp16->s8
        MicroAPI::MaskReg mask1;
        MicroAPI::MaskReg mask2 = MicroAPI::CreateMask<ScaleT, MicroAPI::MaskPattern::ALLF>();
        MicroAPI::RegTensor<float> f32Vreg;
        MicroAPI::RegTensor<float> f32Vreg2;
        MicroAPI::RegTensor<ScaleT> srcVreg2;
        MicroAPI::RegTensor<DstT> dstVreg2;
        MicroAPI::MaskInterleave<ScaleT>(mask1, mask2, mask, mask2);
        MicroAPI::Interleave(srcVreg, srcVreg2, srcVreg, srcVreg2);
        MicroAPI::Cast<float, ScaleT, layoutZMrgZ>(f32Vreg, srcVreg, mask1);
        MicroAPI::Cast<float, ScaleT, layoutZMrgZ>(f32Vreg2, srcVreg2, mask2);
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND ||
                      castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR ||
                      castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<int16_t, float, castTrait>((MicroAPI::RegTensor<int16_t> &)f32Vreg, f32Vreg, mask1);
            MicroAPI::Cast<int16_t, float, castTrait>((MicroAPI::RegTensor<int16_t> &)f32Vreg2, f32Vreg2, mask2);
        } else {
            MicroAPI::Cast<int16_t, float, LayoutZMrgZRndRSatS>((MicroAPI::RegTensor<int16_t> &)f32Vreg, f32Vreg, mask1);
            MicroAPI::Cast<int16_t, float, LayoutZMrgZRndRSatS>((MicroAPI::RegTensor<int16_t> &)f32Vreg2, f32Vreg2, mask2);
        }
        MicroAPI::Cast<half, int16_t, LayoutZMrgZRndRSatS>(
            (MicroAPI::RegTensor<half> &)f32Vreg, (MicroAPI::RegTensor<int16_t> &)f32Vreg, mask1);
        MicroAPI::Cast<half, int16_t, LayoutZMrgZRndRSatS>(
            (MicroAPI::RegTensor<half> &)f32Vreg2, (MicroAPI::RegTensor<int16_t> &)f32Vreg2, mask2);
        MicroAPI::Cast<DstT, half, LayoutZMrgZRndRSatS>(dstVreg, (MicroAPI::RegTensor<half> &)f32Vreg, mask1);
        MicroAPI::Cast<DstT, half, LayoutZMrgZRndRSatS>(dstVreg2, (MicroAPI::RegTensor<half> &)f32Vreg2, mask2);
        MicroAPI::DeInterleave((MicroAPI::RegTensor<ScaleT> &)dstVreg, (MicroAPI::RegTensor<ScaleT> &)dstVreg2,
            (MicroAPI::RegTensor<ScaleT> &)dstVreg, (MicroAPI::RegTensor<ScaleT> &)dstVreg2);
    } else if constexpr (SupportType<ScaleT, float>()) {
        // fp32->s16->fp16->s8
        MicroAPI::RegTensor<half> f16Vreg;
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND ||
                      castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR ||
                      castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<int16_t, ScaleT, castTrait>((MicroAPI::RegTensor<int16_t> &)f16Vreg, srcVreg, mask);
        } else {
            MicroAPI::Cast<int16_t, ScaleT, LayoutZMrgZRndRSatS>((MicroAPI::RegTensor<int16_t> &)f16Vreg, srcVreg, mask);
        }
        MicroAPI::Cast<half, int16_t, LayoutZMrgZRndRSatS>(f16Vreg, (MicroAPI::RegTensor<int16_t> &)f16Vreg, mask);
        MicroAPI::Cast<DstT, half, LayoutZMrgZRndRSatS>(dstVreg, f16Vreg, mask);
    } else if constexpr (SupportType<ScaleT, half>()) {
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND ||
                      castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR ||
                      castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<DstT, ScaleT, castTrait>(dstVreg, srcVreg, mask);
        } else {
            MicroAPI::Cast<DstT, ScaleT, LayoutZMrgZRndRSatS>(dstVreg, srcVreg, mask);
        }
    }
}

template <typename DstT, typename ScaleT, const MicroAPI::CastTrait& castTrait>
__simd_callee__ inline void TransRegForFp4(
    MicroAPI::RegTensor<ScaleT> &vreg, MicroAPI::RegTensor<DstT> &dstVreg, MicroAPI::MaskReg &mask)
{
    MicroAPI::RegTensor<bfloat16_t> bf16Vreg;
    if constexpr (SupportType<ScaleT, float>()) {
        MicroAPI::Cast<bfloat16_t, ScaleT, LayoutZMrgZRndRSatS>(bf16Vreg, vreg, mask);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
            (MicroAPI::RegTensor<uint16_t> &)bf16Vreg, (MicroAPI::RegTensor<uint32_t> &)bf16Vreg);
        MicroAPI::MaskPack(mask, mask);
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND ||
                      castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR ||
                      castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<DstT, bfloat16_t, castTrait>(dstVreg, bf16Vreg, mask);
        } else {
            MicroAPI::Cast<DstT, bfloat16_t, LayoutZMrgZRndRSatS>(dstVreg, bf16Vreg, mask);
        }
    } else if constexpr (SupportType<ScaleT, half>()) {
        MicroAPI::Cast<bfloat16_t, ScaleT, LayoutZMrgZRndRSatS>(bf16Vreg, vreg, mask);
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND ||
                      castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR ||
                      castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<DstT, bfloat16_t, castTrait>(dstVreg, bf16Vreg, mask);
        } else {
            MicroAPI::Cast<DstT, bfloat16_t, LayoutZMrgZRndRSatS>(dstVreg, bf16Vreg, mask);
        }
    } else if constexpr (SupportType<ScaleT, bfloat16_t>()) {
        if constexpr (castTrait.roundMode == RoundMode::CAST_RINT || castTrait.roundMode == RoundMode::CAST_ROUND ||
                      castTrait.roundMode == RoundMode::CAST_CEIL || castTrait.roundMode == RoundMode::CAST_FLOOR ||
                      castTrait.roundMode == RoundMode::CAST_TRUNC) {
            MicroAPI::Cast<DstT, bfloat16_t, castTrait>(dstVreg, vreg, mask);
        } else {
            MicroAPI::Cast<DstT, bfloat16_t, LayoutZMrgZRndRSatS>(dstVreg, vreg, mask);
        }
    }
}

template <typename T>
__simd_callee__ inline void GetPerGroupScale(__ubuf__ T* scaleUb, const int32_t start, const uint32_t groupSize,
                                        MicroAPI::RegTensor<T>& scaleVreg)
{
    if constexpr (SupportType<T, half, bfloat16_t>()) {
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int16_t> vciVreg;
        MicroAPI::RegTensor<uint16_t> indexVreg;
        MicroAPI::RegTensor<uint16_t> gsizeVreg;
        MicroAPI::Duplicate(gsizeVreg, static_cast<uint16_t>(groupSize));
        MicroAPI::Arange(vciVreg, static_cast<int16_t>(start));
        MicroAPI::Div(indexVreg, (MicroAPI::RegTensor<uint16_t> &)vciVreg, gsizeVreg, fullMask);
        MicroAPI::DataCopyGather(scaleVreg, scaleUb, indexVreg, fullMask);
    } else {
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int32_t> vciVreg;
        MicroAPI::RegTensor<uint32_t> indexVreg;
        MicroAPI::RegTensor<uint32_t> gsizeVreg;
        MicroAPI::Duplicate(gsizeVreg, static_cast<uint32_t>(groupSize));
        MicroAPI::Arange(vciVreg, static_cast<int32_t>(start));
        MicroAPI::Div(indexVreg, (MicroAPI::RegTensor<uint32_t> &)vciVreg, gsizeVreg, fullMask);
        MicroAPI::DataCopyGather(scaleVreg, scaleUb, indexVreg, fullMask);
    }
}

template <typename T>
__simd_callee__ inline void GetPerGroupScaleToFloat(__ubuf__ T* scaleAddr, const int32_t start,
                                               const uint32_t groupSize, MicroAPI::RegTensor<float>& floatVreg,
                                               MicroAPI::RegTensor<T>& tempVreg, MicroAPI::MaskReg& mask)
{
    if constexpr (SupportType<T, half, bfloat16_t>()) {
        GetPerGroupScale(scaleAddr, start, groupSize, tempVreg);
        MicroAPI::UnPack<uint32_t, uint16_t>(
            (MicroAPI::RegTensor<uint32_t>&)tempVreg, (MicroAPI::RegTensor<uint16_t>&)tempVreg);
        MicroAPI::Cast<float, T, layoutZMrgZ>(floatVreg, tempVreg, mask);
    } else {
        GetPerGroupScale(scaleAddr, start, groupSize, floatVreg);
    }
}

template <typename T>
__simd_callee__ inline void DuplicateScalarToFloatVector(MicroAPI::RegTensor<float>& floatVreg, const T& scalar,
                                                    MicroAPI::RegTensor<T>& tempVreg, MicroAPI::MaskReg& mask)
{
    if constexpr (SupportType<T, half, bfloat16_t>()) {
        MicroAPI::Duplicate(tempVreg, scalar);
        MicroAPI::Cast<float, T, layoutZMrgZ>(floatVreg, tempVreg, mask);
    } else {
        MicroAPI::Duplicate(floatVreg, scalar);
    }
}
      
template <typename DstT, const MicroAPI::CastTrait& castTrait>
__simd_callee__ inline void CastFp32DstToExpect(MicroAPI::RegTensor<float>& srcVreg, MicroAPI::RegTensor<DstT>& dstVreg,
    MicroAPI::MaskReg& mask)
{
    if constexpr (SupportType<DstT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        QuantizeUtils::TransRegForFp8<DstT, float, castTrait>(srcVreg, dstVreg, mask);
    } else if constexpr (IsSameType<DstT, hifloat8_t>::value) {
        QuantizeUtils::TransRegForHif8<DstT, float, castTrait>(srcVreg, dstVreg, mask);
    } else {
        QuantizeUtils::TransRegForS8<DstT, float, castTrait>(srcVreg, dstVreg, mask);
    }
}
} // namespace QuantizeUtils

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT, typename ActualScaleT>
__simd_vf__ inline void QuantizePerGroupForKColCommonVF(__ubuf__ DstT* dstUb, __ubuf__ SrcT* srcUb,
    __ubuf__ ActualScaleT* scaleUb, __ubuf__ ActualScaleT* offsetUb, const OffsetT offset, const QuantizeParams params)
{
    constexpr bool isScalarOffset = TypeUtils::IsInnerDefaultType<OffsetT>();
    uint16_t rowNum = params.m;
    uint32_t vecLen = GetVecLen() / sizeof(float);
    uint16_t repeat = CeilDivision(params.n, vecLen);
    uint16_t scaleK = CeilDivision(params.n, params.groupSize);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<DstT> dstVreg;
    MicroAPI::RegTensor<float> scaleVreg;
    MicroAPI::RegTensor<float> offsetVreg;
    MicroAPI::RegTensor<ActualScaleT> tempScaleVreg;
    MicroAPI::RegTensor<ActualScaleT> tempOffsetVreg;
    MicroAPI::RegTensor<SrcT> tempSrcVreg;
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    if constexpr (config.hasOffset && isScalarOffset) {
        QuantizeUtils::DuplicateScalarToFloatVector<ActualScaleT>(offsetVreg, offset, tempOffsetVreg, fullMask);
    }
    for (uint16_t i = 0; i < rowNum; ++i) {
        uint32_t sreg = params.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            mask = MicroAPI::UpdateMask<float>(sreg);
            QuantizeUtils::GetPerGroupScaleToFloat(scaleUb + i * scaleK, j * vecLen, params.groupSize,
                scaleVreg, tempScaleVreg, fullMask);
            if constexpr (config.hasOffset && !isScalarOffset) {
                QuantizeUtils::GetPerGroupScaleToFloat(offsetUb  + i * scaleK, j * vecLen, params.groupSize,
                    offsetVreg, tempOffsetVreg, fullMask);
            }
            if constexpr (SupportType<SrcT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    tempSrcVreg, srcUb + i * params.n + j * vecLen);
                MicroAPI::Cast<float, SrcT, layoutZMrgZ>(srcVreg, tempSrcVreg, mask);
            } else {
                MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * params.n + j * vecLen);
            }
            MicroAPI::Mul(srcVreg, srcVreg, scaleVreg, mask);
            if constexpr (config.hasOffset) {
                MicroAPI::Add(srcVreg, srcVreg, offsetVreg, mask);
            }
            QuantizeUtils::CastFp32DstToExpect<DstT, castTrait>(srcVreg, dstVreg, mask);
            QuantizeUtils::StoreRes<DstT, float>(dstUb + i * params.n + j * vecLen, dstVreg, mask);
        }
    }
}

/******************* PerGroup **********************/
template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void QuantizePerGroupForKColCommon(const LocalTensor<DstT> &dstTensor, const LocalTensor<SrcT> &srcTensor,
    const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
{
    using ActualScaleT = typename ScaleT::PrimType;
    constexpr bool isScalarOffset = TypeUtils::IsInnerDefaultType<OffsetT>();
    __ubuf__ DstT* dstUb = (__ubuf__ DstT*)dstTensor.GetPhyAddr();
    __ubuf__ SrcT* srcUb = (__ubuf__ SrcT*)srcTensor.GetPhyAddr();
    __ubuf__ ActualScaleT* scaleUb = (__ubuf__ ActualScaleT*)scale.GetPhyAddr();
    __ubuf__ ActualScaleT* offsetUb = nullptr;
    if constexpr (!isScalarOffset) {
        offsetUb = (__ubuf__ ActualScaleT*)offset.GetPhyAddr();
    }
    QuantizePerGroupForKColCommonVF<config, DstT, SrcT, ScaleT, OffsetT, ActualScaleT>(dstUb, srcUb, scaleUb,
        offsetUb, offset, params);
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename ActualScaleT>
__simd_vf__ inline void QuantizePerGroupForKColFp4VF(__ubuf__ DstT* dstUb, __ubuf__ SrcT* srcUb,
    __ubuf__ ActualScaleT* scaleUb, const QuantizeParams params)
{
    uint16_t rowNum = params.m;
    uint32_t vecLen = GetVecLen() / sizeof(float);
    uint16_t repeat = CeilDivision(params.n, vecLen);
    uint16_t scaleK = CeilDivision(params.n, params.groupSize);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<DstT> dstVreg;
    MicroAPI::RegTensor<float> scaleVreg;
    MicroAPI::RegTensor<SrcT> tempVreg;
    MicroAPI::RegTensor<ActualScaleT> tempScaleVreg;
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < rowNum; ++i) {
        uint32_t sreg = params.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            mask = MicroAPI::UpdateMask<float>(sreg);
            QuantizeUtils::GetPerGroupScaleToFloat(scaleUb + i * scaleK, j * vecLen, params.groupSize,
                scaleVreg, tempScaleVreg, fullMask);
            if constexpr (SupportType<SrcT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    tempVreg, srcUb + i * params.n + j * vecLen);
                MicroAPI::Cast<float, SrcT, layoutZMrgZ>(srcVreg, tempVreg, mask);
            } else {
                MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * params.n + j * vecLen);
            }
            MicroAPI::Mul(srcVreg, srcVreg, scaleVreg, mask);
            QuantizeUtils::TransRegForFp4<DstT, float, castTrait>(srcVreg, dstVreg, mask);
            MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                (__ubuf__ uint8_t *)dstUb + (i * params.n + j * vecLen) / 2,
                (MicroAPI::RegTensor<uint8_t> &)dstVreg, mask);
        }
    }
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT>
__aicore__ inline void QuantizePerGroupForKColFp4(const LocalTensor<DstT> &dstTensor, const LocalTensor<SrcT> &srcTensor,
    const ScaleT& scale, const QuantizeParams& params)
{
    using ActualScaleT = typename ScaleT::PrimType;
    __ubuf__ DstT* dstUb = (__ubuf__ DstT*)dstTensor.GetPhyAddr();
    __ubuf__ SrcT* srcUb = (__ubuf__ SrcT*)srcTensor.GetPhyAddr();
    __ubuf__ ActualScaleT* scaleUb = (__ubuf__ ActualScaleT*)scale.GetPhyAddr();
    QuantizePerGroupForKColFp4VF<config, DstT, SrcT, ScaleT, ActualScaleT>(dstUb, srcUb, scaleUb, params);
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void QuantizePerGroupForKCol(const LocalTensor<DstT> &dstTensor, const LocalTensor<SrcT> &srcTensor,
    const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
{
    static_assert(TypeUtils::IsLocalTensorType<ScaleT>(), "Quantize PerGroup ScaleT should be Tensor");
    using ActualScaleT = typename ScaleT::PrimType;
    constexpr bool isScalarOffset = TypeUtils::IsInnerDefaultType<OffsetT>();
    if constexpr (isScalarOffset) {
        static_assert(IsSameType<ActualScaleT, OffsetT>::value, "scale and offset should be the same PrimType");
    } else {
        using ActualOffsetT = typename OffsetT::PrimType;
        static_assert(IsSameType<ActualScaleT, ActualOffsetT>::value, "scale and offset should be the same PrimType");
    }
    QuantizeUtils::CheckApiDtypeValid<DstT, SrcT, ActualScaleT>();
    // fp16, fp32, bf16 -> fp8 should always cast to fp32
    if constexpr (SupportType<DstT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t, int8_t>()) {
        QuantizePerGroupForKColCommon<config, DstT, SrcT, ScaleT, OffsetT>(dstTensor, srcTensor, scale, offset, params);
    } else if constexpr (SupportType<DstT, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
        QuantizePerGroupForKColFp4<config, DstT, SrcT, ScaleT>(dstTensor, srcTensor, scale, params);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for Quantize!"); });
    }
}

template <typename DstT, typename SrcT, typename ActualScaleT, bool hasOffset, bool isScalarOffset, const MicroAPI::CastTrait& castTrait>
__simd_callee__ inline void QuantizePerGroupForKRowCommonOneRow(__ubuf__ DstT *dstAddr, __ubuf__ SrcT *srcAddr,
    __ubuf__ ActualScaleT *scaleAddr, __ubuf__ ActualScaleT *offsetAddr, MicroAPI::RegTensor<DstT> &dstVreg,
    MicroAPI::RegTensor<float> &srcVreg, MicroAPI::RegTensor<float> &scaleVreg,
    MicroAPI::RegTensor<float> &offsetVreg,  uint16_t repeat, uint32_t n, uint32_t vecLen)
{
    MicroAPI::RegTensor<ActualScaleT> tempScaleVreg;
    MicroAPI::RegTensor<ActualScaleT> tempOffsetVreg;
    MicroAPI::RegTensor<SrcT> tempSrcVreg;
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg mask;
    for (uint16_t j = 0; j < repeat; ++j) {
        if constexpr (SupportType<ActualScaleT, half, bfloat16_t>()) {
            MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                tempScaleVreg, scaleAddr + j * vecLen);
            MicroAPI::Cast<float, ActualScaleT, layoutZMrgZ>(scaleVreg, tempScaleVreg, fullMask);
            if constexpr (hasOffset && !isScalarOffset) {
                MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    tempOffsetVreg, offsetAddr + j * vecLen);
                MicroAPI::Cast<float, ActualScaleT, layoutZMrgZ>(offsetVreg, tempOffsetVreg, fullMask);
            }
        } else {
            MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleAddr + j * vecLen);
            if constexpr (hasOffset && !isScalarOffset) {
                MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_NORM>(offsetVreg, offsetAddr + j * vecLen);
            }
        }
        mask = MicroAPI::UpdateMask<float>(n);
        if constexpr (SupportType<SrcT, half, bfloat16_t>()) {
            MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempSrcVreg, srcAddr + j * vecLen);
            MicroAPI::Cast<float, SrcT, layoutZMrgZ>(srcVreg, tempSrcVreg, mask);
        } else {
            MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcAddr + j * vecLen);
        }
        MicroAPI::Mul(srcVreg, srcVreg, scaleVreg, mask);
        if constexpr (hasOffset) {
            MicroAPI::Add(srcVreg, srcVreg, offsetVreg, mask);
        }
        QuantizeUtils::CastFp32DstToExpect<DstT, castTrait>(srcVreg, dstVreg, mask);
        QuantizeUtils::StoreRes<DstT, float>(dstAddr + j * vecLen, dstVreg, mask);
    }
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT, typename ActualScaleT>
__simd_vf__ inline void QuantizePerGroupForKRowCommonVF(__ubuf__ DstT* dstUb, __ubuf__ SrcT* srcUb,
    __ubuf__ ActualScaleT* scaleUb, __ubuf__ ActualScaleT* offsetUb, const OffsetT offset, const QuantizeParams params,
    uint16_t rowNum, uint16_t tailRow)
{
    constexpr bool isScalarOffset = TypeUtils::IsInnerDefaultType<OffsetT>();
    uint16_t mainRowGroup = rowNum / params.groupSize;
    uint32_t vecLen = GetVecLen() / sizeof(float);
    uint16_t repeat = CeilDivision(params.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<DstT> dstVreg;
    MicroAPI::RegTensor<float> scaleVreg;
    MicroAPI::RegTensor<float> offsetVreg;
    MicroAPI::RegTensor<ActualScaleT> tempOffsetVreg;
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    if constexpr (config.hasOffset && isScalarOffset) {
        QuantizeUtils::DuplicateScalarToFloatVector<ActualScaleT>(offsetVreg, offset, tempOffsetVreg, fullMask);
    }
    for (uint16_t i0 = 0; i0 < mainRowGroup; ++i0) {
        for (uint16_t i1 = 0; i1 < static_cast<uint16_t>(params.groupSize); ++i1) {
            if constexpr (isScalarOffset) {
                QuantizePerGroupForKRowCommonOneRow<DstT, SrcT, ActualScaleT, config.hasOffset, isScalarOffset, castTrait>(
                    dstUb + (i0 * params.groupSize + i1) * params.n,
                    srcUb + (i0 * params.groupSize + i1) * params.n,
                    scaleUb + i0 * params.n, nullptr,
                    dstVreg, srcVreg, scaleVreg, offsetVreg,
                    repeat, params.n, vecLen);
            } else {
                QuantizePerGroupForKRowCommonOneRow<DstT, SrcT, ActualScaleT, config.hasOffset, isScalarOffset, castTrait>(
                    dstUb + (i0 * params.groupSize + i1) * params.n,
                    srcUb + (i0 * params.groupSize + i1) * params.n,
                    scaleUb + i0 * params.n, offsetUb + i0 * params.n,
                    dstVreg, srcVreg, scaleVreg, offsetVreg,
                    repeat, params.n, vecLen);
            }
        }
    }
    for (uint16_t i = 0; i < tailRow; ++i) {
        if constexpr (isScalarOffset) {
            QuantizePerGroupForKRowCommonOneRow<DstT, SrcT, ActualScaleT, config.hasOffset, isScalarOffset, castTrait>(
                dstUb + (mainRowGroup * params.groupSize + i) * params.n,
                srcUb + (mainRowGroup * params.groupSize + i) * params.n,
                scaleUb + mainRowGroup * params.n, nullptr,
                dstVreg, srcVreg, scaleVreg, offsetVreg,
                repeat, params.n, vecLen);
        } else {
            QuantizePerGroupForKRowCommonOneRow<DstT, SrcT, ActualScaleT, config.hasOffset, isScalarOffset, castTrait>(
                dstUb + (mainRowGroup * params.groupSize + i) * params.n,
                srcUb + (mainRowGroup * params.groupSize + i) * params.n,
                scaleUb + mainRowGroup * params.n, offsetUb + mainRowGroup * params.n,
                dstVreg, srcVreg, scaleVreg, offsetVreg,
                repeat, params.n, vecLen);
        }
    }
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void QuantizePerGroupForKRowCommon(const LocalTensor<DstT> &dstTensor, const LocalTensor<SrcT> &srcTensor,
    const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
{
    using ActualScaleT = typename ScaleT::PrimType;
    constexpr bool isScalarOffset = TypeUtils::IsInnerDefaultType<OffsetT>();
    __ubuf__ DstT* dstUb = (__ubuf__ DstT*)dstTensor.GetPhyAddr();
    __ubuf__ SrcT* srcUb = (__ubuf__ SrcT*)srcTensor.GetPhyAddr();
    __ubuf__ ActualScaleT* scaleUb = (__ubuf__ ActualScaleT*)scale.GetPhyAddr();
    __ubuf__ ActualScaleT* offsetUb = nullptr;
    if constexpr (!isScalarOffset) {
        offsetUb = (__ubuf__ ActualScaleT*)offset.GetPhyAddr();
    }
    uint16_t rowNum = params.m;
    uint16_t tailRow = rowNum % params.groupSize;
    QuantizePerGroupForKRowCommonVF<config, DstT, SrcT, ScaleT, OffsetT, ActualScaleT>(dstUb, srcUb,
        scaleUb, offsetUb, offset, params, rowNum, tailRow);
}

template <typename DstT, typename SrcT, typename ActualScaleT, const MicroAPI::CastTrait& castTrait>
__simd_callee__ inline void QuantizePerGroupForKRowFp4OneRow(__ubuf__ DstT *dstAddr, __ubuf__ SrcT *srcAddr,
    __ubuf__ ActualScaleT *scaleAddr, MicroAPI::RegTensor<DstT> &dstVreg,
    MicroAPI::RegTensor<float> &srcVreg,MicroAPI::RegTensor<float> &scaleVreg,
    uint16_t repeat, uint32_t n, uint32_t vecLen)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<SrcT> tempVreg;
    MicroAPI::RegTensor<ActualScaleT> tempScaleVreg;
    for (uint16_t j = 0; j < repeat; ++j) {
        mask = MicroAPI::UpdateMask<float>(n);
        if constexpr (SupportType<ActualScaleT, half, bfloat16_t>()) {
            MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempScaleVreg, scaleAddr + j * vecLen);
            MicroAPI::Cast<float, ActualScaleT, layoutZMrgZ>(scaleVreg, tempScaleVreg, mask);
        } else {
            MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleAddr + j * vecLen);
        }
        if constexpr (SupportType<SrcT, half, bfloat16_t>()) {
            MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempVreg, srcAddr + j * vecLen);
            MicroAPI::Cast<float, SrcT, layoutZMrgZ>(srcVreg, tempVreg, mask);
        } else {
            MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcAddr + j * vecLen);
        }
        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, mask);
        QuantizeUtils::TransRegForFp4<DstT, float, castTrait>(srcVreg, dstVreg, mask);
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
            (__ubuf__ uint8_t *)dstAddr + (j * vecLen) / 2, (MicroAPI::RegTensor<uint8_t> &)dstVreg, mask);
    }
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename ActualScaleT>
__simd_vf__ inline void QuantizePerGroupForKRowFp4VF(__ubuf__ DstT* dstUb, __ubuf__ SrcT* srcUb,
    __ubuf__ ActualScaleT* scaleUb, const QuantizeParams params, uint16_t rowNum, uint16_t tailRow)
{
    uint16_t mainRowGroup = rowNum / params.groupSize;
    uint32_t vecLen = GetVecLen() / sizeof(float);
    uint16_t repeat = CeilDivision(params.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::RegTensor<DstT> dstVreg;
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> scaleVreg;
    for (uint16_t i0 = 0; i0 < mainRowGroup; ++i0) {
        for (uint16_t i1 = 0; i1 < static_cast<uint16_t>(params.groupSize); ++i1) {
            QuantizePerGroupForKRowFp4OneRow<DstT, SrcT, ActualScaleT, castTrait>(
                dstUb + ((i0 * params.groupSize + i1) * params.n) / 2,
                srcUb + (i0 * params.groupSize + i1) * params.n,
                scaleUb + i0 * params.n,
                dstVreg, srcVreg, scaleVreg,
                repeat, params.n, vecLen);
        }
    }
    for (uint16_t i = 0; i < tailRow; ++i) {
        QuantizePerGroupForKRowFp4OneRow<DstT, SrcT, ActualScaleT, castTrait>(
            dstUb + ((mainRowGroup * params.groupSize + i) * params.n) / 2,
            srcUb + (mainRowGroup * params.groupSize + i) * params.n,
            scaleUb + mainRowGroup * params.n,
            dstVreg, srcVreg, scaleVreg,
            repeat, params.n, vecLen);
    }
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT>
__aicore__ inline void QuantizePerGroupForKRowFp4(const LocalTensor<DstT> &dstTensor, const LocalTensor<SrcT> &srcTensor,
    const ScaleT& scale, const QuantizeParams& params)
{
    using ActualScaleT = typename ScaleT::PrimType;
    __ubuf__ DstT* dstUb = (__ubuf__ DstT*)dstTensor.GetPhyAddr();
    __ubuf__ SrcT* srcUb = (__ubuf__ SrcT*)srcTensor.GetPhyAddr();
    __ubuf__ ActualScaleT* scaleUb = (__ubuf__ ActualScaleT*)scale.GetPhyAddr();
    uint16_t rowNum = params.m;
    uint16_t tailRow = rowNum % params.groupSize;
    QuantizePerGroupForKRowFp4VF<config, DstT, SrcT, ScaleT, ActualScaleT>(dstUb, srcUb,
        scaleUb, params, rowNum, tailRow);
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void QuantizePerGroupForKRow(const LocalTensor<DstT> &dstTensor, const LocalTensor<SrcT> &srcTensor,
    const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
{
    static_assert(TypeUtils::IsLocalTensorType<ScaleT>(), "Quantize PerGroup ScaleT should be Tensor");
    using ActualScaleT = typename ScaleT::PrimType;
    constexpr bool isScalarOffset = TypeUtils::IsInnerDefaultType<OffsetT>();
    if constexpr (isScalarOffset) {
        static_assert(IsSameType<ActualScaleT, OffsetT>::value, "scale and offset should be the same PrimType");
    } else {
        using ActualOffsetT = typename OffsetT::PrimType;
        static_assert(IsSameType<ActualScaleT, ActualOffsetT>::value, "scale and offset should be the same PrimType");
    }
    QuantizeUtils::CheckApiDtypeValid<DstT, SrcT, ActualScaleT>();
    // fp16, fp32, bf16 -> fp8 should always cast to fp32
    if constexpr (SupportType<DstT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t, int8_t>()) {
        QuantizePerGroupForKRowCommon<config, DstT, SrcT, ScaleT, OffsetT>(dstTensor, srcTensor, scale, offset, params);
    } else if constexpr (SupportType<DstT, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
        QuantizePerGroupForKRowFp4<config, DstT, SrcT, ScaleT>(dstTensor, srcTensor, scale, params);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for Quantize!"); });
    }
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT, typename ActualScaleT>
__simd_vf__ inline void QuantizePerTokenCommonVF(__ubuf__ DstT* dstUb, __ubuf__ SrcT* srcUb,
    __ubuf__ ActualScaleT* scaleUb, __ubuf__ ActualScaleT* offsetUb, const OffsetT offset,
    const QuantizeParams params)
{
    constexpr bool isScalarOffset = TypeUtils::IsInnerDefaultType<OffsetT>();
    uint16_t rowNum = params.m;
    uint32_t vecLen = GetVecLen() / sizeof(float);
    uint16_t repeat = CeilDivision(params.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<DstT> dstVreg;
    MicroAPI::RegTensor<float> scaleVreg;
    MicroAPI::RegTensor<float> offsetVreg;
    MicroAPI::RegTensor<ActualScaleT> tempScaleVreg;
    MicroAPI::RegTensor<ActualScaleT> tempOffsetVreg;
    MicroAPI::RegTensor<SrcT> tempSrcVreg;
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    if constexpr (config.hasOffset && isScalarOffset) {
        QuantizeUtils::DuplicateScalarToFloatVector<ActualScaleT>(offsetVreg, offset, tempOffsetVreg, fullMask);
    }
    for (uint16_t i = 0; i < rowNum; ++i) {
        if constexpr (SupportType<ActualScaleT, half, bfloat16_t>()) {
            MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_BRC_B16>(tempScaleVreg, scaleUb + i);
            MicroAPI::Cast<float, ActualScaleT, layoutZMrgZ>(scaleVreg, tempScaleVreg, fullMask);
            if constexpr (config.hasOffset && !isScalarOffset) {
                MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_BRC_B16>(tempOffsetVreg, offsetUb + i);
                MicroAPI::Cast<float, ActualScaleT, layoutZMrgZ>(offsetVreg, tempOffsetVreg, fullMask);
            }
        } else {
            MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_BRC_B32>(scaleVreg, scaleUb + i);
            if constexpr (config.hasOffset && !isScalarOffset) {
                MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_BRC_B32>(offsetVreg, offsetUb + i);
            }
        }
        uint32_t sreg = params.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            mask = MicroAPI::UpdateMask<float>(sreg);
            if constexpr (SupportType<SrcT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    tempSrcVreg, srcUb + i * params.n + j * vecLen);
                MicroAPI::Cast<float, SrcT, layoutZMrgZ>(srcVreg, tempSrcVreg, mask);
            } else {
                MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * params.n + j * vecLen);
            }
            MicroAPI::Mul(srcVreg, srcVreg, scaleVreg, mask);
            if constexpr (config.hasOffset) {
                MicroAPI::Add(srcVreg, srcVreg, offsetVreg, mask);
            }
            QuantizeUtils::CastFp32DstToExpect<DstT, castTrait>(srcVreg, dstVreg, mask);
            QuantizeUtils::StoreRes<DstT, float>(dstUb + i * params.n + j * vecLen, dstVreg, mask);
        }
    }
}

/******************* PerToken **********************/
template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void QuantizePerTokenCommon(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor,
    const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
{
    using ActualScaleT = typename ScaleT::PrimType;
    constexpr bool isScalarOffset = TypeUtils::IsInnerDefaultType<OffsetT>();
    __ubuf__ DstT* dstUb = (__ubuf__ DstT*)dstTensor.GetPhyAddr();
    __ubuf__ SrcT* srcUb = (__ubuf__ SrcT*)srcTensor.GetPhyAddr();
    __ubuf__ ActualScaleT* scaleUb = (__ubuf__ ActualScaleT*)scale.GetPhyAddr();
    __ubuf__ ActualScaleT* offsetUb = nullptr;
    if constexpr (!isScalarOffset) {
        offsetUb = (__ubuf__ ActualScaleT*)offset.GetPhyAddr();
    }
    QuantizePerTokenCommonVF<config, DstT, SrcT, ScaleT, OffsetT, ActualScaleT>(dstUb, srcUb,
        scaleUb, offsetUb, offset, params);
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void QuantizePerToken(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor,
    const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
{
    static_assert(TypeUtils::IsLocalTensorType<ScaleT>(), "Quantize PerToken ScaleT should be Tensor");
    using ActualScaleT = typename ScaleT::PrimType;
    constexpr bool isScalarOffset = TypeUtils::IsInnerDefaultType<OffsetT>();
    if constexpr (isScalarOffset) {
        static_assert(IsSameType<ActualScaleT, OffsetT>::value, "scale and offset should be the same PrimType");
    } else {
        using ActualOffsetT = typename OffsetT::PrimType;
        static_assert(IsSameType<ActualScaleT, ActualOffsetT>::value, "scale and offset should be the same PrimType");
    }
    QuantizeUtils::CheckApiDtypeValid<DstT, SrcT, ActualScaleT>();
    QuantizePerTokenCommon<config, DstT, SrcT, ScaleT, OffsetT>(dstTensor, srcTensor, scale, offset, params);
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT, typename ActualScaleT>
__simd_vf__ inline void QuantizePerChannelCommonVF(__ubuf__ DstT* dstUb, __ubuf__ SrcT* srcUb,
    __ubuf__ ActualScaleT* scaleUb, __ubuf__ ActualScaleT* offsetUb, const OffsetT offset,
    const QuantizeParams params)
{
    constexpr bool isScalarOffset = TypeUtils::IsInnerDefaultType<OffsetT>();
    uint16_t colNum = params.n;
    uint32_t vecLen = GetVecLen() / sizeof(float);
    uint16_t repeat = CeilDivision(colNum, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<DstT> dstVreg;
    MicroAPI::RegTensor<float> scaleVreg;
    MicroAPI::RegTensor<float> offsetVreg;
    MicroAPI::RegTensor<ActualScaleT> tempScaleVreg;
    MicroAPI::RegTensor<ActualScaleT> tempOffsetVreg;
    MicroAPI::RegTensor<SrcT> tempSrcVreg;
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    if constexpr (config.hasOffset && isScalarOffset) {
        QuantizeUtils::DuplicateScalarToFloatVector<ActualScaleT>(offsetVreg, offset, tempOffsetVreg, fullMask);
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(params.m); ++i) {
        uint32_t sreg = colNum;
        for (uint16_t j = 0; j < repeat; ++j) {
            mask = MicroAPI::UpdateMask<float>(sreg);
            if constexpr (SupportType<ActualScaleT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    tempScaleVreg, scaleUb + j * vecLen);
                MicroAPI::Cast<float, ActualScaleT, layoutZMrgZ>(scaleVreg, tempScaleVreg, fullMask);
                if constexpr (config.hasOffset && !isScalarOffset) {
                    MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        tempOffsetVreg, offsetUb + j * vecLen);
                    MicroAPI::Cast<float, ActualScaleT, layoutZMrgZ>(offsetVreg, tempOffsetVreg, fullMask);
                }
            } else {
                MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb + j * vecLen);
                if constexpr (config.hasOffset && !isScalarOffset) {
                    MicroAPI::DataCopy<ActualScaleT, MicroAPI::LoadDist::DIST_NORM>(offsetVreg, offsetUb + j * vecLen);
                }
            }

            if constexpr (SupportType<SrcT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    tempSrcVreg, srcUb + i * params.n + j * vecLen);
                MicroAPI::Cast<float, SrcT, layoutZMrgZ>(srcVreg, tempSrcVreg, mask);
            } else {
                MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * params.n + j * vecLen);
            }
            MicroAPI::Mul(srcVreg, srcVreg, scaleVreg, mask);
            if constexpr (config.hasOffset) {
                MicroAPI::Add(srcVreg, srcVreg, offsetVreg, mask);
            }
            QuantizeUtils::CastFp32DstToExpect<DstT, castTrait>(srcVreg, dstVreg, mask);
            QuantizeUtils::StoreRes<DstT, float>(dstUb + i * colNum + j * vecLen, dstVreg, mask);
        }
    }
}

/******************* PerChannel **********************/
template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void QuantizePerChannelCommon(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor,
    const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
{
    using ActualScaleT = typename ScaleT::PrimType;
    constexpr bool isScalarOffset = TypeUtils::IsInnerDefaultType<OffsetT>();
    __ubuf__ DstT* dstUb = (__ubuf__ DstT*)dstTensor.GetPhyAddr();
    __ubuf__ SrcT* srcUb = (__ubuf__ SrcT*)srcTensor.GetPhyAddr();
    __ubuf__ ActualScaleT* scaleUb = (__ubuf__ ActualScaleT*)scale.GetPhyAddr();
    __ubuf__ ActualScaleT* offsetUb = nullptr;
    if constexpr (!isScalarOffset) {
        offsetUb = (__ubuf__ ActualScaleT*)offset.GetPhyAddr();
    }
    QuantizePerChannelCommonVF<config, DstT, SrcT, ScaleT, OffsetT, ActualScaleT>(dstUb, srcUb,
        scaleUb, offsetUb, offset, params);
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void QuantizePerChannel(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor,
    const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
{
    static_assert(TypeUtils::IsLocalTensorType<ScaleT>(), "Quantize PerChannel ScaleT should be Tensor");
    using ActualScaleT = typename ScaleT::PrimType;
    constexpr bool isScalarOffset = TypeUtils::IsInnerDefaultType<OffsetT>();
    if constexpr (isScalarOffset) {
        static_assert(IsSameType<ActualScaleT, OffsetT>::value, "scale and offset should be the same PrimType");
    } else {
        using ActualOffsetT = typename OffsetT::PrimType;
        static_assert(IsSameType<ActualScaleT, ActualOffsetT>::value, "scale and offset should be the same PrimType");
    }
    QuantizeUtils::CheckApiDtypeValid<DstT, SrcT, ActualScaleT>();
    QuantizePerChannelCommon<config, DstT, SrcT, ScaleT, OffsetT>(dstTensor, srcTensor, scale, offset, params);
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__simd_vf__ inline void QuantizePerTensorCommonVF(__ubuf__ DstT* dstUb, __ubuf__ SrcT* srcUb,
    const ScaleT scale, const OffsetT offset, const QuantizeParams params)
{
    uint16_t rowNum = params.m;
    constexpr uint32_t vecLen = GetVecLen() / sizeof(float);
    uint16_t repeat = CeilDivision(params.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<DstT> dstVreg;
    MicroAPI::RegTensor<float> scaleVreg;
    MicroAPI::RegTensor<float> offsetVreg;
    MicroAPI::RegTensor<ScaleT> tempScaleVreg;
    MicroAPI::RegTensor<OffsetT> tempOffsetVreg;
    MicroAPI::RegTensor<SrcT> tempSrcVreg;
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    QuantizeUtils::DuplicateScalarToFloatVector<ScaleT>(scaleVreg, scale, tempScaleVreg, fullMask);
    if constexpr (config.hasOffset) {
        QuantizeUtils::DuplicateScalarToFloatVector<ScaleT>(offsetVreg, offset, tempOffsetVreg, fullMask);
    }
    for (uint16_t i = 0; i < rowNum; ++i) {
        uint32_t sreg = params.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            mask = MicroAPI::UpdateMask<float>(sreg);
            if constexpr (SupportType<SrcT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    tempSrcVreg, srcUb + i * params.n + j * vecLen);
                MicroAPI::Cast<float, SrcT, layoutZMrgZ>(srcVreg, tempSrcVreg, mask);
            } else {
                MicroAPI::DataCopy<SrcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * params.n + j * vecLen);
            }
            MicroAPI::Mul(srcVreg, srcVreg, scaleVreg, mask);
            if constexpr (config.hasOffset) {
                MicroAPI::Add(srcVreg, srcVreg, offsetVreg, mask);
            }
            QuantizeUtils::CastFp32DstToExpect<DstT, castTrait>(srcVreg, dstVreg, mask);
            QuantizeUtils::StoreRes<DstT, float>(dstUb + i * params.n + j * vecLen, dstVreg, mask);
        }
    }
}

/******************* PerTensor **********************/
template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void QuantizePerTensorCommon(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor,
    const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
{
    __ubuf__ DstT* dstUb = (__ubuf__ DstT*)dstTensor.GetPhyAddr();
    __ubuf__ SrcT* srcUb = (__ubuf__ SrcT*)srcTensor.GetPhyAddr();
    QuantizePerTensorCommonVF<config, DstT, SrcT, ScaleT, OffsetT>(dstUb, srcUb, scale, offset, params);
}

template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void QuantizePerTensor(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor,
    const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
{
    static_assert(TypeUtils::IsInnerDefaultType<ScaleT>(), "Quantize PerTensor ScaleT should be Scalar");
    static_assert(TypeUtils::IsInnerDefaultType<OffsetT>(), "Quantize PerTensor OffsetT should be Scalar");
    static_assert(IsSameType<ScaleT, OffsetT>::value, "ScaleT and OffsetT should be the same type");
    QuantizeUtils::CheckApiDtypeValid<DstT, SrcT, ScaleT>();
    QuantizePerTensorCommon<config, DstT, SrcT, ScaleT, OffsetT>(dstTensor, srcTensor, scale, offset, params);
}

template <typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void CheckQuantizeParams(const LocalTensor<DstT> &dstTensor, const LocalTensor<SrcT> &srcTensor,
    const ScaleT &scale, const OffsetT &offset)
{
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    if constexpr (TypeUtils::IsLocalTensorType<ScaleT>()) {
        CheckTensorPosition(scale, "scale", "VECIN, VECOUT, VECCALC");
    }
    if constexpr (TypeUtils::IsLocalTensorType<OffsetT>()) {
        CheckTensorPosition(offset, "offset", "VECIN, VECOUT, VECCALC");
    }
    static_assert(SupportType<SrcT, half, float, bfloat16_t>(),
        "Quantize only support half/float/bfloat16_t input dtype");
    if constexpr (TypeUtils::IsInnerDefaultType<ScaleT>()) {
        static_assert(SupportType<ScaleT, half, float, bfloat16_t>(),
        "Quantize only support half/float/bfloat16_t scale dtype");
    } else {
        using ActualScaleT = typename ScaleT::PrimType;
        static_assert(SupportType<ActualScaleT, half, float, bfloat16_t>(),
        "Quantize only support half/float/bfloat16_t scale dtype");
    }
}

/*********************** Impl *****************************/
template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
__aicore__ inline void QuantizeImpl(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor,
    const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    CheckQuantizeParams(dstTensor, srcTensor, scale, offset);

    static_assert((config.policy == QuantizePolicy::PER_TENSOR || config.policy == QuantizePolicy::PER_CHANNEL ||
        config.policy == QuantizePolicy::PER_TOKEN || config.policy == QuantizePolicy::PER_GROUP),
        "unsupported policy for Quantize in current device!");
    ASCENDC_ASSERT(
        (params.n % GetDataBlockSizeInBytes() == 0), { KERNEL_LOG(KERNEL_ERROR, "n must be 32B aligned"); });
    if constexpr (config.policy == QuantizePolicy::PER_TENSOR) {
        static_assert(
            SupportType<DstT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t>(),
            "Quantize PerTensor only support int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t output dtype");
        QuantizePerTensor<config, DstT, SrcT, ScaleT, OffsetT>(dstTensor, srcTensor, scale, offset, params);
    } else if constexpr (config.policy == QuantizePolicy::PER_CHANNEL) {
        static_assert(
            SupportType<DstT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t>(),
            "Quantize PerChannel only support int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t output dtype");
        QuantizePerChannel<config, DstT, SrcT, ScaleT, OffsetT>(dstTensor, srcTensor, scale, offset, params);
    } else if constexpr (config.policy == QuantizePolicy::PER_TOKEN) {
        static_assert(
            SupportType<DstT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t>(),
            "Quantize PerToken only support int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t output dtype");
        QuantizePerToken<config, DstT, SrcT, ScaleT, OffsetT>(dstTensor, srcTensor, scale, offset, params);
    } else if constexpr (config.policy == QuantizePolicy::PER_GROUP) {
        static_assert(
            SupportType<DstT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t, fp4x2_e2m1_t, fp4x2_e1m2_t>(),
            "Quantize PerGroup only support "
            "int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t/fp4x2_e2m1_t/fp4x2_e1m2_t output dtype");
        static_assert(
            ((config.kDim == 1) || (config.kDim == 0)), "Quantize PerGroup only support K is axis 0/1!");
        ASCENDC_ASSERT((params.groupSize > 0 && params.groupSize % 32 == 0),
            { KERNEL_LOG(KERNEL_ERROR, "groupSize must be an integer multiple of 32 and greater then 0 !"); });
        if constexpr (config.kDim == 1) {
            QuantizePerGroupForKCol<config, DstT, SrcT, ScaleT, OffsetT>(dstTensor, srcTensor, scale, offset, params);
        } else {
            QuantizePerGroupForKRow<config, DstT, SrcT, ScaleT, OffsetT>(dstTensor, srcTensor, scale, offset, params);
        }
    }
}
}  //  namespace AscendC
#endif  // IMPL_QUANTIZATION_QUANTIZE_QUANTIZE_IMPL_H
