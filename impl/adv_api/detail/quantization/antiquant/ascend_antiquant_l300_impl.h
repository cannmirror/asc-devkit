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
 * \file ascend_antiquant_l300_impl.h
 * \brief
 */
#ifndef IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_L300_IMPL_H
#define IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_L300_IMPL_H

#include "kernel_tensor.h"
#include "kernel_basic_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "ascend_antiquant_common.h"
#include "../../common/check.h"

namespace AscendC {
constexpr uint32_t ANTIQUANT_B16_VF_LEN = GetVecLen() / sizeof(uint16_t);
constexpr uint32_t ANTIQUANT_B32_VF_LEN = GetVecLen() / sizeof(uint32_t);

constexpr MicroAPI::CastTrait castTraitB42Fp16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};

template <typename SrcType, typename OutType>
__aicore__ inline void CheckApiDtypeValid()
{
    constexpr bool inputValid = (IsSameType<SrcType, int8_t>::value);
    constexpr bool outputValid = (IsSameType<OutType, half>::value);
    ASCENDC_ASSERT((inputValid && outputValid), {
        KERNEL_LOG(KERNEL_ERROR, 
        "Failed to check dtype in AscendAntiQuant, "
        "current api support dtype combination is src: int8_t, dst: half.");
    });
}

/* **************************************************************************************************
 * perTensor for B8                                             *
 * ************************************************************************************************* */
template <typename SrcType, typename OutputDataType>
__simd_vf__ inline void PerTensorProcessForFp8(__ubuf__ OutputDataType* dst, __ubuf__ SrcType* src,
    const OutputDataType offset, const OutputDataType scale, const uint32_t srcCalCount)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SrcType> vreg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<OutputDataType> outReg;

    uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B32_VF_LEN);
    uint32_t sreg = static_cast<uint32_t>(srcCalCount);
    uint16_t repeat = CeilDivision(srcCalCount, sregLower);

    for (uint16_t i = 0; i < static_cast<uint16_t>(repeat); ++i) {
        preg = MicroAPI::UpdateMask<uint32_t>(sreg);
        MicroAPI::DataCopy<SrcType, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vreg, src + i * sregLower);
        MicroAPI::Cast<float, SrcType, layoutZMrgZ>(f32vreg, vreg, preg);
        MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(
            f32vreg, f32vreg, static_cast<float>(offset), preg);
        MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(
            f32vreg, f32vreg, static_cast<float>(scale), preg);
        MicroAPI::Cast<OutputDataType, float, LayoutZMrgZRndRSatS>(outReg, f32vreg, preg);
        MicroAPI::DataCopy<OutputDataType, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * sregLower, outReg, preg);
    }
}
template <typename SrcType>
__simd_vf__ inline void PerTensorProcessForB8(
    __ubuf__ half* dst, __ubuf__ SrcType* src, const half offset, const half scale, const uint32_t srcCalCount)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SrcType> vreg;
    MicroAPI::RegTensor<half> f16vreg;

    uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B16_VF_LEN);
    uint32_t sreg = static_cast<uint32_t>(srcCalCount);
    uint16_t repeat = CeilDivision(srcCalCount, sregLower);

    for (uint16_t i = 0; i < static_cast<uint16_t>(repeat); ++i) {
        preg = MicroAPI::UpdateMask<uint16_t>(sreg);
        MicroAPI::DataCopy<SrcType, MicroAPI::LoadDist::DIST_UNPACK_B8>(vreg, src + i * sregLower);
        MicroAPI::Cast<half, SrcType, layoutZMrgZ>(f16vreg, vreg, preg); // hif8->f16 or int8->f16
        MicroAPI::Adds<half, half, MicroAPI::MaskMergeMode::ZEROING>(f16vreg, f16vreg, offset, preg);
        MicroAPI::Muls<half, half, MicroAPI::MaskMergeMode::ZEROING>(f16vreg, f16vreg, scale, preg);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_NORM_B16>(dst + i * sregLower, f16vreg, preg);
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AntiQuantPertensorImpl(const LocalTensor<OutputDataType>& dst, const LocalTensor<SrcType>& src, 
    const OutputDataType offset, const OutputDataType scale, const LocalTensor<uint8_t>& sharedTmpBuffer, 
    const uint32_t K, const AntiQuantShapeInfo& shapeInfo = {})
{
    static_assert(SupportType<SrcType, int8_t>(), "This AscendAntiQuant only support int8 input dtype");
    static_assert(SupportType<OutputDataType, half>(), "This AscendAntiQuant only support f16 output dtype");
    __ubuf__ OutputDataType* dstUb = (__ubuf__ OutputDataType*)dst.GetPhyAddr();
    __ubuf__ SrcType* srcUb = (__ubuf__ SrcType*)src.GetPhyAddr();
    auto tmpbuffer = sharedTmpBuffer.ReinterpretCast<OutputDataType>();
    __ubuf__ OutputDataType* tmpbufferUb = (__ubuf__ OutputDataType*)tmpbuffer.GetPhyAddr();

    uint32_t srcCalCount = src.GetSize();
    // vfcall not support overload function
    PerTensorProcessForB8<SrcType>(dstUb, srcUb, offset, scale, srcCalCount);
}

template <typename SrcType, typename OutputDataType>
__simd_vf__ inline void PerchannelNoTransposeForB8(__ubuf__ OutputDataType* dst, __ubuf__ SrcType* src,
    __ubuf__ OutputDataType* offset, __ubuf__ OutputDataType* scale, const uint32_t K, const uint32_t N)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SrcType> vreg;
    MicroAPI::RegTensor<OutputDataType> scaleB16Vreg;
    MicroAPI::RegTensor<OutputDataType> offsetB16Vreg;
    MicroAPI::RegTensor<half> f16vreg;
    MicroAPI::RegTensor<OutputDataType> b16vreg;

    uint32_t sregLower = ANTIQUANT_B16_VF_LEN;
    uint32_t sreg = N;
    uint16_t repeat = CeilDivision(N, sregLower);

    for (uint16_t i = 0; i < repeat; ++i) {
        preg = MicroAPI::UpdateMask<uint16_t>(sreg);
        MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_NORM>(offsetB16Vreg, offset + i * sregLower);
        MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_NORM>(scaleB16Vreg, scale + i * sregLower);

        for (uint16_t j = 0; j < static_cast<uint16_t>(K); ++j) {
            MicroAPI::DataCopy<SrcType, MicroAPI::LoadDist::DIST_UNPACK_B8>(vreg, src + j * N + i * sregLower);
            MicroAPI::Cast<OutputDataType, SrcType, layoutZMrgZ>(b16vreg, vreg, preg);
            MicroAPI::Add(b16vreg, b16vreg, offsetB16Vreg, preg);
            MicroAPI::Mul(b16vreg, b16vreg, scaleB16Vreg, preg);
            MicroAPI::DataCopy<OutputDataType, MicroAPI::StoreDist::DIST_NORM_B16>(
                dst + j * N + i * sregLower, b16vreg, preg);
        }
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void PerchannelNoTransposeForB4(__ubuf__ OutputDataType *dst, __ubuf__ SrcType *src,
    __ubuf__ OutputDataType * offset, __ubuf__ OutputDataType *scale, const uint32_t K, const uint32_t N)
    {
        uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B16_VF_LEN);
        uint32_t sregLowerDivTwo = static_cast<uint32_t>(CeilDivision(sregLower, 2));
        uint32_t nDivTwo = static_cast<uint32_t>(CeilDivision(N,2));
        uint32_t nDivTwoBak = nDivTwo;
        uint32_t sreg = N;
        uint16_t repeat =CeilDivision(N, sregLower);

        __VEC_SCOPE__
        {
            MicroAPI::MaskReg preg;
            MicroAPI::MaskReg castPreg;
            MicroAPI::RegTensor<uint8_t> vreg;
            MicroAPI::RegTensor<OutputDataType> scaleB16Vreg;
            MicroAPI::RegTensor<OutputDataType> offsetB16Vreg;
            MicroAPI::RegTensor<OutputDataType> b16Vreg;

            for (uint16_t i = 0; i < repeat; ++i) {
                preg = MicroAPI::UpdateMask<uint16_t>(sreg);
                castPreg = MicroAPI::UpdateMask<uint32_t>(nDivTwoBak);

                MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_NORM>(offsetB16Vreg, offset + i * sregLower);
                MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_NORM>(scaleB16Vreg, scale + i * sregLower);

                for (uint16_t j = 0; j < static_cast<uint16_t>(K); ++j){
                    MicroAPI::DataCopy<uint8_t,MicroAPI::LoadDist::DIST_NORM>(
                        vreg, (__ubuf__ uint8_t *)src + j * nDivTwo + i * sregLowerDivTwo);
                    MicroAPI::UnPack<uint16_t, uint8_t>(
                        (MicroAPI::RegTensor<uint16_t> &)vreg, (MicroAPI::RegTensor<uint8_t> &)vreg); 
                    MicroAPI::UnPack<uint32_t, uint16_t>(
                        (MicroAPI::RegTensor<uint32_t> &)vreg, (MicroAPI::RegTensor<uint16_t> &)vreg); 
                    MicroAPI::Cast<OutputDataType, int4x2_t, castTraitB42Fp16>(
                        b16Vreg, (MicroAPI::RegTensor<int4x2_t> &)vreg, castPreg);
                    MicroAPI::Add(b16Vreg, b16Vreg, offsetB16Vreg, preg);
                    MicroAPI::Mul(b16Vreg, b16Vreg, scaleB16Vreg, preg);
                    MicroAPI::DataCopy<OutputDataType>(dst + j * N + i * sregLower, b16Vreg, preg);
                }
            }
        }
    }

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AntiQuantPerchannelNoTranspose(const LocalTensor<OutputDataType>& dst,
    const LocalTensor<SrcType>& src, const LocalTensor<OutputDataType>& offset,
    const LocalTensor<OutputDataType>& scale, const uint32_t K, const uint32_t N)
{
    __ubuf__ OutputDataType* scaleUb = (__ubuf__ OutputDataType*)scale.GetPhyAddr();
    __ubuf__ OutputDataType* offsetUb = (__ubuf__ OutputDataType*)offset.GetPhyAddr();
    __ubuf__ OutputDataType* dstUb = (__ubuf__ OutputDataType*)dst.GetPhyAddr();
    __ubuf__ SrcType* srcUb = (__ubuf__ SrcType*)src.GetPhyAddr();

    if constexpr (SupportType <SrcType, int4b_t>()) {
        PerchannelNoTransposeForB4<SrcType, OutputDataType>(dstUb, srcUb, offsetUb, scaleUb, K, N);
    } else {
        PerchannelNoTransposeForB8<SrcType, OutputDataType>(dstUb, srcUb, offsetUb, scaleUb, K, N);
    }
}

template <typename SrcType, typename OutputDataType>
__simd_vf__ inline void PerchannelUnlignedForB8(__ubuf__ OutputDataType* dst, __ubuf__ SrcType* src,
    __ubuf__ OutputDataType* offset, __ubuf__ OutputDataType* scale, const uint32_t srcCalCount)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SrcType> vreg;
    MicroAPI::RegTensor<OutputDataType> b16vreg;
    MicroAPI::RegTensor<OutputDataType> scaleB16Vreg;
    MicroAPI::RegTensor<OutputDataType> offsetB16Vreg;
    MicroAPI::RegTensor<half> f16vreg;

    uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B16_VF_LEN);
    uint32_t sreg = static_cast<uint32_t>(srcCalCount);
    uint16_t repeat = CeilDivision(srcCalCount, sregLower);

    MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_BLK>(offsetB16Vreg, offset);
    MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_BLK>(scaleB16Vreg, scale);

    for (uint16_t i = 0; i < static_cast<uint16_t>(repeat); ++i) {
        preg = MicroAPI::UpdateMask<uint16_t>(sreg);
        MicroAPI::DataCopy<SrcType, MicroAPI::LoadDist::DIST_UNPACK_B8>(vreg, src + i * sregLower);
        MicroAPI::Cast<OutputDataType, SrcType, layoutZMrgZ>(b16vreg, vreg, preg);
        MicroAPI::Add<OutputDataType, MicroAPI::MaskMergeMode::ZEROING>(b16vreg, b16vreg, offsetB16Vreg, preg);
        MicroAPI::Mul<OutputDataType, MicroAPI::MaskMergeMode::ZEROING>(b16vreg, b16vreg, scaleB16Vreg, preg);
        MicroAPI::DataCopy<OutputDataType, MicroAPI::StoreDist::DIST_NORM_B16>(dst + i * sregLower, b16vreg, preg);
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AntiQuantUnlignedProcess(const LocalTensor<OutputDataType>& dst, const LocalTensor<SrcType>& src, 
    const LocalTensor<OutputDataType>& offset, const LocalTensor<OutputDataType>& scale, const uint32_t K, 
    const uint32_t N)
{
    __ubuf__ OutputDataType* scaleUb = (__ubuf__ OutputDataType*)scale.GetPhyAddr();
    __ubuf__ OutputDataType* offsetUb = (__ubuf__ OutputDataType*)offset.GetPhyAddr();
    __ubuf__ OutputDataType* dstUb = (__ubuf__ OutputDataType*)dst.GetPhyAddr();
    __ubuf__ SrcType* srcUb = (__ubuf__ SrcType*)src.GetPhyAddr();

    PerchannelUnlignedForB8<SrcType, OutputDataType>(dstUb, srcUb, offsetUb, scaleUb, N * K);
}

template <typename SrcType>
__simd_vf__ inline void PerchannelTransposeForB8(__ubuf__ half* dst, __ubuf__ SrcType* src, __ubuf__ half* offset, 
    __ubuf__ half* scale, const uint32_t K, const uint32_t N)
{
    MicroAPI::MaskReg preg;

    MicroAPI::RegTensor<SrcType> vreg;
    MicroAPI::RegTensor<half> scaleB16Vreg;
    MicroAPI::RegTensor<half> offsetB16Vreg;
    MicroAPI::RegTensor<half> f16vreg;

    uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B16_VF_LEN);
    uint16_t repeat = CeilDivision(K, sregLower);

    for (uint16_t i = 0; i < static_cast<uint16_t>(N); i++) {
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_BRC_B16>(scaleB16Vreg, scale + i);
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_BRC_B16>(offsetB16Vreg, offset + i);

        uint32_t sreg = static_cast<uint32_t>(K);
        for (uint16_t j = 0; j < static_cast<uint16_t>(repeat); ++j) {
            preg = MicroAPI::UpdateMask<uint16_t>(sreg);
            MicroAPI::DataCopy<SrcType, MicroAPI::LoadDist::DIST_UNPACK_B8>(vreg, src + i * K + j * sregLower);
            MicroAPI::Cast<half, SrcType, layoutZMrgZ>(f16vreg, vreg, preg); // hif8->f16 or int8->f16

            MicroAPI::Add<half, MicroAPI::MaskMergeMode::ZEROING>(f16vreg, f16vreg, offsetB16Vreg, preg);
            MicroAPI::Mul<half, MicroAPI::MaskMergeMode::ZEROING>(f16vreg, f16vreg, scaleB16Vreg, preg);

            MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_NORM_B16>(dst + i * K + j * sregLower, f16vreg, preg);
        }
    }
}

template <typename SrcType>
__aicore__ inline void PerchannelTransposeForB4(__ubuf__ half *dst, __ubuf__ SrcType *src, __ubuf__ half * offset, 
    __ubuf__ half *scale, const uint32_t K, const uint32_t N)
    {
        uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B16_VF_LEN);
        uint32_t sregLowerDivTwo = static_cast<uint32_t>(CeilDivision(sregLower, 2));
        uint32_t kDivTwo = static_cast<uint32_t>(CeilDivision(K,2));
        uint16_t repeat =CeilDivision(K, sregLower);
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg preg;
            MicroAPI::MaskReg castPreg;

            MicroAPI::RegTensor<uint8_t> vreg;
            MicroAPI::RegTensor<half> scaleB16Vreg;
            MicroAPI::RegTensor<half> offsetB16Vreg;
            MicroAPI::RegTensor<half> f16Vreg;

            for (uint16_t i = 0; i < static_cast<uint16_t>(N); i++) {
                MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_BRC_B16>(scaleB16Vreg, scale + i);
                MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_BRC_B16>(offsetB16Vreg, offset + i);

                uint32_t sreg0 = static_cast<uint32_t>(K);
                uint32_t sreg1 = static_cast<uint32_t>(kDivTwo);
                for (uint16_t j = 0; j < static_cast<uint16_t>(repeat); ++j){
                    preg = MicroAPI::UpdateMask<uint16_t>(sreg0);
                    castPreg = MicroAPI::UpdateMask<uint32_t>(sreg1);

                    MicroAPI::DataCopy<uint8_t>(vreg, (__ubuf__ uint8_t *)src + i * kDivTwo + j * sregLowerDivTwo);
                    MicroAPI::UnPack<uint16_t, uint8_t>(
                        (MicroAPI::RegTensor<uint16_t> &)vreg, (MicroAPI::RegTensor<uint8_t> &)vreg); 
                    MicroAPI::UnPack<uint32_t, uint16_t>(
                        (MicroAPI::RegTensor<uint32_t> &)vreg, (MicroAPI::RegTensor<uint16_t> &)vreg); 

                    MicroAPI::Cast<half, int4x2_t, castTraitB42Fp16>(
                        f16Vreg, (MicroAPI::RegTensor<int4x2_t> &)vreg, castPreg);

                    MicroAPI::Add<half,MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, offsetB16Vreg, preg);
                    MicroAPI::Mul<half,MicroAPI::MaskMergeMode::ZEROING>(f16Vreg, f16Vreg, scaleB16Vreg, preg);

                    MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_NORM_B16>(
                        dst + i * K + j * sregLower, f16Vreg, preg);
                }
            }
        }
    }

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AntiQuantPerchannelTranspose(const LocalTensor<OutputDataType>& dst,
    const LocalTensor<SrcType>& src, const LocalTensor<OutputDataType>& offset,
    const LocalTensor<OutputDataType>& scale, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t K,
    const uint32_t N)
{
    __ubuf__ OutputDataType* dstUb = (__ubuf__ OutputDataType*)dst.GetPhyAddr();
    __ubuf__ SrcType* srcUb = (__ubuf__ SrcType*)src.GetPhyAddr();
    __ubuf__ OutputDataType* scaleUb = (__ubuf__ OutputDataType*)scale.GetPhyAddr();
    __ubuf__ OutputDataType* offsetUb = (__ubuf__ OutputDataType*)offset.GetPhyAddr();

    if constexpr (SupportType<SrcType, int4b_t>()){
        PerchannelTransposeForB4<SrcType>(dstUb, srcUb, offsetUb, scaleUb, K, N);
    } else {
        PerchannelTransposeForB8<SrcType>(dstUb, srcUb, offsetUb, scaleUb, K, N);
    }
}

template <typename SrcType, typename OutputDataType, bool isTranspose>
__aicore__ inline void AntiQuantPerchannelImpl(const LocalTensor<OutputDataType>& dst, const LocalTensor<SrcType>& src, 
    const LocalTensor<OutputDataType>& offset, const LocalTensor<OutputDataType>& scale, 
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t k, const AntiQuantShapeInfo& shapeInfo = {})
{
    static_assert(SupportType<SrcType, int8_t, int4b_t>(), "This AscendAntiQuant only support int8/int4 input dtype");
    static_assert(SupportType<OutputDataType, half>(), "This AscendAntiQuant only support f16 output dtype");

    if constexpr (isTranspose) { // src [n,k] offset [n,1]
        uint32_t n = (shapeInfo.offsetWidth == 0 ? offset.GetShapeInfo().shape[0] : shapeInfo.offsetWidth);
        AntiQuantPerchannelTranspose(dst, src, offset, scale, sharedTmpBuffer, k, n);
    } else { // src [k,n] offset [1,n]
        uint32_t n = (shapeInfo.offsetWidth == 0 ? offset.GetShapeInfo().shape[1] : shapeInfo.offsetWidth);
        if (n < 32) { // b8 input single line is not 32B aligned such as input n == 16
            if constexpr (SupportType<SrcType, int8_t>()) {
                ASCENDC_ASSERT((k % 2 == 0), { KERNEL_LOG(KERNEL_ERROR, "input calculate size must be 32B aligned"); });
                AntiQuantUnlignedProcess<SrcType, OutputDataType>(dst, src, offset, scale, k, n);
            } else {
                ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "not support int4b_t for AntiQuant when n < 32"); });
            }
            
        } else {
            AntiQuantPerchannelNoTranspose<SrcType, OutputDataType>(dst, src, offset, scale, k, n);
        }
    }
}

template <typename OutputDataType>
__aicore__ inline bool AntiQuantCheckPerformanceMode(
    const LocalTensor<OutputDataType> &scale, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K)
{
    return true;
}

// Brcb version
// allocate tmp buffer
template <typename OutputDataType>
__aicore__ inline void GetAntiquantTensorInfo(
    const LocalTensor<OutputDataType> &scale, const LocalTensor<float> &stackBuffer, AntiquantParams<float> &params)
{
    uint32_t N = scale.GetSize();                                  // scale and offset are shape [N]
    params.tempTensorOffset = stackBuffer[0];                      // store 8 * N * FP32    N -> brcb -> 8 * N
    params.tempTensorScale = stackBuffer[ANTIQUANT_BRCB_BASE * N]; // store 8 * N * FP32    N -> brcb -> 8 * N
    params.tempTensorInput = 
        stackBuffer[ANTIQUANT_BRCB_BASE * ANTIQUANT_TWO * N]; // need [N * 64 * FP32, N * K * FP32]
}

// 1. BF16 / FP16 -> cast -> FP32      2. N -> brcb -> 8 * N
// nLength means shape [N] for offset and scale
template <typename OutputDataType, bool withOffset = true>
__aicore__ inline void CastAndBrcb(const LocalTensor<OutputDataType> &offset, const LocalTensor<OutputDataType> &scale,
    AntiquantParams<float> &params, const uint32_t nLength)
{
    UnaryRepeatParams unaryParams;
    unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    uint32_t N = offset.GetSize();

    // shape [N]  BF16/ FP16 offset, scale -> cast -> FP32
    SetVectorMask<half, MaskMode::COUNTER>(0, nLength);
    if constexpr (withOffset) {
        Cast<float, OutputDataType>(
            params.tempTensorOffset[ANTIQUANT_BRCB_BASE * N - nLength], offset, RoundMode::CAST_NONE, nLength);
    }
    Cast<float, OutputDataType>(
        params.tempTensorScale[ANTIQUANT_BRCB_BASE * N - nLength], scale, RoundMode::CAST_NONE, nLength);
    PipeBarrier<PIPE_V>();

    constexpr uint16_t brcbDstBlkStride = 1;                   // 1 num -> 8 num(1 block)
    constexpr uint16_t brcbDstRepStride = ANTIQUANT_BRCB_BASE; // 1 brcb: 8 num -> 64 num
    const uint8_t repeatTimes = nLength / ANTIQUANT_BRCB_BASE; // 1 brcb cmd needs 8 input num
    BrcbRepeatParams brcbParams(brcbDstBlkStride, brcbDstRepStride);

    SetMaskNorm();
    ResetMask();
    // brcb: 1 FP32 A -> 1 block contains 8 FP32 A, after 1 block, do the same to the next FP32 B
    if constexpr (withOffset) {
        Brcb(params.tempTensorOffset, 
        params.tempTensorOffset[ANTIQUANT_BRCB_BASE * N - nLength], 
        repeatTimes,
        brcbParams);
        PipeBarrier<PIPE_V>();
    }
    Brcb(params.tempTensorScale, params.tempTensorScale[ANTIQUANT_BRCB_BASE * N - nLength], repeatTimes, brcbParams);
    PipeBarrier<PIPE_V>();
    SetMaskCount();
}

// scale * (src + offset)   src: N * K, scale: N, offset: N  NOffset: offset used for tmpTensorOffset, tmpTensorScale
// For now, calCount must equal to N * K then can use brcb   calCount: 64 * N
template <typename SrcType, typename OutputDataType, bool withOffset>
__aicore__ inline void CalculationMin(const LocalTensor<SrcType>& src, const LocalTensor<OutputDataType>& dst,
    AntiquantParams<float>& params, const uint32_t calCount, const uint32_t n, const uint32_t srcN, const uint32_t k)
{
    // store FP16 result in second half of FP32 tmpTensor to avoid input FP16 being replaced
    uint32_t srcFp16Pos = calCount / ANTIQUANT_TWO; // therefore start from (calCount / 2)th FP32 tmpTensor
    uint32_t n1 = k / ANTIQUANT_SINGLE_N_SIZE;       // K = 64 * n1
    UnaryRepeatParams unaryParamsInt8Fp16;
    unaryParamsInt8Fp16.srcRepStride = ANTIQUANT_TWO * n1; // K(num) / 32(num per block)
    // one repeat calculate 64 int8 -> 64 fp16, 4 block
    unaryParamsInt8Fp16.dstRepStride = ANTIQUANT_SINGLE_N_SIZE / (ONE_BLK_SIZE / sizeof(half));
    UnaryRepeatParams unaryParamsFp16Fp32;
    unaryParamsFp16Fp32.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;

    // Must use NORM for calculation instead of counter
    SetMaskNorm();
    SetVectorMask<half, MaskMode::NORMAL>(0, FULL_MASK); // the first 64 num for calculation
    // INT8 -> FP16
    auto fp16TmpBuffer = params.tempTensorInput[srcFp16Pos].ReinterpretCast<half>();
    Cast<half, int8_t, false>(fp16TmpBuffer, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, n, unaryParamsInt8Fp16);
    PipeBarrier<PIPE_V>();
    // FP16 -> FP32
    Cast<float, half, false>(
        params.tempTensorInput, fp16TmpBuffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER, n,  unaryParamsFp16Fp32);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    BinaryRepeatParams binaryParams;
    binaryParams.src1BlkStride = 0; // same line for add and mul
    binaryParams.src1RepStride = 1; // one line for 64 num calculation

    SetVectorMask<float, MaskMode::COUNTER>(0, ANTIQUANT_SINGLE_N_SIZE * n);
    // scale * (src + offset)
    if constexpr (withOffset) {
        Add<float>(
            params.tempTensorInput, params.tempTensorInput, params.tempTensorOffset,  ANTIQUANT_SINGLE_N_SIZE * n);
        PipeBarrier<PIPE_V>();
    }
    Mul<float>(params.tempTensorInput, params.tempTensorInput, params.tempTensorScale, ANTIQUANT_SINGLE_N_SIZE * n);
    PipeBarrier<PIPE_V>();

    // FP32 -> BF16
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(0, FULL_MASK);
    UnaryRepeatParams f322f16Params;
    f322f16Params.dstRepStride = ANTIQUANT_SINGLE_N_SIZE * n1 / (ONE_BLK_SIZE / sizeof(half));
    Cast<OutputDataType, float, false>(
        dst, params.tempTensorInput, RoundMode::CAST_RINT, MASK_PLACEHOLDER, srcN,  f322f16Params);
    PipeBarrier<PIPE_V>();
}

// Method2: min: N * 64
template <typename SrcType, typename OutputDataType>
__aicore__ inline void CalculateByBrcbMin(const LocalTensor<OutputDataType>& dst, const LocalTensor<SrcType>& src,
    const LocalTensor<OutputDataType>& offset, const LocalTensor<OutputDataType>& scale,
    const LocalTensor<float>& stackBuffer, const uint32_t calCount, const uint32_t n, const uint32_t k)
{
    AntiquantParams<float> antiquantParams;
    GetAntiquantTensorInfo<OutputDataType>(scale, stackBuffer, antiquantParams);

    SetMaskCount();
    CastAndBrcb<OutputDataType, true>(offset, scale, antiquantParams, n); // store FP32 offset and scale into params

    uint32_t curNKOffset = 0;
    uint32_t loopNum = k / ANTIQUANT_SINGLE_N_SIZE;
    uint32_t srcN = src.GetSize() / k;
    // calculate  N * 64
    for (uint32_t i = 0; i < loopNum; i++) {
        curNKOffset = ANTIQUANT_SINGLE_N_SIZE * i;
        CalculationMin<SrcType, OutputDataType, true>(
            src[curNKOffset], dst[curNKOffset], antiquantParams, ANTIQUANT_SINGLE_N_SIZE * n, n, srcN, k);
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void CalculateByBrcbMin(const LocalTensor<OutputDataType>& dst, const LocalTensor<SrcType>& src,
    const LocalTensor<OutputDataType>& scale, const LocalTensor<float>& stackBuffer, const uint32_t calCount,
    const uint32_t n, const uint32_t k)
{
    AntiquantParams<float> antiquantParams;
    GetAntiquantTensorInfo<OutputDataType>(scale, stackBuffer, antiquantParams);

    SetMaskCount();
    CastAndBrcb<OutputDataType, false>(scale, scale, antiquantParams, n); // store FP32 offset and scale into params

    uint32_t curNKOffset = 0;
    uint32_t loopNum = k / ANTIQUANT_SINGLE_N_SIZE;
    uint32_t srcN = src.GetSize() / k;
    // calculate  N * 64
    for (uint32_t i = 0; i < loopNum; i++) {
        curNKOffset = ANTIQUANT_SINGLE_N_SIZE * i;
        CalculationMin<SrcType, OutputDataType, false>(
            src[curNKOffset], dst[curNKOffset], antiquantParams, ANTIQUANT_SINGLE_N_SIZE * n, n, srcN, k);
    }
}

template <typename OutputDataType>
__aicore__ inline void CalculateByBrcbMin(const LocalTensor<OutputDataType>& dst, const LocalTensor<int4b_t>& src,
    const LocalTensor<OutputDataType>& scale, const LocalTensor<float>& stackBuffer, const uint32_t calCount,
    const uint32_t n, const uint32_t k)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported type: int4b_t for AntiQuant"); });
}

template <typename OutputDataType>
__aicore__ inline void CalculateByBrcbMin(const LocalTensor<OutputDataType>& dst, const LocalTensor<int4b_t>& src,
    const LocalTensor<OutputDataType>& offset, const LocalTensor<OutputDataType>& scale,
    const LocalTensor<float>& stackBuffer, const uint32_t calCount, const uint32_t n, const uint32_t k)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported type: int4b_t for AntiQuant"); });
}

template <bool withOffset = true>
__aicore__ inline void AntiQuantFp16Brcb(const LocalTensor<half> &scale, const LocalTensor<half> &offset,
    AntiquantParams<half> &params, const uint32_t scaleN)
{
    // step 1: do brcb for scale and offset
    const uint8_t repeatTimes = scaleN / BRCB_BROADCAST_NUMBER;
    BrcbRepeatParams brcbParams(DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    SetMaskNorm();
    ResetMask();
    Brcb(params.tempTensorScale, scale, repeatTimes, brcbParams);
    PipeBarrier<PIPE_V>();
    if constexpr (withOffset) {
        Brcb(params.tempTensorOffset, offset, repeatTimes, brcbParams);
        PipeBarrier<PIPE_V>();
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AscendAntiQuantTranspose(const LocalTensor<OutputDataType>& dst, const LocalTensor<SrcType>& src,
    const LocalTensor<OutputDataType>& offset, const LocalTensor<OutputDataType>& scale,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t K, const AntiQuantShapeInfo& shapeInfo = {})
{
    uint32_t calCount = src.GetSize();
    uint32_t N = offset.GetSize();
    if constexpr (IsSameType<OutputDataType, half>::value || IsSameType<SrcType, int4b_t>::value) {
        return AntiQuantImplScalar(dst, src, offset, scale, sharedTmpBuffer, calCount, K, shapeInfo);
    }
    if (K > ANTIQUANT_MAX_K * ANTIQUANT_BRCB_BASE || (K % ANTIQUANT_SINGLE_N_SIZE != 0)) {
        return AntiQuantImplScalar(dst, src, offset, scale, sharedTmpBuffer, calCount, K, shapeInfo);
    }

    auto stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    // input and scale & offset
    uint32_t stackBufferSize = N * ANTIQUANT_SINGLE_N_SIZE + N * ANTIQUANT_BRCB_BASE * ANTIQUANT_TWO;
    stackBuffer.SetSize(stackBufferSize);
    CalculateByBrcbMin(dst, src, offset, scale, stackBuffer, calCount, N, K);
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AscendAntiQuantTranspose(const LocalTensor<OutputDataType>& dst, const LocalTensor<SrcType>& src,
    const LocalTensor<OutputDataType>& scale, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t K,
    const AntiQuantShapeInfo& shapeInfo = {})
{
    uint32_t calCount = src.GetSize();
    uint32_t N = scale.GetSize();
    if constexpr (IsSameType<OutputDataType, half>::value || IsSameType<SrcType, int4b_t>::value) {
        return AntiQuantImplScalar(dst, src, scale, sharedTmpBuffer, calCount, K, shapeInfo);
    }
    if (K > ANTIQUANT_MAX_K * ANTIQUANT_BRCB_BASE || (K % ANTIQUANT_SINGLE_N_SIZE != 0) ||
        IsSameType<SrcType, int4b_t>::value) {
        return AntiQuantImplScalar(dst, src, scale, sharedTmpBuffer, calCount, K, shapeInfo);
    }

    auto stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    uint32_t stackBufferSize = N * ANTIQUANT_SINGLE_N_SIZE + N * ANTIQUANT_BRCB_BASE * ANTIQUANT_TWO;
    stackBuffer.SetSize(stackBufferSize);
    CalculateByBrcbMin(dst, src, scale, stackBuffer, calCount, N, K);
}

template <typename scaleT, const AscendAntiQuantConfig& config>
__simd_callee__ inline void LoadPerTokenScaleAndOffset(__ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb,
    MicroAPI::RegTensor<scaleT>& scaleVreg,  MicroAPI::RegTensor<scaleT>& offsetVreg)
{
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B16>(scaleVreg, scaleUb);
        if constexpr (config.hasOffset) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B16>(offsetVreg, offsetUb);
        }
    } else {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B32>(scaleVreg, scaleUb);
        if constexpr (config.hasOffset) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B32>(offsetVreg, offsetUb);
        }
    }
}

template <typename scaleT, const AscendAntiQuantConfig& config>
__simd_callee__ inline void LoadPerTokenTransposeScaleAndOffset(__ubuf__ scaleT* scaleUb,  __ubuf__ scaleT* offsetUb,
    MicroAPI::RegTensor<scaleT>& scaleVreg,  MicroAPI::RegTensor<scaleT>& offsetVreg)
{
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb);
        if constexpr (config.hasOffset) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(offsetVreg, offsetUb);
        }
    } else {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb);
        if constexpr (config.hasOffset) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(offsetVreg, offsetUb);
        }
    }
}

template <typename T, const AscendAntiQuantConfig& config>
__simd_callee__ inline void GetPerGroupScaleAndOffset(__ubuf__ T* scaleUb,  __ubuf__ T* offsetUb, const int32_t start,
    const AscendAntiQuantParam& para, MicroAPI::RegTensor<T>& scaleReg, MicroAPI::RegTensor<T>& offsetReg)
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
        MicroAPI::Div(index_vreg, (MicroAPI::RegTensor<uint16_t>&)vci_vreg, gsize_vreg, preg);
        MicroAPI::DataCopyGather(scaleReg, scaleUb, index_vreg, preg);
        if constexpr (config.hasOffset) {
            MicroAPI::DataCopyGather(offsetReg, offsetUb, index_vreg, preg);
        }
    } else {
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int32_t> vci_vreg;
        MicroAPI::RegTensor<uint32_t> index_vreg;
        MicroAPI::RegTensor<uint32_t> gsize_vreg;
        MicroAPI::Duplicate(gsize_vreg, static_cast<uint32_t>(groupSize));
        MicroAPI::Arange(vci_vreg, static_cast<int32_t>(start));
        MicroAPI::Div(index_vreg, (MicroAPI::RegTensor<uint32_t>&)vci_vreg, gsize_vreg, preg);
        MicroAPI::DataCopyGather(scaleReg, scaleUb, index_vreg, preg);
        if constexpr (config.hasOffset) {
            MicroAPI::DataCopyGather(offsetReg, offsetUb, index_vreg, preg);
        }
    }
}

template <typename dstT>
__simd_callee__ inline void StoreF32Res(
    __ubuf__ dstT* dstAddr, MicroAPI::RegTensor<float>& vreg, MicroAPI::MaskReg& preg)
{
    if constexpr (SupportType<dstT, float>()) {
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(dstAddr, vreg, preg);
    } else {
        MicroAPI::RegTensor<dstT> tempVreg;
        MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(tempVreg, vreg, preg);
        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr, tempVreg, preg);
    }
}

template <typename scaleT, typename srcT>
__simd_callee__ inline void LoadSrc(__ubuf__ srcT* srcAddr, MicroAPI::RegTensor<srcT>& srcVreg)
{
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B8>(srcVreg, srcAddr);
    } else {
        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(srcVreg, srcAddr);
    }
}

template <typename scaleT, typename srcT>
__simd_callee__ inline void ConvertSrc(
    MicroAPI::RegTensor<scaleT>& vreg, MicroAPI::RegTensor<srcT>& srcVreg, MicroAPI::MaskReg& preg)
{
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::Cast<scaleT, srcT, layoutZMrgZ>(vreg, srcVreg, preg);
    } else {
        MicroAPI::RegTensor<int32_t> s32Vreg;
        if constexpr (SupportType<srcT, int8_t>()) {
            MicroAPI::Cast<int32_t, srcT, layoutZMrgZ>(s32Vreg, srcVreg, preg);
            MicroAPI::Cast<float, int32_t, MrgZRndA>(vreg, s32Vreg, preg);
        } else {
            MicroAPI::Cast<float, srcT, layoutZMrgZ>(vreg, srcVreg, preg);
        }
    }
}

template <typename scaleT, const AscendAntiQuantConfig& config>
__simd_callee__ inline void AddOffsetIfExist(
    MicroAPI::RegTensor<scaleT>& vreg, MicroAPI::RegTensor<scaleT>& offsetVreg, MicroAPI::MaskReg& preg)
{
    if constexpr (config.hasOffset) {
        MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, offsetVreg, preg);
    }
}

template <typename scaleT>
__simd_callee__ inline void GenZeroVreg(MicroAPI::RegTensor<scaleT>& vreg)
{
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::MaskReg b16FullPreg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(vreg, static_cast<scaleT>(0), b16FullPreg);
    } else {
        MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(vreg, static_cast<scaleT>(0), b32FullPreg);
    }
}

template <typename scaleT, const AscendAntiQuantConfig& config>
__simd_callee__ inline void ConvertToF32ScaleAndOffset(MicroAPI::RegTensor<scaleT>& scaleVreg,
    MicroAPI::RegTensor<scaleT>& offsetVreg, MicroAPI::MaskReg& preg, MicroAPI::RegTensor<float>& f32ScaleVreg,
    MicroAPI::RegTensor<float>& f32OffsetVreg)
{
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::RegTensor<scaleT> zeroVreg;
        GenZeroVreg<scaleT>(zeroVreg);
        MicroAPI::RegTensor<scaleT> tempOffsetVreg;
        MicroAPI::RegTensor<scaleT> tempScaleVreg;
        MicroAPI::RegTensor<scaleT> tempVreg;
        MicroAPI::Interleave(tempScaleVreg, tempVreg, scaleVreg, zeroVreg);
        MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32ScaleVreg, tempScaleVreg, preg);
        if constexpr (config.hasOffset) {
            MicroAPI::Interleave(tempOffsetVreg, tempVreg, offsetVreg, zeroVreg);
            MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32OffsetVreg, tempOffsetVreg, preg);
        }
    }
}

template <typename scaleT, const AscendAntiQuantConfig& config>
__simd_callee__ inline void LoadNormScaleAndOffset(__ubuf__ scaleT* scaleAddr, __ubuf__ scaleT* offsetAddr,
     MicroAPI::RegTensor<scaleT>& scaleVreg, MicroAPI::RegTensor<scaleT>& offsetVreg)
{
    MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleAddr);
    if constexpr (config.hasOffset) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(offsetVreg, offsetAddr);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerTokenForB8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb, __ubuf__ scaleT* scaleUb, 
    __ubuf__ scaleT* offsetUb, const AscendAntiQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = GetVecLen() / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint32_t sreg = para.n;

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<scaleT> vreg;
    for (uint16_t i = 0; i < rowNum; ++i) {
        LoadPerTokenScaleAndOffset<scaleT, config>(scaleUb + i, offsetUb + i, scaleVreg, offsetVreg);
        sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<scaleT>(sreg);
            LoadSrc<scaleT, srcT>((srcUb + i * para.n + j * vecLen), srcVreg);
            ConvertSrc<scaleT, srcT>(vreg, srcVreg, preg);
            AddOffsetIfExist<scaleT, config>(vreg, offsetVreg, preg);
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, scaleVreg, preg);
            if constexpr (SupportType<scaleT, half>()) {
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_NORM_B16>(
                    dstUb + i * para.n + j * vecLen, vreg, preg);
            } else {
                if constexpr (SupportType<dstT, float>()) {
                    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                        dstUb + i * para.n + j * vecLen, vreg, preg);
                } else {
                    MicroAPI::RegTensor<dstT> tempVreg;
                    MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(tempVreg, vreg, preg);
                    MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B32>(
                        dstUb + i * para.n + j * vecLen, tempVreg, preg);
                }
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantPerTokenForB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
 const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    __ubuf__ scaleT* offsetUb = (__ubuf__ scaleT*)offsetTensor.GetPhyAddr();
    AntiQuantPerTokenForB8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerTokenTransposeForB8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb, const AscendAntiQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = GetVecLen() / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint32_t sreg = para.n;

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<scaleT> vreg;
    for (uint16_t i = 0; i < rowNum; ++i) {
        sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<scaleT>(sreg);
            LoadPerTokenTransposeScaleAndOffset<scaleT, config>(
                scaleUb + j * vecLen, offsetUb + j * vecLen, scaleVreg, offsetVreg);
            LoadSrc<scaleT, srcT>((srcUb + i * para.n + j * vecLen), srcVreg);
            ConvertSrc<scaleT, srcT>(vreg, srcVreg, preg);
            AddOffsetIfExist<scaleT, config>(vreg, offsetVreg, preg);
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, scaleVreg, preg);
            if constexpr (SupportType<scaleT, half>()) {
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_NORM_B16>(
                    dstUb + i * para.n + j * vecLen, vreg, preg);
            } else {
                if constexpr (SupportType<dstT, float>()) {
                    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                        dstUb + i * para.n + j * vecLen, vreg, preg);
                } else {
                    MicroAPI::RegTensor<dstT> tempVreg;
                    MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(tempVreg, vreg, preg);
                    MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B32>(
                        dstUb + i * para.n + j * vecLen, tempVreg, preg);
                }
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantPerTokenTransposeForB8(const LocalTensor<dstT>& dstTensor,
    const LocalTensor<srcT>& srcTensor, const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
    const AscendAntiQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    __ubuf__ scaleT* offsetUb = (__ubuf__ scaleT*)offsetTensor.GetPhyAddr();
    AntiQuantPerTokenTransposeForB8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerGroupForColB8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb, const AscendAntiQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = GetVecLen() / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint32_t sreg = para.n;
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<scaleT> vreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    for (uint16_t i = 0; i < rowNum; ++i) {
        sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<scaleT>(sreg);
            GetPerGroupScaleAndOffset<scaleT, config>(
                scaleUb + i * scaleK, offsetUb + i * scaleK, j * vecLen, para, scaleVreg, offsetVreg);
            LoadSrc<scaleT, srcT>((srcUb + i * para.n + j * vecLen), srcVreg);
            ConvertSrc<scaleT, srcT>(vreg, srcVreg, preg);
            AddOffsetIfExist<scaleT, config>(vreg, offsetVreg, preg);
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, scaleVreg, preg);
            if constexpr (SupportType<scaleT, half>()) {
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_NORM_B16>(
                    dstUb + i * para.n + j * vecLen, vreg, preg);
            } else {
                if constexpr (SupportType<dstT, float>()) {
                    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                        dstUb + i * para.n + j * vecLen, vreg, preg);
                } else {
                    MicroAPI::RegTensor<dstT> tempVreg;
                    MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(tempVreg, vreg, preg);
                    MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B32>(
                        dstUb + i * para.n + j * vecLen, tempVreg, preg);
                }
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantPerGroupForColB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
  const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    __ubuf__ scaleT* offsetUb = (__ubuf__ scaleT*)offsetTensor.GetPhyAddr();
    AntiQuantPerGroupForColB8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_callee__ inline void AntiQuantPerGroupForRowB8TailBlock(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
   __ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb, uint16_t repeat, uint16_t tailRow, uint32_t n, uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<scaleT> vreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            LoadNormScaleAndOffset<scaleT, config>(
                (scaleUb + j * vecLen), (offsetUb + j * vecLen), scaleVreg, offsetVreg);
            preg = MicroAPI::UpdateMask<scaleT>(sreg);
            LoadSrc<scaleT, srcT>(srcUb + i * n + j * vecLen, srcVreg);
            ConvertSrc<scaleT, srcT>(vreg, srcVreg, preg);
            AddOffsetIfExist<scaleT, config>(vreg, offsetVreg, preg);
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, scaleVreg, preg);
            if constexpr (SupportType<scaleT, half>()) {
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_NORM_B16>(dstUb + i * n + j * vecLen, vreg, preg);
            } else {
                if constexpr (SupportType<dstT, float>()) {
                    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                        dstUb + i * n + j * vecLen, vreg, preg);
                } else {
                    MicroAPI::RegTensor<dstT> tempVreg;
                    MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(tempVreg, vreg, preg);
                    MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B32>(
                        dstUb + i * n + j * vecLen, tempVreg, preg);
                }
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerGroupForRowB8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb, const AscendAntiQuantParam para, uint16_t rowNum, 
    uint16_t tailRow)
{
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint32_t vecLen = GetVecLen() / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<scaleT> vreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    for (uint16_t i = 0; i < mainRowGroup; ++i) {
        for (uint16_t j = 0; j < static_cast<uint16_t>(para.groupSize); ++j) {
            uint32_t sreg = para.n;
            for (uint16_t k = 0; k < repeat; ++k) {
                LoadNormScaleAndOffset<scaleT, config>(
                    (scaleUb + i * para.n + k * vecLen), (offsetUb + i * para.n + k * vecLen), scaleVreg, offsetVreg);
                preg = MicroAPI::UpdateMask<scaleT>(sreg);
                LoadSrc<scaleT, srcT>(srcUb + (i * para.groupSize + j) * para.n + k * vecLen, srcVreg);
                ConvertSrc<scaleT, srcT>(vreg, srcVreg, preg);
                AddOffsetIfExist<scaleT, config>(vreg, offsetVreg, preg);
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, scaleVreg, preg);
                if constexpr (SupportType<scaleT, half>()) {
                    MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_NORM_B16>(
                        dstUb + (i * para.groupSize + j) * para.n + k * vecLen, vreg, preg);
                } else {
                    if constexpr (SupportType<dstT, float>()) {
                        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                            dstUb + (i * para.groupSize + j) * para.n + k * vecLen, vreg, preg);
                    } else {
                        MicroAPI::RegTensor<dstT> tempVreg;
                        MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(tempVreg, vreg, preg);
                        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B32>(
                            dstUb + (i * para.groupSize + j) * para.n + k * vecLen, tempVreg, preg);
                    }
                }
            }
        }
    }
    AntiQuantPerGroupForRowB8TailBlock<dstT, srcT, scaleT, config>( dstUb + mainRowGroup * para.groupSize * para.n, 
        srcUb + mainRowGroup * para.groupSize * para.n,
        scaleUb + mainRowGroup * para.n, 
        offsetUb + mainRowGroup * para.n, 
        repeat, 
        tailRow, 
        para.n, 
        vecLen);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantPerGroupForRowB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    __ubuf__ scaleT* offsetUb = (__ubuf__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t tailRow = rowNum % para.groupSize;
    AntiQuantPerGroupForRowB8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para, rowNum, tailRow);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AscendAntiQuantPerToken(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                               const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                               const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    if constexpr (config.isTranspose) {
        AntiQuantPerTokenTransposeForB8<dstT, srcT, scaleT, config>(
            dstTensor, srcTensor, scaleTensor, offsetTensor, para);
        return;
    }
    AntiQuantPerTokenForB8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AscendAntiQuantPerGroupForCol(const LocalTensor<dstT>& dstTensor, 
    const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, 
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    AntiQuantPerGroupForColB8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AscendAntiQuantPerGroupForRow(const LocalTensor<dstT>& dstTensor, 
    const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, 
    const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    AntiQuantPerGroupForRowB8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config,
    const AscendAntiQuantPolicy& policy>
__aicore__ inline void AscendAntiQuantImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
    const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(scaleTensor, "scaleTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(offsetTensor, "offsetTensor", "VECIN, VECOUT, VECCALC");
    static_assert(SupportType<dstT, half, float>(), "AscendAntiQuant only support half/float output dtype");
    static_assert(SupportType<scaleT, half, float>(), "AscendAntiQuant only support half/float scale/offset dtype");
    static_assert((policy == AscendAntiQuantPolicy::PER_TOKEN || policy == AscendAntiQuantPolicy::PER_GROUP),
        "unsupported policy for AscendAntiQuant in current device!");
    ASCENDC_ASSERT(
        (para.calCount <= srcTensor.GetSize() && para.calCount <= dstTensor.GetSize() && para.calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, 
            "calCount is %u, which should be in [0, min(%u, %u)]",
            para.calCount, 
            srcTensor.GetSize(), 
            dstTensor.GetSize());
    });
    ASCENDC_ASSERT(
        (para.calCount % para.n == 0), { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of n!"); });
    if constexpr (policy == AscendAntiQuantPolicy::PER_TOKEN) {
        static_assert(SupportType<srcT, int8_t>(), "AscendAntiQuant PerToken only support int8_t input dtype");
        AscendAntiQuantPerToken<dstT, srcT, scaleT, config>(
            dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
    } else if constexpr (policy == AscendAntiQuantPolicy::PER_GROUP) {
        static_assert( SupportType<srcT, int8_t>(), "AscendAntiQuant PerGroup only support int8_t input dtype");
        static_assert(
            ((config.kDim == 1) || (config.kDim == 0)), "AscendAntiQuant PerGroup only support K is axis 0/1!");
        ASCENDC_ASSERT((para.groupSize > 0 && para.groupSize % 32 == 0),
            { KERNEL_LOG(KERNEL_ERROR, "groupSize must be an integer multiple of 32 and greater than 0 !"); });
        if constexpr ((config.kDim == 1 && !config.isTranspose) || (config.kDim == 0 && config.isTranspose)) {
            AscendAntiQuantPerGroupForCol<dstT, srcT, scaleT, config>(
                dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
        } else {
            AscendAntiQuantPerGroupForRow<dstT, srcT, scaleT, config>(
                dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
        }
    }
}

template <typename SrcType, typename DstType, bool isTranspose>
__aicore__ inline void AscendAntiQuantImplCommon(const LocalTensor<DstType>& dst, const LocalTensor<SrcType>& src,
    const LocalTensor<DstType>& offset, const LocalTensor<DstType>& scale, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const uint32_t k, const AntiQuantShapeInfo& shapeInfo = {})
{
    CheckTensorPosition(dst, "dst", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(src, "src", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(offset, "offset", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(scale, "scale", "VECIN, VECOUT, VECCALC");
    AntiQuantPerchannelImpl<SrcType, DstType, isTranspose>(dst, src, offset, scale, sharedTmpBuffer, k, shapeInfo);
}

template <typename SrcType, typename DstType, bool isTranspose>
__aicore__ inline void AscendAntiQuantImplCommon(const LocalTensor<DstType>& dst, const LocalTensor<SrcType>& src,
    const DstType offset, const DstType scale, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t k,
    const AntiQuantShapeInfo& shapeInfo = {})
{
    CheckTensorPosition(dst, "dst", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(src, "src", "VECIN, VECOUT, VECCALC");
    AntiQuantPertensorImpl<SrcType, DstType>(dst, src, offset, scale, sharedTmpBuffer, k, shapeInfo);
}
}  // namespace AscendC
#endif  // IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_L300_IMPL_H
