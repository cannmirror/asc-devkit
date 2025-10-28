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

/* !
 * \file ascend_antiquant_c310_impl.h
 * \brief
 */
#ifndef IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_C310_IMPL_H
#define IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "ascend_antiquant_common.h"

namespace AscendC {
constexpr uint32_t ANTIQUANT_B16_VF_LEN = VECTOR_REG_WIDTH / sizeof(uint16_t);
constexpr uint32_t ANTIQUANT_B32_VF_LEN = VECTOR_REG_WIDTH / sizeof(uint32_t);

template <typename SrcType, typename OutType>
__aicore__ inline void CheckApiDtypeValid()
{
    constexpr bool inputValid = (IsSameType<SrcType, int8_t>::value) || (IsSameType<SrcType, int4b_t>::value);
    constexpr bool outputValid = (IsSameType<OutType, half>::value) || (IsSameType<OutType, bfloat16_t>::value);
    ASCENDC_ASSERT((inputValid && outputValid), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in AscendAntiQuant, "
        "current api support dtype combination is src: int8_t / int4b_t, dst: half / bfloat16_t.");});
}

__simd_callee__ inline void SelectZeroNan(MicroAPI::RegTensor<bfloat16_t>& b16vreg, MicroAPI::RegTensor<uint16_t>& bf16Zero,
    MicroAPI::RegTensor<uint16_t>& bf16Nan, MicroAPI::RegTensor<uint16_t>& e8m0Zero,
    MicroAPI::RegTensor<uint16_t>& e8m0Nan, MicroAPI::MaskReg& selPreg, MicroAPI::MaskReg& preg)
{
    Compare<uint16_t, CMPMODE::NE>(selPreg, (MicroAPI::RegTensor<uint16_t>&)b16vreg, e8m0Zero, preg);
    Select<uint16_t>((MicroAPI::RegTensor<uint16_t>&)b16vreg, (MicroAPI::RegTensor<uint16_t>&)b16vreg, bf16Zero,
        selPreg);
    Compare<uint16_t, CMPMODE::NE>(selPreg, (MicroAPI::RegTensor<uint16_t>&)b16vreg, e8m0Nan, preg);
    Select<uint16_t>((MicroAPI::RegTensor<uint16_t>&)b16vreg, (MicroAPI::RegTensor<uint16_t>&)b16vreg, bf16Nan,
        selPreg);
}

template <typename OutputDataType>
__aicore__ inline void CastScale(__local_mem__ OutputDataType* dst, __local_mem__ fp8_e8m0_t* scale,
    const uint32_t srcCalCount)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<uint8_t> vreg;
    MicroAPI::RegTensor<bfloat16_t> b16vreg;
    MicroAPI::RegTensor<half> halfvreg;
    MicroAPI::MaskReg selPreg;
    MicroAPI::RegTensor<uint16_t> bf16Zero;
    MicroAPI::RegTensor<uint16_t> bf16Nan;
    MicroAPI::RegTensor<uint16_t> e8m0Zero;
    MicroAPI::RegTensor<uint16_t> e8m0Nan;
    Duplicate(bf16Zero, (uint16_t)0x0040); // if e8m0 = 0b00000000, bf16 is 0x0040
    Duplicate(bf16Nan, (uint16_t)0x7fff);  // if e8m0 = 0b11111111, use 0x7fff as bf16 nan
    Duplicate(e8m0Zero, 0);
    Duplicate(e8m0Nan, (uint16_t)0x7f80); // if e8m0 = 0b11111111, after << 7 is 0x7f80
    uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B16_VF_LEN);
    uint32_t scaleCalCount = srcCalCount / ANTIQUANT_FP4_PERGROUP_SIZE; // perGroupSize = 32 default
    uint32_t sreg = static_cast<uint32_t>(scaleCalCount);
    uint16_t repeatScale = CeilDivision(scaleCalCount, sregLower);

    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatScale); ++i) {
        preg = MicroAPI::UpdateMask<uint16_t>(sreg);
        MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(vreg, scale + i * sregLower);
        MicroAPI::ShiftLefts<uint16_t, int16_t>((MicroAPI::RegTensor<uint16_t>&)b16vreg,
            (MicroAPI::RegTensor<uint16_t>&)vreg, ANTIQUANT_BF16_MAN_LEN, preg);

        // 00000000 and 11111111 need special process
        SelectZeroNan(b16vreg, bf16Zero, bf16Nan, e8m0Zero, e8m0Nan, selPreg, preg);

        if constexpr (SupportType<OutputDataType, half>()) {
            MicroAPI::Cast<half, bfloat16_t, MrgZRndRSatS>(halfvreg, b16vreg, preg);
            MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_NORM_B16>(dst + i * sregLower, halfvreg, preg);
        } else {
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_NORM_B16>(dst + i * sregLower, b16vreg, preg);
        }
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AntiQuantProcessByLine(__local_mem__ OutputDataType* dstUb, __local_mem__ SrcType* srcUb,
    __local_mem__ fp8_e8m0_t* scaleUb, const uint32_t calCount, MicroAPI::RegTensor<uint16_t>& bf16Zero,
    MicroAPI::RegTensor<uint16_t>& bf16Nan, MicroAPI::RegTensor<uint16_t>& e8m0Zero,
    MicroAPI::RegTensor<uint16_t>& e8m0Nan)
{
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<bfloat16_t> b16vreg1;
    MicroAPI::RegTensor<bfloat16_t> b16vreg2;
    MicroAPI::RegTensor<SrcType> fp4vreg0;
    MicroAPI::RegTensor<half> halfvreg1;
    MicroAPI::RegTensor<half> halfvreg2;
    MicroAPI::MaskReg selPreg;

    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg preg;
    uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B16_VF_LEN);
    uint16_t repeatTimes = CeilDivision(calCount, sregLower);
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
        preg = MicroAPI::UpdateMask<uint16_t>(sreg);
        MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(vreg1, scaleUb + i * sregLower);
        MicroAPI::ShiftLefts<uint16_t, int16_t>((MicroAPI::RegTensor<uint16_t>&)b16vreg1,
            (MicroAPI::RegTensor<uint16_t>&)vreg1, ANTIQUANT_BF16_MAN_LEN, preg);

        // 00000000 and 11111111 need special process
        SelectZeroNan(b16vreg1, bf16Zero, bf16Nan, e8m0Zero, e8m0Nan, selPreg, preg);

        MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>((MicroAPI::RegTensor<uint8_t>&)fp4vreg0,
            (__local_mem__ uint8_t*)srcUb + (i * sregLower / HALF_FACTOR));
        MicroAPI::Cast<bfloat16_t, SrcType, layoutZMrgZ>(b16vreg2, fp4vreg0, preg);
        if constexpr (SupportType<OutputDataType, half>()) {
            MicroAPI::Cast<half, bfloat16_t, MrgZRndRSatS>(halfvreg2, b16vreg2, preg);
            MicroAPI::Cast<half, bfloat16_t, MrgZRndRSatS>(halfvreg1, b16vreg1, preg);
            MicroAPI::Mul<half>(halfvreg2, halfvreg1, halfvreg2, preg);
            MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_NORM_B16>(dstUb + i * sregLower, halfvreg2, preg);
        } else {
            MicroAPI::Mul<bfloat16_t>(b16vreg2, b16vreg1, b16vreg2, preg);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_NORM_B16>(dstUb + i * sregLower, b16vreg2, preg);
        }
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AntiQuantProcessByNum(__local_mem__ OutputDataType* dst, __local_mem__ SrcType* src,
    __local_mem__ OutputDataType* scale, const uint32_t srcCalCount, const uint16_t newRepeatTime)
{
    uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B16_VF_LEN);
    MicroAPI::MaskReg preg1;
    MicroAPI::RegTensor<bfloat16_t> b16vreg2;
    MicroAPI::RegTensor<OutputDataType> scaleReg1;
    MicroAPI::RegTensor<OutputDataType> scaleReg2;
    MicroAPI::RegTensor<OutputDataType> tmpReg;
    MicroAPI::RegTensor<SrcType> fp4vreg0;
    MicroAPI::RegTensor<half> halfvreg2;
    uint32_t sreg1 = static_cast<uint32_t>(srcCalCount);
    for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTime); i++) {
        preg1 = MicroAPI::UpdateMask<uint16_t>(sreg1);
        MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_E2B_B16>(tmpReg, scale + i * DEFAULT_BLK_NUM);
        MicroAPI::Interleave<uint16_t>((MicroAPI::RegTensor<uint16_t>&)scaleReg1,
            (MicroAPI::RegTensor<uint16_t>&)scaleReg2, (MicroAPI::RegTensor<uint16_t>&)tmpReg,
            (MicroAPI::RegTensor<uint16_t>&)tmpReg);

        MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>((MicroAPI::RegTensor<uint8_t>&)fp4vreg0,
            (__local_mem__ uint8_t*)src + (2 * i) * sregLower / HALF_FACTOR); // once process half of sregLower num
        MicroAPI::Cast<bfloat16_t, SrcType, layoutZMrgZ>(b16vreg2, fp4vreg0, preg1);
        if constexpr (SupportType<OutputDataType, half>()) {
            MicroAPI::Cast<half, bfloat16_t, MrgZRndRSatS>(halfvreg2, b16vreg2, preg1);
            MicroAPI::Mul<half>(halfvreg2, scaleReg1, halfvreg2, preg1);
            MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_NORM_B16>(dst + (2 * i) * sregLower, halfvreg2, preg1);
        } else {
            MicroAPI::Mul<bfloat16_t>(b16vreg2, scaleReg1, b16vreg2, preg1);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_NORM_B16>(dst + (2 * i) * sregLower, b16vreg2,
                preg1);
        }

        preg1 = MicroAPI::UpdateMask<uint16_t>(sreg1);
        MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>((MicroAPI::RegTensor<uint8_t>&)fp4vreg0,
            (__local_mem__ uint8_t*)src + (2 * i + 1) * sregLower / HALF_FACTOR); // once process half of sregLower num
        MicroAPI::Cast<bfloat16_t, SrcType, layoutZMrgZ>(b16vreg2, fp4vreg0, preg1);

        if constexpr (SupportType<OutputDataType, half>()) {
            MicroAPI::Cast<half, bfloat16_t, MrgZRndRSatS>(halfvreg2, b16vreg2, preg1);
            MicroAPI::Mul<half>(halfvreg2, scaleReg2, halfvreg2, preg1);
            MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_NORM_B16>(dst + (2 * i + 1) * sregLower, halfvreg2,
                preg1);
        } else {
            MicroAPI::Mul<bfloat16_t>(b16vreg2, scaleReg2, b16vreg2, preg1);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_NORM_B16>(dst + (2 * i + 1) * sregLower, b16vreg2,
                preg1);
        }
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AscendAntiQuantNoTranspose(const LocalTensor<OutputDataType>& dst,
    const LocalTensor<SrcType>& src, const LocalTensor<fp8_e8m0_t>& scale, const uint32_t k, const uint32_t n)
{
    MicroAPI::RegTensor<uint16_t> bf16Zero;
    MicroAPI::RegTensor<uint16_t> bf16Nan;
    MicroAPI::RegTensor<uint16_t> e8m0Zero;
    MicroAPI::RegTensor<uint16_t> e8m0Nan;
    Duplicate(bf16Zero, (uint16_t)0x0040); // if e8m0 = 0b00000000, bf16 is 0x0040
    Duplicate(bf16Nan, (uint16_t)0x7fff);  // if e8m0 = 0b11111111, use 0x7fff as bf16 nan
    Duplicate(e8m0Zero, 0);
    Duplicate(e8m0Nan, (uint16_t)0x7f80); // if e8m0 = 0b11111111, after << 7 is 0x7f80
    uint16_t repeatTimes = CeilDivision(k, ANTIQUANT_FP4_PERGROUP_SIZE);
    for (uint16_t i = 0; i < repeatTimes; i++) {
        for (uint16_t j = 0; j < ANTIQUANT_FP4_PERGROUP_SIZE; j++) {
            __local_mem__ OutputDataType* dstUb =
                (__local_mem__ OutputDataType*)dst[i * ANTIQUANT_FP4_PERGROUP_SIZE * n + j * n].GetPhyAddr();
            __local_mem__ SrcType* srcUb =
                (__local_mem__ SrcType*)src[i * ANTIQUANT_FP4_PERGROUP_SIZE * n + j * n].GetPhyAddr();
            __local_mem__ fp8_e8m0_t* scaleUb = (__local_mem__ fp8_e8m0_t*)scale[i * n].GetPhyAddr();
            AntiQuantProcessByLine<SrcType, OutputDataType>(dstUb, srcUb, scaleUb, n, bf16Zero, bf16Nan, e8m0Zero,
                e8m0Nan);
        }
    }
}

__aicore__ inline constexpr uint32_t GetAscendAntiQuantTmpBufferLiveNode() {
    constexpr uint32_t tmpBufferLiveNode = 1;
    return tmpBufferLiveNode;
}

template<typename SrcType>
__aicore__ inline uint32_t GetAscendAntiQuantTmpBufferSize(const LocalTensor<uint8_t>& sharedTmpBuffer) {
    uint32_t sharedTmpBufferSize = sharedTmpBuffer.GetSize() / GetAscendAntiQuantTmpBufferLiveNode();
    return AlignUp(sharedTmpBufferSize, GetDataBlockSizeInBytes()) / sizeof(SrcType);
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AscendAntiQuantTranspose(const LocalTensor<OutputDataType>& dst,
    const LocalTensor<SrcType>& src, const LocalTensor<fp8_e8m0_t>& scale,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t k, const uint32_t n)
{
    __local_mem__ OutputDataType* dstUb = (__local_mem__ OutputDataType*)dst.GetPhyAddr();
    __local_mem__ SrcType* srcUb = (__local_mem__ SrcType*)src.GetPhyAddr();
    __local_mem__ fp8_e8m0_t* scaleUb = (__local_mem__ fp8_e8m0_t*)scale.GetPhyAddr();
    auto tmpbuffer = sharedTmpBuffer.ReinterpretCast<OutputDataType>();
    __local_mem__ OutputDataType* tmpbufferUb = (__local_mem__ OutputDataType*)tmpbuffer.GetPhyAddr();

    uint32_t srcCalCount = n * k;
    if (scale.GetSize() == 1) {
        uint32_t sharedTmpBufferSize = GetAscendAntiQuantTmpBufferSize<SrcType>(sharedTmpBuffer);
        uint32_t count = srcCalCount;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(srcCalCount,sharedTmpBufferSize));
        for (uint16_t i = 0; i < repeatTimes; i++) {
            uint32_t remainCount = count - sharedTmpBufferSize * i;
            uint32_t oneRepSize = remainCount < sharedTmpBufferSize ? remainCount : sharedTmpBufferSize;
            VF_CALL<CastScale<OutputDataType>>(tmpbufferUb, scaleUb + i * sharedTmpBufferSize, oneRepSize);
            uint16_t againRepeatTimes = CeilDivision(oneRepSize, ANTIQUANT_B16_VF_LEN);
            uint16_t newRepeatTime = (againRepeatTimes == 1) ? 1 : (againRepeatTimes / HALF_FACTOR); // if calcount <=128 need repeat once
            VF_CALL<AntiQuantProcessByNum<SrcType, OutputDataType>>(dstUb + i * sharedTmpBufferSize, srcUb + i * sharedTmpBufferSize,
                tmpbufferUb, oneRepSize, newRepeatTime);
        }
    } else {
        VF_CALL<CastScale<OutputDataType>>(tmpbufferUb, scaleUb, srcCalCount);
        uint16_t repeatTimes = CeilDivision(srcCalCount, ANTIQUANT_B16_VF_LEN);
        uint16_t newRepeatTime = (repeatTimes == 1) ? 1 : (repeatTimes / HALF_FACTOR); // if calcount <=128 need repeat once
        VF_CALL<AntiQuantProcessByNum<SrcType, OutputDataType>>(dstUb, srcUb, tmpbufferUb, srcCalCount,
            newRepeatTime);
    }
}

template <typename SrcType, typename OutputDataType, bool isTranspose>
__aicore__ inline void AscendAntiQuantImpl(const LocalTensor<OutputDataType>& dst,
    const LocalTensor<SrcType>& src, const LocalTensor<fp8_e8m0_t>& scale,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t k, const AntiQuantShapeInfo& shapeInfo = {})
{
    static_assert(SupportType<SrcType, fp4x2_e2m1_t, fp4x2_e1m2_t>(),
        "This AscendAntiQuant only support fp4 input dtype");
    static_assert(SupportType<OutputDataType, half, bfloat16_t>(),
        "This AscendAntiQuant only support half/bf16 output dtype");
    if constexpr (isTranspose) {
        ASCENDC_ASSERT((k != 0 && (k / HALF_FACTOR) % ONE_BLK_SIZE == 0),
                       { KERNEL_LOG(KERNEL_ERROR, "K should be larger than 0 && should be 32B aligned!"); });
        uint32_t n = (shapeInfo.scaleHeight == 0 ? scale.GetShapeInfo().shape[0] : shapeInfo.scaleHeight);
        AscendAntiQuantTranspose(dst, src, scale, sharedTmpBuffer, k, n);
    } else {
        uint32_t n1 = (shapeInfo.scaleWidth == 0 ? scale.GetShapeInfo().shape[1] : shapeInfo.scaleWidth);
        ASCENDC_ASSERT((n1 != 0 && (n1 / HALF_FACTOR) % ONE_BLK_SIZE == 0),
                       { KERNEL_LOG(KERNEL_ERROR, "k should be larger than 0 && should be 32B aligned!"); });
        VF_CALL<AscendAntiQuantNoTranspose<SrcType, OutputDataType>>(dst, src, scale, k, n1);
    }
}

/* **************************************************************************************************
 * perTensor for B8                                             *
 * ************************************************************************************************* */
template <typename SrcType, typename OutputDataType>
__aicore__ inline void PerTensorProcessForFp8(__local_mem__ OutputDataType* dst, __local_mem__ SrcType* src,
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

        if constexpr (SupportType<OutputDataType, bfloat16_t>()) {
            MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, ToFloat(offset), preg);
            MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, ToFloat(scale), preg);
        } else {
            MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, static_cast<float>(offset), preg);
            MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, static_cast<float>(scale), preg);
        }

        MicroAPI::Cast<OutputDataType, float, LayoutZMrgZRndRSatS>(outReg, f32vreg, preg);
        MicroAPI::DataCopy<OutputDataType, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * sregLower, outReg, preg);
    }
}
template <typename SrcType>
__simd_vf__ inline void PerTensorProcessForB8(__local_mem__ half* dst, __local_mem__ SrcType* src,
    const half offset, const half scale, const uint32_t srcCalCount)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SrcType> vreg;
    MicroAPI::RegTensor<bfloat16_t> b16vreg;
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

template <typename SrcType>
__simd_vf__ inline void PerTensorProcessForB8(__local_mem__ bfloat16_t* dst, __local_mem__ SrcType* src,
    const bfloat16_t offset, const bfloat16_t scale, const uint32_t srcCalCount)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SrcType> vreg;
    MicroAPI::RegTensor<bfloat16_t> b16vreg;
    MicroAPI::RegTensor<half> f16vreg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<half> vregTmp;

    uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B32_VF_LEN);
    uint32_t sreg = static_cast<uint32_t>(srcCalCount);
    uint16_t repeat = CeilDivision(srcCalCount, sregLower);

    uint32_t f16sreg = static_cast<uint32_t>(ANTIQUANT_B16_VF_LEN);
    MicroAPI::MaskReg f16preg = MicroAPI::UpdateMask<uint16_t>(f16sreg);

    for (uint16_t i = 0; i < static_cast<uint16_t>(repeat); ++i) {
        preg = MicroAPI::UpdateMask<uint32_t>(sreg);
        MicroAPI::DataCopy<SrcType, MicroAPI::LoadDist::DIST_UNPACK_B8>(vreg, src + i * sregLower);
        MicroAPI::Cast<half, SrcType, layoutZMrgZ>(f16vreg, vreg, f16preg); // hif8->f16 or int8->f16

        MicroAPI::Interleave(f16vreg, vregTmp, f16vreg, vregTmp);

        MicroAPI::Cast<float, half, layoutZMrgZ>(f32vreg, f16vreg, preg); // f16->f32
        MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, ToFloat(offset), preg);
        MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, ToFloat(scale), preg);
        MicroAPI::Cast<bfloat16_t, float, LayoutZMrgZRndRSatS>(b16vreg, f32vreg, preg);
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * sregLower, b16vreg, preg);
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AntiQuantPertensorImpl(const LocalTensor<OutputDataType>& dst,
    const LocalTensor<SrcType>& src, const OutputDataType offset, const OutputDataType scale,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t K, const AntiQuantShapeInfo& shapeInfo = {})
{
    static_assert(SupportType<SrcType, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t, int8_t>(),
        "This AscendAntiQuant only support fp8/hif8/int8 input dtype");
    static_assert(SupportType<OutputDataType, half, bfloat16_t>(),
        "This AscendAntiQuant only support f16/bf16 output dtype");
    __local_mem__ OutputDataType* dstUb = (__local_mem__ OutputDataType*)dst.GetPhyAddr();
    __local_mem__ SrcType* srcUb = (__local_mem__ SrcType*)src.GetPhyAddr();
    auto tmpbuffer = sharedTmpBuffer.ReinterpretCast<OutputDataType>();
    __local_mem__ OutputDataType* tmpbufferUb = (__local_mem__ OutputDataType*)tmpbuffer.GetPhyAddr();

    uint32_t srcCalCount = src.GetSize();
    if constexpr (SupportType<SrcType, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        VF_CALL<PerTensorProcessForFp8<SrcType, OutputDataType>>(dstUb, srcUb, offset, scale, srcCalCount);
    } else {
        // vfcall not support overload function
        PerTensorProcessForB8<SrcType>(dstUb, srcUb, offset, scale, srcCalCount);
    }
}

/* **************************************************************************************************
 * perChannel for B8                                             *
 * ************************************************************************************************* */
template <typename SrcType, typename OutputDataType>
__aicore__ inline void PerchannelNoTransposeForFp8(__local_mem__ OutputDataType* dst, __local_mem__ SrcType* src,
    __local_mem__ OutputDataType* offset, __local_mem__ OutputDataType* scale, const uint32_t K, const uint32_t N)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SrcType> vreg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<OutputDataType> outReg;
    MicroAPI::RegTensor<OutputDataType> scaleB16Vreg;
    MicroAPI::RegTensor<OutputDataType> offsetB16Vreg;
    MicroAPI::RegTensor<float> scaleB32Vreg;
    MicroAPI::RegTensor<float> offsetB32Vreg;

    uint32_t sregLower = ANTIQUANT_B32_VF_LEN;
    uint32_t sreg = N;
    uint16_t repeat = CeilDivision(N, sregLower);

    for (uint16_t i = 0; i < repeat; ++i) {
        preg = MicroAPI::UpdateMask<uint32_t>(sreg);
        // load offset and scale ,then cast to float to add &&mul
        MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_UNPACK_B16>(offsetB16Vreg, offset + i * sregLower);
        MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_UNPACK_B16>(scaleB16Vreg, scale + i * sregLower);
        MicroAPI::Cast<float, OutputDataType, layoutZMrgZ>(offsetB32Vreg, offsetB16Vreg, preg); // b16->fp32
        MicroAPI::Cast<float, OutputDataType, layoutZMrgZ>(scaleB32Vreg, scaleB16Vreg, preg);   // b16->fp32

        for (uint16_t j = 0; j < static_cast<uint16_t>(K); ++j) {
            MicroAPI::DataCopy<SrcType, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vreg, src + j * N + i * sregLower);
            MicroAPI::Cast<float, SrcType, layoutZMrgZ>(f32vreg, vreg, preg);

            MicroAPI::Add(f32vreg, f32vreg, offsetB32Vreg, preg);
            MicroAPI::Mul(f32vreg, f32vreg, scaleB32Vreg, preg);

            MicroAPI::Cast<OutputDataType, float, LayoutZMrgZRndRSatS>(outReg, f32vreg, preg);
            MicroAPI::DataCopy<OutputDataType, MicroAPI::StoreDist::DIST_PACK_B32>(dst + j * N + i * sregLower, outReg,
                preg);
        }
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void PerchannelNoTransposeForB8(__local_mem__ OutputDataType* dst, __local_mem__ SrcType* src,
    __local_mem__ OutputDataType* offset, __local_mem__ OutputDataType* scale, const uint32_t K, const uint32_t N)
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
            if constexpr (SupportType<OutputDataType, bfloat16_t>()) {
                MicroAPI::Cast<half, SrcType, layoutZMrgZ>(f16vreg, vreg, preg); // hif8->f16 or int8->f16
                MicroAPI::Cast<bfloat16_t, half, MrgZRndR>(b16vreg, f16vreg, preg);    // f16->bf16
            } else {
                MicroAPI::Cast<OutputDataType, SrcType, layoutZMrgZ>(b16vreg, vreg, preg);
            }

            MicroAPI::Add(b16vreg, b16vreg, offsetB16Vreg, preg);
            MicroAPI::Mul(b16vreg, b16vreg, scaleB16Vreg, preg);
            MicroAPI::DataCopy<OutputDataType, MicroAPI::StoreDist::DIST_NORM_B16>(dst + j * N + i * sregLower, b16vreg,
                preg);
        }
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AntiQuantPerchannelNoTranspose(const LocalTensor<OutputDataType>& dst,
    const LocalTensor<SrcType>& src, const LocalTensor<OutputDataType>& offset,
    const LocalTensor<OutputDataType>& scale, const uint32_t K, const uint32_t N)
{
    __local_mem__ OutputDataType* scaleUb = (__local_mem__ OutputDataType*)scale.GetPhyAddr();
    __local_mem__ OutputDataType* offsetUb = (__local_mem__ OutputDataType*)offset.GetPhyAddr();
    __local_mem__ OutputDataType* dstUb = (__local_mem__ OutputDataType*)dst.GetPhyAddr();
    __local_mem__ SrcType* srcUb = (__local_mem__ SrcType*)src.GetPhyAddr();

    if constexpr (SupportType<SrcType, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        VF_CALL<PerchannelNoTransposeForFp8<SrcType, OutputDataType>>(dstUb, srcUb, offsetUb, scaleUb, K, N);
    } else {
        VF_CALL<PerchannelNoTransposeForB8<SrcType, OutputDataType>>(dstUb, srcUb, offsetUb, scaleUb, K, N);
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void PerchannelUnlignedForFp8(__local_mem__ OutputDataType* dst, __local_mem__ SrcType* src,
    __local_mem__ OutputDataType* offset, __local_mem__ OutputDataType* scale, const uint32_t srcCalCount)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SrcType> vreg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<OutputDataType> outReg;
    MicroAPI::RegTensor<OutputDataType> scaleB16Vreg;
    MicroAPI::RegTensor<OutputDataType> offsetB16Vreg;
    MicroAPI::RegTensor<OutputDataType> scaleB16Vreg1;
    MicroAPI::RegTensor<OutputDataType> offsetB16Vreg1;
    MicroAPI::RegTensor<OutputDataType> scaleB16Vreg2;
    MicroAPI::RegTensor<OutputDataType> offsetB16Vreg2;
    MicroAPI::RegTensor<float> scaleB32Vreg;
    MicroAPI::RegTensor<float> offsetB32Vreg;

    uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B32_VF_LEN);
    uint32_t sreg = static_cast<uint32_t>(srcCalCount);
    uint16_t repeat = CeilDivision(srcCalCount, sregLower);

    uint32_t scaleLen = static_cast<uint32_t>(ANTIQUANT_B32_VF_LEN);
    MicroAPI::MaskReg b16Preg = MicroAPI::UpdateMask<uint32_t>(scaleLen);
    MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_BLK>(offsetB16Vreg, offset);
    MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_BLK>(scaleB16Vreg, scale);
    MicroAPI::Interleave<half>((MicroAPI::RegTensor<half>&)offsetB16Vreg1, (MicroAPI::RegTensor<half>&)offsetB16Vreg2,
        (MicroAPI::RegTensor<half>&)offsetB16Vreg, (MicroAPI::RegTensor<half>&)offsetB16Vreg);
    MicroAPI::Interleave<half>((MicroAPI::RegTensor<half>&)scaleB16Vreg1, (MicroAPI::RegTensor<half>&)scaleB16Vreg2,
        (MicroAPI::RegTensor<half>&)scaleB16Vreg, (MicroAPI::RegTensor<half>&)scaleB16Vreg);
    MicroAPI::Cast<float, OutputDataType, layoutZMrgZ>(offsetB32Vreg, offsetB16Vreg1, b16Preg);
    MicroAPI::Cast<float, OutputDataType, layoutZMrgZ>(scaleB32Vreg, scaleB16Vreg1, b16Preg);

    for (uint16_t i = 0; i < static_cast<uint16_t>(repeat); ++i) {
        preg = MicroAPI::UpdateMask<uint32_t>(sreg);
        MicroAPI::DataCopy<SrcType, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vreg, src + i * sregLower);
        MicroAPI::Cast<float, SrcType, layoutZMrgZ>(f32vreg, vreg, preg);

        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, offsetB32Vreg, preg);
        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, scaleB32Vreg, preg);

        MicroAPI::Cast<OutputDataType, float, LayoutZMrgZRndRSatS>(outReg, f32vreg, preg);
        MicroAPI::DataCopy<OutputDataType, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * sregLower, outReg, preg);
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void PerchannelUnlignedForB8(__local_mem__ OutputDataType* dst, __local_mem__ SrcType* src,
    __local_mem__ OutputDataType* offset, __local_mem__ OutputDataType* scale, const uint32_t srcCalCount)
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

        if constexpr (SupportType<OutputDataType, bfloat16_t>()) {
            MicroAPI::Cast<half, SrcType, layoutZMrgZ>(f16vreg, vreg, preg); // hif8->f16 or int8->f16
            MicroAPI::Cast<bfloat16_t, half, MrgZRndA>(b16vreg, f16vreg, preg);
        } else {
            MicroAPI::Cast<OutputDataType, SrcType, layoutZMrgZ>(b16vreg, vreg, preg);
        }
        MicroAPI::Add<OutputDataType, MicroAPI::MaskMergeMode::ZEROING>(b16vreg, b16vreg, offsetB16Vreg, preg);
        MicroAPI::Mul<OutputDataType, MicroAPI::MaskMergeMode::ZEROING>(b16vreg, b16vreg, scaleB16Vreg, preg);
        MicroAPI::DataCopy<OutputDataType, MicroAPI::StoreDist::DIST_NORM_B16>(dst + i * sregLower, b16vreg, preg);
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AntiQuantUnlignedProcess(const LocalTensor<OutputDataType>& dst,
    const LocalTensor<SrcType>& src, const LocalTensor<OutputDataType>& offset,
    const LocalTensor<OutputDataType>& scale, const uint32_t K, const uint32_t N)
{
    __local_mem__ OutputDataType* scaleUb = (__local_mem__ OutputDataType*)scale.GetPhyAddr();
    __local_mem__ OutputDataType* offsetUb = (__local_mem__ OutputDataType*)offset.GetPhyAddr();
    __local_mem__ OutputDataType* dstUb = (__local_mem__ OutputDataType*)dst.GetPhyAddr();
    __local_mem__ SrcType* srcUb = (__local_mem__ SrcType*)src.GetPhyAddr();

    if constexpr (SupportType<SrcType, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        VF_CALL<PerchannelUnlignedForFp8<SrcType, OutputDataType>>(dstUb, srcUb, offsetUb, scaleUb, N * K);
    } else { // now only support hifloat8 and int8
        VF_CALL<PerchannelUnlignedForB8<SrcType, OutputDataType>>(dstUb, srcUb, offsetUb, scaleUb, N * K);
    }
}

template <typename SrcType>
__simd_vf__ inline void PerchannelTransposeForB8(__local_mem__ half* dst, __local_mem__ SrcType* src,
    __local_mem__ half* offset, __local_mem__ half* scale, const uint32_t K, const uint32_t N)
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
            MicroAPI::DataCopy<SrcType, MicroAPI::LoadDist::DIST_UNPACK_B8>(vreg,
                src + i * K + j * sregLower);
            MicroAPI::Cast<half, SrcType, layoutZMrgZ>(f16vreg, vreg, preg); // hif8->f16 or int8->f16

            MicroAPI::Add<half, MicroAPI::MaskMergeMode::ZEROING>(f16vreg, f16vreg, offsetB16Vreg, preg);
            MicroAPI::Mul<half, MicroAPI::MaskMergeMode::ZEROING>(f16vreg, f16vreg, scaleB16Vreg, preg);

            MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_NORM_B16>(dst + i * K + j * sregLower, f16vreg,
                preg);
        }
    }
}

template <typename SrcType>
__simd_vf__ inline void PerchannelTransposeForB8(__local_mem__ bfloat16_t* dst, __local_mem__ SrcType* src,
    __local_mem__ bfloat16_t* offset, __local_mem__ bfloat16_t* scale, const uint32_t K, const uint32_t N)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SrcType> vreg;
    uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B32_VF_LEN);
    uint16_t repeat = CeilDivision(K, sregLower);

    uint32_t f16sreg = static_cast<uint32_t>(ANTIQUANT_B16_VF_LEN);
    MicroAPI::MaskReg f16preg = MicroAPI::UpdateMask<uint16_t>(f16sreg);

    MicroAPI::RegTensor<bfloat16_t> b16vreg;
    MicroAPI::RegTensor<bfloat16_t> scaleB16Vreg;
    MicroAPI::RegTensor<bfloat16_t> offsetB16Vreg;
    MicroAPI::RegTensor<float> scaleB32Vreg;
    MicroAPI::RegTensor<float> offsetB32Vreg;
    MicroAPI::RegTensor<half> f16vreg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<half> vregTmp;

    for (uint16_t i = 0; i < static_cast<uint16_t>(N); i++) {
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_BRC_B16>(scaleB16Vreg, scale + i);
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_BRC_B16>(offsetB16Vreg, offset + i);

        MicroAPI::Cast<float, bfloat16_t, layoutZMrgZ>(offsetB32Vreg, offsetB16Vreg, f16preg);
        MicroAPI::Cast<float, bfloat16_t, layoutZMrgZ>(scaleB32Vreg, scaleB16Vreg, f16preg);

        uint32_t sreg = static_cast<uint32_t>(K);
        for (uint16_t j = 0; j < repeat; ++j) { // process single line
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy<SrcType, MicroAPI::LoadDist::DIST_UNPACK_B8>(vreg,
                src + i * K + j * sregLower);
            MicroAPI::Cast<half, SrcType, layoutZMrgZ>(f16vreg, vreg,
                f16preg); // hif8->f16 or int8->f16
            MicroAPI::Interleave(f16vreg, vregTmp, f16vreg, vregTmp);
            MicroAPI::Cast<float, half, layoutZMrgZ>(f32vreg, f16vreg, preg); // f16->f32

            MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, offsetB32Vreg, preg);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, scaleB32Vreg, preg);
            MicroAPI::Cast<bfloat16_t, float, LayoutZMrgZRndRSatS>(b16vreg, f32vreg, preg);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * K + j * sregLower, b16vreg,
                preg);
        }
    }
}

template <typename SrcType, typename OutputDataType>
__simd_vf__ inline void PerchannelTransposeForFp8(__local_mem__ OutputDataType* dst, __local_mem__ SrcType* src,
    __local_mem__ OutputDataType* offset, __local_mem__ OutputDataType* scale, const uint32_t K, const uint32_t N)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SrcType> vreg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<OutputDataType> outReg;
    MicroAPI::RegTensor<OutputDataType> scaleB16Vreg;
    MicroAPI::RegTensor<OutputDataType> offsetB16Vreg;
    MicroAPI::RegTensor<float> scaleB32Vreg;
    MicroAPI::RegTensor<float> offsetB32Vreg;
    uint32_t f16sreg = static_cast<uint32_t>(ANTIQUANT_B16_VF_LEN);
    MicroAPI::MaskReg f16preg = MicroAPI::UpdateMask<uint16_t>(f16sreg);
    uint32_t sregLower = static_cast<uint32_t>(ANTIQUANT_B32_VF_LEN);
    uint16_t repeat = CeilDivision(K, sregLower);

    for (uint16_t i = 0; i < static_cast<uint16_t>(N); i++) {
        MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_BRC_B16>(scaleB16Vreg, scale + i);
        MicroAPI::DataCopy<OutputDataType, MicroAPI::LoadDist::DIST_BRC_B16>(offsetB16Vreg, offset + i);

        MicroAPI::Cast<float, OutputDataType, layoutZMrgZ>(offsetB32Vreg, offsetB16Vreg, f16preg);
        MicroAPI::Cast<float, OutputDataType, layoutZMrgZ>(scaleB32Vreg, scaleB16Vreg, f16preg);

        uint32_t sreg = static_cast<uint32_t>(K);
        for (uint16_t j = 0; j < static_cast<uint16_t>(repeat); ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy<SrcType, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vreg,
                src + i * K + j * sregLower);
            MicroAPI::Cast<float, SrcType, layoutZMrgZ>(f32vreg, vreg, preg);

            MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, offsetB32Vreg, preg);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32vreg, f32vreg, scaleB32Vreg, preg);

            MicroAPI::Cast<OutputDataType, float, LayoutZMrgZRndRSatS>(outReg, f32vreg, preg);
            MicroAPI::DataCopy<OutputDataType, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * K + j * sregLower,
                outReg, preg);
        }
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void AntiQuantPerchannelTranspose(const LocalTensor<OutputDataType>& dst,
    const LocalTensor<SrcType>& src, const LocalTensor<OutputDataType>& offset,
    const LocalTensor<OutputDataType>& scale, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t K,
    const uint32_t N)
{
    __local_mem__ OutputDataType* dstUb = (__local_mem__ OutputDataType*)dst.GetPhyAddr();
    __local_mem__ SrcType* srcUb = (__local_mem__ SrcType*)src.GetPhyAddr();
    __local_mem__ OutputDataType* scaleUb = (__local_mem__ OutputDataType*)scale.GetPhyAddr();
    __local_mem__ OutputDataType* offsetUb = (__local_mem__ OutputDataType*)offset.GetPhyAddr();

    if constexpr (SupportType<SrcType, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        PerchannelTransposeForFp8<SrcType, OutputDataType>(dstUb, srcUb, offsetUb, scaleUb, K, N);
    } else { // now only support hifloat8 and int8
        PerchannelTransposeForB8<SrcType>(dstUb, srcUb, offsetUb, scaleUb, K, N);
    }
}

template <typename SrcType, typename OutputDataType, bool isTranspose>
__aicore__ inline void AntiQuantPerchannelImpl(const LocalTensor<OutputDataType>& dst,
    const LocalTensor<SrcType>& src, const LocalTensor<OutputDataType>& offset,
    const LocalTensor<OutputDataType>& scale, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t k,
    const AntiQuantShapeInfo& shapeInfo = {})
{
    static_assert(SupportType<SrcType, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t, int8_t>(),
        "This AscendAntiQuant only support fp8/hif8/int8 input dtype");
    static_assert(SupportType<OutputDataType, half, bfloat16_t>(),
        "This AscendAntiQuant only support f16/bf16 output dtype");

    if constexpr (isTranspose) { // src [n,k] offset [n,1]
        uint32_t n = (shapeInfo.offsetWidth == 0 ? offset.GetShapeInfo().shape[0] : shapeInfo.offsetWidth);
        AntiQuantPerchannelTranspose(dst, src, offset, scale, sharedTmpBuffer, k, n);
    } else { // src [k,n] offset [1,n]
        uint32_t n = (shapeInfo.offsetWidth == 0 ? offset.GetShapeInfo().shape[1] : shapeInfo.offsetWidth);
        if (n < 32) { // b8 input single line is not 32B aligned such as input n == 16
            ASCENDC_ASSERT((k % 2 == 0), { KERNEL_LOG(KERNEL_ERROR, "input calculate size must be 32B aligned!"); });
            AntiQuantUnlignedProcess<SrcType, OutputDataType>(dst, src, offset, scale, k, n);
        } else {
            AntiQuantPerchannelNoTranspose<SrcType, OutputDataType>(dst, src, offset, scale, k, n);
        }
    }
}

template <typename SrcType, bool withOffset = true>
__aicore__ inline void AntiQuantInnerLoop(const LocalTensor<bfloat16_t> &dst, const LocalTensor<SrcType> &src,
    const LocalTensor<bfloat16_t> &offset, const LocalTensor<bfloat16_t> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const UnaryRepeatParams &unaryParamsCastSrc,
    const UnaryRepeatParams &unaryParamsToFP32, const UnaryRepeatParams &unaryParamsFP32ToDst,
    const BinaryRepeatParams &binaryParams, const uint32_t calCount)
{
    uint32_t srcFp16Pos = calCount * sizeof(bfloat16_t);
    uint32_t offsetFp32Pos = calCount * sizeof(float);
    auto fp16TmpBuffer = sharedTmpBuffer[srcFp16Pos].ReinterpretCast<half>();
    auto offsetBuffer = sharedTmpBuffer[offsetFp32Pos].ReinterpretCast<float>();
    auto resultBuffer = sharedTmpBuffer.ReinterpretCast<float>();

    UnaryRepeatParams src2f16unaryParams;
    if constexpr (IsSameType<SrcType, int8_t>::value) {
        src2f16unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    } else {
        src2f16unaryParams.srcRepStride = ONE_FOURTH_DEFAULT_REPEAT_STRIDE;
    }
    src2f16unaryParams.srcRepStride = ONE_FOURTH_DEFAULT_REPEAT_STRIDE;
    UnaryRepeatParams unaryParams;
    unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    UnaryRepeatParams f322f16Params;
    f322f16Params.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;

    SetVectorMask<float, MaskMode::COUNTER>(0, calCount);
    Cast<half, SrcType>(fp16TmpBuffer, src, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();
    Cast<float, half>(resultBuffer, fp16TmpBuffer, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();
    if constexpr (withOffset) {
        Cast<float, bfloat16_t>(offsetBuffer, offset, RoundMode::CAST_NONE, calCount);
        PipeBarrier<PIPE_V>();
        Add<float>(resultBuffer, resultBuffer, offsetBuffer, calCount);
        PipeBarrier<PIPE_V>();
    }
    Cast<float, bfloat16_t>(offsetBuffer, scale, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();
    Mul<float>(resultBuffer, resultBuffer, offsetBuffer, calCount);
    PipeBarrier<PIPE_V>();
    Cast<bfloat16_t, float>(dst, resultBuffer, RoundMode::CAST_RINT, calCount);
    PipeBarrier<PIPE_V>();
}

template <typename SrcType, bool withOffset = true>
__aicore__ inline void AntiQuantInnerLoop(const LocalTensor<bfloat16_t> &dst, const LocalTensor<SrcType> &src,
    const bfloat16_t offset, const bfloat16_t scale, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const UnaryRepeatParams &unaryParamsCastSrc, const UnaryRepeatParams &unaryParamsToFP32,
    const UnaryRepeatParams &unaryParamsFP32ToDst, const UnaryRepeatParams &unaryParamsScalar, const uint32_t calCount)
{
    uint32_t srcFp16Pos = calCount * sizeof(bfloat16_t);
    auto fp16TmpBuffer = sharedTmpBuffer[srcFp16Pos].ReinterpretCast<half>();
    auto resultBuffer = sharedTmpBuffer.ReinterpretCast<float>();

    UnaryRepeatParams src2f16unaryParams;
    if constexpr (IsSameType<SrcType, int8_t>::value) {
        src2f16unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    } else {
        src2f16unaryParams.srcRepStride = ONE_FOURTH_DEFAULT_REPEAT_STRIDE;
    }
    UnaryRepeatParams unaryParams;
    unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    UnaryRepeatParams f322f16Params;
    f322f16Params.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;

    SetVectorMask<float, MaskMode::COUNTER>(0, calCount);
    Cast<half, SrcType>(fp16TmpBuffer, src, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();
    Cast<float, half>(resultBuffer, fp16TmpBuffer, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();
    if constexpr (withOffset) {
        Adds<float>(resultBuffer, resultBuffer, ToFloat(offset), calCount);
        PipeBarrier<PIPE_V>();
    }
    Muls<float>(resultBuffer, resultBuffer, ToFloat(scale), calCount);
    PipeBarrier<PIPE_V>();
    Cast<bfloat16_t, float>(dst, resultBuffer, RoundMode::CAST_RINT, calCount);
    PipeBarrier<PIPE_V>();
}

template <typename SrcType>
__aicore__ inline void AscendAntiQuantNoTransposePerformance(const LocalTensor<bfloat16_t> &dst,
    const LocalTensor<SrcType> &src, const LocalTensor<bfloat16_t> &offset, const LocalTensor<bfloat16_t> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K, const uint32_t N)
{
    uint32_t posOffsetScale = N * sizeof(float) * ANTIQUANT_TWO;
    uint32_t posCast = posOffsetScale + ANTIQUANT_SINGLE_N_SIZE_BF16 * K * sizeof(half);
    auto fp16TmpBuffer = sharedTmpBuffer[posCast].ReinterpretCast<half>();
    auto resultBuffer = sharedTmpBuffer[posOffsetScale].ReinterpretCast<float>();

    UnaryRepeatParams s42f16unaryParams;
    s42f16unaryParams.srcRepStride = N / ANTIQUANT_TWO / ONE_BLK_SIZE;
    s42f16unaryParams.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    UnaryRepeatParams s82f16unaryParams;
    s82f16unaryParams.srcRepStride = N * sizeof(int8_t) / ONE_BLK_SIZE;
    s82f16unaryParams.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    UnaryRepeatParams f162f32unaryParams;
    f162f32unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    BinaryRepeatParams binaryParams;
    binaryParams.src1RepStride = 0;
    UnaryRepeatParams f322f16Params;
    f322f16Params.dstRepStride = N * sizeof(half) / ONE_BLK_SIZE;

    for (uint32_t i = 0; i < N / ANTIQUANT_SINGLE_N_SIZE_BF16; i++) {
        if constexpr (IsSameType<SrcType, int4b_t>::value) {
            // 1.cast 64K to fp16, use norm mode
            Cast<half, int4b_t>(fp16TmpBuffer, src[ANTIQUANT_SINGLE_N_SIZE_BF16 * i], RoundMode::CAST_NONE,
                ANTIQUANT_SINGLE_N_SIZE_BF16, K, s42f16unaryParams);
        } else {
            // 1.cast 64K to fp16, use norm mode
            Cast<half, int8_t>(fp16TmpBuffer, src[ANTIQUANT_SINGLE_N_SIZE_BF16 * i], RoundMode::CAST_NONE,
                ANTIQUANT_SINGLE_N_SIZE_BF16, K, s82f16unaryParams);
        }

        Cast<float, half>(resultBuffer, fp16TmpBuffer, RoundMode::CAST_NONE, ANTIQUANT_SINGLE_N_SIZE_BF16, K,
            f162f32unaryParams);
        // 2.add offset
        auto offsetBuffer = sharedTmpBuffer[ANTIQUANT_SINGLE_N_SIZE_BF16 * i * sizeof(float)].ReinterpretCast<float>();

        Add<float>(resultBuffer, resultBuffer, offsetBuffer, ANTIQUANT_SINGLE_N_SIZE_BF16, K, binaryParams);

        // 3.mul scale
        auto scaleBuffer = sharedTmpBuffer[N * sizeof(float) + ANTIQUANT_SINGLE_N_SIZE_BF16 * i * sizeof(float)]
            .ReinterpretCast<float>();
        Mul<float>(resultBuffer, resultBuffer, scaleBuffer, ANTIQUANT_SINGLE_N_SIZE_BF16, K, binaryParams);
        // 4.cast back to bf16
        Cast<bfloat16_t, float>(dst[ANTIQUANT_SINGLE_N_SIZE_BF16 * i], resultBuffer, RoundMode::CAST_RINT,
            ANTIQUANT_SINGLE_N_SIZE_BF16, K, f322f16Params);
    }
}

template <typename SrcType>
__aicore__ inline void AscendAntiQuantNoTransposePerformanceTail(const LocalTensor<bfloat16_t> &dst,
    const LocalTensor<SrcType> &src, const LocalTensor<bfloat16_t> &offset, const LocalTensor<bfloat16_t> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K, const uint32_t N, const uint32_t mask)
{
    uint32_t index = N / ANTIQUANT_SINGLE_N_SIZE_BF16 * ANTIQUANT_SINGLE_N_SIZE_BF16;
    uint32_t posOffset = N * sizeof(float);
    uint32_t posOffsetScale = posOffset * ANTIQUANT_TWO;
    uint32_t posCast = posOffsetScale + ANTIQUANT_SINGLE_N_SIZE_BF16 * K * sizeof(half);
    auto fp16TmpBuffer = sharedTmpBuffer[posCast].ReinterpretCast<half>();
    auto resultBuffer = sharedTmpBuffer[posOffsetScale].ReinterpretCast<float>();
    auto offsetBuffer = sharedTmpBuffer[index * sizeof(float)].ReinterpretCast<float>();
    auto scaleBuffer = sharedTmpBuffer[posOffset + index * sizeof(float)].ReinterpretCast<float>();

    UnaryRepeatParams s42f16unaryParams;
    s42f16unaryParams.srcRepStride = N / ANTIQUANT_TWO / ONE_BLK_SIZE;
    UnaryRepeatParams s82f16unaryParams;
    s82f16unaryParams.srcRepStride = N * sizeof(int8_t) / ONE_BLK_SIZE;
    s82f16unaryParams.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    UnaryRepeatParams f162f32unaryParams;
    f162f32unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    BinaryRepeatParams binaryParams;
    binaryParams.src1RepStride = 0;
    UnaryRepeatParams f322f16Params;
    f322f16Params.dstRepStride = N * sizeof(bfloat16_t) / ONE_BLK_SIZE;

    if constexpr (IsSameType<SrcType, int4b_t>::value) {
        Cast<half, int4b_t>(fp16TmpBuffer, src, RoundMode::CAST_NONE, mask, K, s42f16unaryParams);
    } else {
        Cast<half, int8_t>(fp16TmpBuffer, src, RoundMode::CAST_NONE, mask, K, s82f16unaryParams);
    }

    // cast 64K to fp32, use count mode
    Cast<float, half>(resultBuffer, fp16TmpBuffer, RoundMode::CAST_NONE, mask, K, f162f32unaryParams);
    // 2.add offset
    Add<float>(resultBuffer, resultBuffer, offsetBuffer, mask, K, binaryParams);

    // 3.mul scale
    Mul<float>(resultBuffer, resultBuffer, scaleBuffer, mask, K, binaryParams);

    // 4.cast back to bf16
    Cast<bfloat16_t, float>(dst, resultBuffer, RoundMode::CAST_RINT, mask, K, f322f16Params);
}

template <typename SrcType>
__aicore__ inline void PreCast(const LocalTensor<bfloat16_t> &dst, const LocalTensor<SrcType> &src,
    const LocalTensor<bfloat16_t> &offset, const LocalTensor<bfloat16_t> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K)
{
    uint32_t posOffset = offset.GetSize() * sizeof(float);
    uint32_t repeatEle = ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t);
    uint32_t repeatTimes =
        offset.GetSize() % repeatEle == 0 ? offset.GetSize() / repeatEle : offset.GetSize() / repeatEle + 1;
    auto offsetBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    auto scaleBuffer = sharedTmpBuffer[posOffset].ReinterpretCast<float>();

    UnaryRepeatParams unaryParams;
    unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;

    Cast<float, bfloat16_t>(offsetBuffer, offset, RoundMode::CAST_NONE, offset.GetSize());
    PipeBarrier<PIPE_V>();
    Cast<float, bfloat16_t>(scaleBuffer, scale, RoundMode::CAST_NONE, offset.GetSize());
    PipeBarrier<PIPE_V>();
}

template <typename OutputDataType>
__aicore__ inline bool AntiQuantCheckPerformanceMode(const LocalTensor<OutputDataType> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K)
{
    if constexpr (IsSameType<OutputDataType, bfloat16_t>::value) {
        uint32_t maxTmpBufferSize =
            scale.GetSize() * ANTIQUANT_TWO * sizeof(float) + ANTIQUANT_SINGLE_N_SIZE_BF16 * K * sizeof(float);
        return sharedTmpBuffer.GetSize() >= maxTmpBufferSize;
    }
    return true;
}

// scale * (src + offset)   src: N * K, scale: N, offset: N  NOffset: offset used for tmpTensorOffset, tmpTensorScale
// For now, calCount must equal to N * K then can use brcb
template <typename SrcType, typename OutputDataType, bool isOffset>
__aicore__ inline void CalculationMax(const LocalTensor<SrcType> &src, const LocalTensor<OutputDataType> &dst,
    AntiquantParams<float> &params, const uint32_t calCount, const uint32_t N, const uint32_t K, const uint32_t NOffset)
{
    // store FP16 result in second half of FP32 tmpTensor to avoid input FP16 being replaced
    uint32_t srcFp16Pos = calCount / ANTIQUANT_TWO; // therefore start from (calCount / 2)th FP32 tmpTensor
    auto fp16TmpBuffer = params.tempTensorInput[srcFp16Pos].ReinterpretCast<half>();

    UnaryRepeatParams unaryParams;
    unaryParams.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    UnaryRepeatParams f322f16Params;
    f322f16Params.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    uint32_t count = K / ANTIQUANT_SINGLE_N_SIZE; // times of for loop   K = n * 64
    // src1BlkStride = 0: need same line for add and mul
    // src1RepStride = 1: 1 block for 64 num calculation
    // dst, src0RepStride = count * 8: one repeat calculate 64 num, need to jump n * 8 block
    BinaryRepeatParams binaryParams(1, 1, 0, count * DEFAULT_REPEAT_STRIDE, count * DEFAULT_REPEAT_STRIDE, 1);

    SetVectorMask<half, MaskMode::COUNTER>(0, calCount);
    // INT8 -> FP16
    Cast<half, int8_t>(fp16TmpBuffer, src, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();
    // FP16 -> FP32
    Cast<float, half>(params.tempTensorInput, fp16TmpBuffer, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();

    SetVectorMask<float, MaskMode::COUNTER>(0, ANTIQUANT_SINGLE_N_SIZE * N); // brcb  src1 has N line, 1 line has 64 num
    for (uint32_t i = 0; i < count; i++) {
        // scale * (src + offset)
        uint32_t curOffset = i * ANTIQUANT_SINGLE_N_SIZE;
        // calculate the first group (0 ~ 64) in first loop, second group (64 ~ 128) in second loop
        if constexpr (isOffset) {
            Add<float>(params.tempTensorInput[curOffset], params.tempTensorInput[curOffset],
                params.tempTensorOffset[NOffset], ANTIQUANT_SINGLE_N_SIZE * N);
            PipeBarrier<PIPE_V>();
        }
        Mul<float>(params.tempTensorInput[curOffset], params.tempTensorInput[curOffset],
            params.tempTensorScale[NOffset], ANTIQUANT_SINGLE_N_SIZE * N);
        PipeBarrier<PIPE_V>();
    }

    // FP32 -> BF16
    SetVectorMask<float, MaskMode::COUNTER>(0, calCount);
    Cast<bfloat16_t, float>(dst, params.tempTensorInput, RoundMode::CAST_RINT, calCount);
    PipeBarrier<PIPE_V>();
}

// Brcb version
// allocate tmp buffer
template <typename OutputDataType>
__aicore__ inline void GetAntiquantTensorInfo(const LocalTensor<OutputDataType> &scale,
    const LocalTensor<float> &stackBuffer, AntiquantParams<float> &params)
{
    uint32_t N = scale.GetSize();                                  // scale and offset are shape [N]
    params.tempTensorOffset = stackBuffer[0];                      // store 8 * N * FP32    N -> brcb -> 8 * N
    params.tempTensorScale = stackBuffer[ANTIQUANT_BRCB_BASE * N]; // store 8 * N * FP32    N -> brcb -> 8 * N
    params.tempTensorInput = stackBuffer[ANTIQUANT_BRCB_BASE * ANTIQUANT_TWO * N]; // need [N * 64 * FP32, N * K * FP32]
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
        Cast<float, OutputDataType>(params.tempTensorOffset[ANTIQUANT_BRCB_BASE * N - nLength], offset,
            RoundMode::CAST_NONE, nLength);
    }
    Cast<float, OutputDataType>(params.tempTensorScale[ANTIQUANT_BRCB_BASE * N - nLength], scale, RoundMode::CAST_NONE,
        nLength);
    PipeBarrier<PIPE_V>();

    constexpr uint16_t brcbDstBlkStride = 1;                   // 1 num -> 8 num(1 block)
    constexpr uint16_t brcbDstRepStride = ANTIQUANT_BRCB_BASE; // 1 brcb: 8 num -> 64 num
    const uint8_t repeatTimes = nLength / ANTIQUANT_BRCB_BASE; // 1 brcb cmd needs 8 input num
    BrcbRepeatParams brcbParams(brcbDstBlkStride, brcbDstRepStride);

    SetMaskNorm();
    ResetMask();
    // brcb: 1 FP32 A -> 1 block contains 8 FP32 A, after 1 block, do the same to the next FP32 B
    if constexpr (withOffset) {
        Brcb(params.tempTensorOffset, params.tempTensorOffset[ANTIQUANT_BRCB_BASE * N - nLength], repeatTimes,
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
__aicore__ inline void CalculationMin(const LocalTensor<SrcType> &src, const LocalTensor<OutputDataType> &dst,
    AntiquantParams<float> &params, const uint32_t calCount, const uint32_t n, const uint32_t srcN, const uint32_t k)
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
    Cast<float, half, false>(params.tempTensorInput, fp16TmpBuffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER, n,
        unaryParamsFp16Fp32);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    BinaryRepeatParams binaryParams;
    binaryParams.src1BlkStride = 0; // same line for add and mul
    binaryParams.src1RepStride = 1; // one line for 64 num calculation

    SetVectorMask<float, MaskMode::COUNTER>(0, ANTIQUANT_SINGLE_N_SIZE * n);
    // scale * (src + offset)
    if constexpr (withOffset) {
        Add<float>(params.tempTensorInput, params.tempTensorInput, params.tempTensorOffset,
            ANTIQUANT_SINGLE_N_SIZE * n);
        PipeBarrier<PIPE_V>();
    }
    Mul<float>(params.tempTensorInput, params.tempTensorInput, params.tempTensorScale, ANTIQUANT_SINGLE_N_SIZE * n);
    PipeBarrier<PIPE_V>();

    // FP32 -> BF16
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(0, FULL_MASK);
    UnaryRepeatParams f322f16Params;
    f322f16Params.dstRepStride = ANTIQUANT_SINGLE_N_SIZE * n1 / (ONE_BLK_SIZE / sizeof(half));
    Cast<OutputDataType, float, false>(dst, params.tempTensorInput, RoundMode::CAST_RINT, MASK_PLACEHOLDER, srcN,
        f322f16Params);
    PipeBarrier<PIPE_V>();
}

// Method2: min: N * 64
template <typename SrcType, typename OutputDataType>
__aicore__ inline void CalculateByBrcbMin(const LocalTensor<OutputDataType> &dst, const LocalTensor<SrcType> &src,
    const LocalTensor<OutputDataType> &offset, const LocalTensor<OutputDataType> &scale,
    const LocalTensor<float> &stackBuffer, const uint32_t calCount, const uint32_t n, const uint32_t k)
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
        CalculationMin<SrcType, OutputDataType, true>(src[curNKOffset], dst[curNKOffset], antiquantParams,
            ANTIQUANT_SINGLE_N_SIZE * n, n, srcN, k);
    }
}

template <typename SrcType, typename OutputDataType>
__aicore__ inline void CalculateByBrcbMin(const LocalTensor<OutputDataType> &dst, const LocalTensor<SrcType> &src,
    const LocalTensor<OutputDataType> &scale, const LocalTensor<float> &stackBuffer, const uint32_t calCount,
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
        CalculationMin<SrcType, OutputDataType, false>(src[curNKOffset], dst[curNKOffset], antiquantParams,
            ANTIQUANT_SINGLE_N_SIZE * n, n, srcN, k);
    }
}

template <typename OutputDataType>
__aicore__ inline void CalculateByBrcbMin(const LocalTensor<OutputDataType> &dst, const LocalTensor<int4b_t> &src,
    const LocalTensor<OutputDataType> &scale, const LocalTensor<float> &stackBuffer, const uint32_t calCount,
    const uint32_t n, const uint32_t k)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported type: int4b_t for AntiQuant"); });
}

template <typename OutputDataType>
__aicore__ inline void CalculateByBrcbMin(const LocalTensor<OutputDataType> &dst, const LocalTensor<int4b_t> &src,
    const LocalTensor<OutputDataType> &offset, const LocalTensor<OutputDataType> &scale,
    const LocalTensor<float> &stackBuffer, const uint32_t calCount, const uint32_t n, const uint32_t k)
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
__aicore__ inline void AscendAntiQuantTranspose(const LocalTensor<OutputDataType> &dst,
    const LocalTensor<SrcType> &src, const LocalTensor<OutputDataType> &offset,
    const LocalTensor<OutputDataType> &scale, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K,
    const AntiQuantShapeInfo &shapeInfo = {})
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
__aicore__ inline void AscendAntiQuantTranspose(const LocalTensor<OutputDataType> &dst,
    const LocalTensor<SrcType> &src, const LocalTensor<OutputDataType> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K, const AntiQuantShapeInfo &shapeInfo = {})
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
__simd_callee__ inline void LoadPerTokenScaleAndOffset(__local_mem__ scaleT* scaleUb,
                                                  __local_mem__ scaleT* offsetUb,
                                                  MicroAPI::RegTensor<scaleT>& scaleVreg,
                                                  MicroAPI::RegTensor<scaleT>& offsetVreg)
{
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
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
__simd_callee__ inline void LoadPerTokenTransposeScaleAndOffset(__local_mem__ scaleT* scaleUb,
                                                           __local_mem__ scaleT* offsetUb,
                                                           MicroAPI::RegTensor<scaleT>& scaleVreg,
                                                           MicroAPI::RegTensor<scaleT>& offsetVreg)
{
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
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
__simd_callee__ inline void GetPerGroupScaleAndOffset(__local_mem__ T* scaleUb,
                                                 __local_mem__ T* offsetUb, const int32_t start,
                                                 const AscendAntiQuantParam& para,
                                                 MicroAPI::RegTensor<T>& scaleReg,
                                                 MicroAPI::RegTensor<T>& offsetReg)
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
__simd_callee__ inline void StoreF32Res(__local_mem__ dstT* dstAddr, MicroAPI::RegTensor<float>& vreg,
                                   MicroAPI::MaskReg& preg)
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
__simd_callee__ inline void LoadSrc(__local_mem__ srcT* srcAddr, MicroAPI::RegTensor<srcT>& srcVreg)
{
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B8>(srcVreg, srcAddr);
    } else {
        MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(srcVreg, srcAddr);
    }
}

template <typename scaleT, typename srcT>
__simd_callee__ inline void ConvertSrc(MicroAPI::RegTensor<scaleT>& vreg, MicroAPI::RegTensor<srcT>& srcVreg,
                                  MicroAPI::MaskReg& preg)
{
    if constexpr (SupportType<scaleT, half>()) {
        MicroAPI::Cast<scaleT, srcT, layoutZMrgZ>(vreg, srcVreg, preg);
    } else if constexpr (SupportType<scaleT, bfloat16_t>()) {
        MicroAPI::RegTensor<half> f16Vreg;
        MicroAPI::Cast<half, srcT, layoutZMrgZ>(f16Vreg, srcVreg, preg);
        MicroAPI::Cast<bfloat16_t, half, MrgZRndR>(vreg, f16Vreg, preg);
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
__simd_callee__ inline void AddOffsetIfExist(MicroAPI::RegTensor<scaleT>& vreg, MicroAPI::RegTensor<scaleT>& offsetVreg,
                                        MicroAPI::MaskReg& preg)
{
    if constexpr (config.hasOffset) {
        MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, offsetVreg, preg);
    }
}

template <typename scaleT>
__simd_callee__ inline void GenZeroVreg(MicroAPI::RegTensor<scaleT>& vreg)
{
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
        MicroAPI::MaskReg b16FullPreg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(vreg, static_cast<scaleT>(0), b16FullPreg);
    } else {
        MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(vreg, static_cast<scaleT>(0), b32FullPreg);
    }
}

template <typename scaleT, const AscendAntiQuantConfig& config>
__simd_callee__ inline void ConvertToF32ScaleAndOffset(MicroAPI::RegTensor<scaleT>& scaleVreg,
                                                  MicroAPI::RegTensor<scaleT>& offsetVreg,
                                                  MicroAPI::MaskReg& preg,
                                                  MicroAPI::RegTensor<float>& f32ScaleVreg,
                                                  MicroAPI::RegTensor<float>& f32OffsetVreg)
{
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
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
__simd_callee__ inline void LoadNormScaleAndOffset(__local_mem__ scaleT* scaleAddr,
                                              __local_mem__ scaleT* offsetAddr,
                                              MicroAPI::RegTensor<scaleT>& scaleVreg,
                                              MicroAPI::RegTensor<scaleT>& offsetVreg)
{
    MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleAddr);
    if constexpr (config.hasOffset) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(offsetVreg, offsetAddr);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerTokenForB8VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb, const AscendAntiQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
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
            if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
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
__aicore__ inline void AntiQuantPerTokenForB8(const LocalTensor<dstT>& dstTensor,
                                              const LocalTensor<srcT>& srcTensor,
                                              const LocalTensor<scaleT>& scaleTensor,
                                              const LocalTensor<scaleT>& offsetTensor,
                                              const AscendAntiQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    AntiQuantPerTokenForB8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerTokenTransposeForB8VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb, const AscendAntiQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
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
            LoadPerTokenTransposeScaleAndOffset<scaleT, config>(scaleUb + j * vecLen,
                offsetUb + j * vecLen, scaleVreg, offsetVreg);
            LoadSrc<scaleT, srcT>((srcUb + i * para.n + j * vecLen), srcVreg);
            ConvertSrc<scaleT, srcT>(vreg, srcVreg, preg);
            AddOffsetIfExist<scaleT, config>(vreg, offsetVreg, preg);
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, scaleVreg, preg);
            if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
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
                                                       const LocalTensor<srcT>& srcTensor,
                                                       const LocalTensor<scaleT>& scaleTensor,
                                                       const LocalTensor<scaleT>& offsetTensor,
                                                       const AscendAntiQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    AntiQuantPerTokenTransposeForB8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerTokenForFp8VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb, const AscendAntiQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint32_t sreg = para.n;

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::RegTensor<float> f32ScaleVreg;
    MicroAPI::RegTensor<float> f32OffsetVreg;
    for (uint16_t i = 0; i < rowNum; ++i) {
        LoadPerTokenScaleAndOffset<scaleT, config>(scaleUb + i, offsetUb + i, scaleVreg, offsetVreg);
        sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                srcVreg, srcUb + i * para.n + j * vecLen);
            MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32Vreg, srcVreg, preg);
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

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantPerTokenForFp8(const LocalTensor<dstT>& dstTensor,
                                               const LocalTensor<srcT>& srcTensor,
                                               const LocalTensor<scaleT>& scaleTensor,
                                               const LocalTensor<scaleT>& offsetTensor,
                                               const AscendAntiQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    AntiQuantPerTokenForFp8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerTokenTransposeForFp8VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb, const AscendAntiQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint32_t sreg = para.n;

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::RegTensor<float> f32ScaleVreg;
    MicroAPI::RegTensor<float> f32OffsetVreg;
    for (uint16_t i = 0; i < rowNum; ++i) {
        sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadPerTokenTransposeScaleAndOffset<scaleT, config>(scaleUb + j * vecLen,
                offsetUb + j * vecLen, scaleVreg, offsetVreg);
            MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                srcVreg, srcUb + i * para.n + j * vecLen);
            MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32Vreg, srcVreg, preg);
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

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantPerTokenTransposeForFp8(const LocalTensor<dstT>& dstTensor,
                                                        const LocalTensor<srcT>& srcTensor,
                                                        const LocalTensor<scaleT>& scaleTensor,
                                                        const LocalTensor<scaleT>& offsetTensor,
                                                        const AscendAntiQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    AntiQuantPerTokenTransposeForFp8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para);
}

__aicore__ inline void ReplaceBf16VmulsWithVmul(MicroAPI::RegTensor<bfloat16_t>& vreg,
                                                const bfloat16_t scale, MicroAPI::MaskReg& preg)
{
    MicroAPI::RegTensor<bfloat16_t> bf16ScaleVreg;
    MicroAPI::Duplicate(bf16ScaleVreg, static_cast<bfloat16_t>(scale), preg);
    MicroAPI::Mul<bfloat16_t, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, bf16ScaleVreg, preg);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerGroupForColB8VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb, const AscendAntiQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
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
            GetPerGroupScaleAndOffset<scaleT, config>(scaleUb + i * scaleK, offsetUb + i * scaleK,
                j * vecLen, para, scaleVreg, offsetVreg);
            LoadSrc<scaleT, srcT>((srcUb + i * para.n + j * vecLen), srcVreg);
            ConvertSrc<scaleT, srcT>(vreg, srcVreg, preg);
            AddOffsetIfExist<scaleT, config>(vreg, offsetVreg, preg);
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, scaleVreg, preg);
            if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_NORM_B16>(dstUb + i * para.n + j * vecLen, vreg, preg);
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
__aicore__ inline void AntiQuantPerGroupForColB8(const LocalTensor<dstT>& dstTensor,
                                                 const LocalTensor<srcT>& srcTensor,
                                                 const LocalTensor<scaleT>& scaleTensor,
                                                 const LocalTensor<scaleT>& offsetTensor,
                                                 const AscendAntiQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    AntiQuantPerGroupForColB8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerGroupForColFp4VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, const AscendAntiQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(dstT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint32_t sreg = para.n;
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);

    MicroAPI::MaskReg preg, selPreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<dstT> dstVreg, dstSrcVreg, dstScaleVreg;
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<scaleT, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint16_t> indexVreg, gsizeVreg, bf16Zero, bf16Nan, e8m0Zero, e8m0Nan;
    Duplicate(bf16Zero, (uint16_t)0x0040);  // if e8m0 = 0b00000000, bf16 is 0x0040
    Duplicate(bf16Nan, (uint16_t)0x7fff);   // if e8m0 = 0b11111111, use 0x7fff as bf16 nan
    Duplicate(e8m0Zero, 0);
    Duplicate(e8m0Nan, (uint16_t)0x7f80);  // if e8m0 = 0b11111111, after << 7 is 0x7f80
    for (uint16_t i = 0; i < rowNum; ++i) {
        sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<dstT>(sreg);
            // step1 load scale procedure
            // gather fp8_e8m0x128 scale data, 8-bit elements is zero-extented to 16-bit
            MicroAPI::Duplicate(gsizeVreg, static_cast<uint16_t>(para.groupSize));
            MicroAPI::Arange((MicroAPI::RegTensor<int16_t> &)indexVreg, static_cast<int16_t>(j * vecLen));
            MicroAPI::Div(indexVreg, indexVreg, gsizeVreg, pregFull);
            MicroAPI::DataCopyGather<uint16_t, uint8_t, uint16_t>((MicroAPI::RegTensor<uint16_t> &)scaleVreg,
                (__local_mem__ uint8_t *)scaleUb + i * scaleK, indexVreg, pregFull);
            // fp8_e8m0x128 -> bf16x128
            MicroAPI::ShiftLefts<uint16_t, int16_t>((MicroAPI::RegTensor<uint16_t> &)dstScaleVreg,
                (MicroAPI::RegTensor<uint16_t> &)scaleVreg, ANTIQUANT_BF16_MAN_LEN, preg);
            // 00000000 and 11111111 need special process
            SelectZeroNan((MicroAPI::RegTensor<bfloat16_t> &)dstScaleVreg, bf16Zero, bf16Nan, e8m0Zero, e8m0Nan,
                selPreg, preg);
            // step2. load src procedure
            // 1. using UNPACK4_B8 to load 64xb8 and store as 64xb32
            // 2. using cast fp4->bf16, to cast 64xb32 to 128xbf16
            MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<uint8_t> &)srcVreg,
                (__local_mem__ uint8_t *)srcUb + (i * para.n + j * vecLen) / 2);
            MicroAPI::Cast<bfloat16_t, srcT, layoutZMrgZ>(
                (MicroAPI::RegTensor<bfloat16_t> &)dstSrcVreg, srcVreg, preg);
            // step3. dst = src * scale
            MicroAPI::Mul<bfloat16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<bfloat16_t> &)dstVreg,
                (MicroAPI::RegTensor<bfloat16_t> &)dstSrcVreg, (MicroAPI::RegTensor<bfloat16_t> &)dstScaleVreg, preg);
            // step4. cast dst to half if DstT is half
            if constexpr (SupportType<dstT, half>()) {
                MicroAPI::Cast<dstT, bfloat16_t, LayoutZMrgZRndRSatS>(
                    dstVreg, (MicroAPI::RegTensor<bfloat16_t> &)dstVreg, preg);
            }
            // step5. store dst->ub
            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_NORM_B16>(
                dstUb + i * para.n + j * vecLen, dstVreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantPerGroupForColFp4(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                  const LocalTensor<scaleT>& scaleTensor,const AscendAntiQuantParam& para)
{
    __local_mem__ dstT *dstUb = (__local_mem__ dstT *)dstTensor.GetPhyAddr();
    __local_mem__ srcT *srcUb = (__local_mem__ srcT *)srcTensor.GetPhyAddr();
    __local_mem__ scaleT *scaleUb = (__local_mem__ scaleT *)scaleTensor.GetPhyAddr();
    AntiQuantPerGroupForColFp4VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, para);
}

template <typename dstT, typename srcT, typename scaleT>
__simd_callee__ inline void AntiQuantPerGroupForRowFp4OneRow(__local_mem__ dstT *dstAddr, __local_mem__ srcT *srcAddr,
    __local_mem__ scaleT *scaleAddr, MicroAPI::RegTensor<dstT> &dstVreg, MicroAPI::RegTensor<srcT> &srcVreg,
    MicroAPI::RegTensor<scaleT> &scaleVreg, MicroAPI::RegTensor<dstT> &dstSrcVreg,
    MicroAPI::RegTensor<dstT> &dstScaleVreg, MicroAPI::MaskReg &preg, uint16_t repeat, uint32_t n, uint32_t vecLen)
{
    MicroAPI::MaskReg selPreg;
    MicroAPI::RegTensor<uint16_t> bf16Zero, bf16Nan, e8m0Zero, e8m0Nan;
    Duplicate(bf16Zero, (uint16_t)0x0040);  // if e8m0 = 0b00000000, bf16 is 0x0040
    Duplicate(bf16Nan, (uint16_t)0x7fff);   // if e8m0 = 0b11111111, use 0x7fff as bf16 nan
    Duplicate(e8m0Zero, 0);
    Duplicate(e8m0Nan, (uint16_t)0x7f80);  // if e8m0 = 0b11111111, after << 7 is 0x7f80
    for (uint16_t j = 0; j < repeat; ++j) {
        preg = MicroAPI::UpdateMask<dstT>(n);
        // step1: load scale procedure
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_UNPACK_B8>(scaleVreg, scaleAddr + j * vecLen);
        MicroAPI::ShiftLefts<uint16_t, int16_t>((MicroAPI::RegTensor<uint16_t> &)dstScaleVreg,
            (MicroAPI::RegTensor<uint16_t> &)scaleVreg,
            ANTIQUANT_BF16_MAN_LEN,
            preg);
        // 00000000 and 11111111 need special process
        SelectZeroNan(
            (MicroAPI::RegTensor<bfloat16_t> &)dstScaleVreg, bf16Zero, bf16Nan, e8m0Zero, e8m0Nan, selPreg, preg);
        // step2: load src
        MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<uint8_t> &)srcVreg, (__local_mem__ uint8_t *)srcAddr + (j * vecLen) / 2);
        MicroAPI::Cast<bfloat16_t, srcT, layoutZMrgZ>((MicroAPI::RegTensor<bfloat16_t> &)dstSrcVreg, srcVreg, preg);
        // step3. dst = src * scale
        MicroAPI::Mul<bfloat16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<bfloat16_t> &)dstVreg,
            (MicroAPI::RegTensor<bfloat16_t> &)dstSrcVreg, (MicroAPI::RegTensor<bfloat16_t> &)dstScaleVreg, preg);
        // step4. cast dst to half if DstT is half
        if constexpr (SupportType<dstT, half>()) {
            MicroAPI::Cast<dstT, bfloat16_t, LayoutZMrgZRndRSatS>(
                dstVreg, (MicroAPI::RegTensor<bfloat16_t> &)dstVreg, preg);
        }
        // step5. store dst->ub
        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_NORM_B16>(dstAddr + j * vecLen, dstVreg, preg);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerGroupForRowFp4VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, const AscendAntiQuantParam para, uint16_t rowNum, uint16_t tailRow)
{
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(dstT);
    uint16_t repeat = CeilDivision(para.n, vecLen);

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<dstT> dstVreg, dstSrcVreg, dstScaleVreg;
    for (uint16_t i0 = 0; i0 < mainRowGroup; ++i0) {
        for (uint16_t i1 = 0; i1 < static_cast<uint16_t>(para.groupSize); ++i1) {
            AntiQuantPerGroupForRowFp4OneRow<dstT, srcT, scaleT>(dstUb + (i0 * para.groupSize + i1) * para.n,
                srcUb + ((i0 * para.groupSize + i1) * para.n) / 2,
                scaleUb + i0 * para.n,
                dstVreg,
                srcVreg,
                scaleVreg,
                dstSrcVreg,
                dstScaleVreg,
                preg,
                repeat,
                para.n,
                vecLen);
        }
    }
    for (uint16_t i = 0; i < tailRow; ++i) {
        AntiQuantPerGroupForRowFp4OneRow<dstT, srcT, scaleT>(dstUb + (mainRowGroup * para.groupSize + i) * para.n,
            srcUb + ((mainRowGroup * para.groupSize + i) * para.n) / 2,
            scaleUb + mainRowGroup * para.n,
            dstVreg,
            srcVreg,
            scaleVreg,
            dstSrcVreg,
            dstScaleVreg,
            preg,
            repeat,
            para.n,
            vecLen);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantPerGroupForRowFp4(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                  const LocalTensor<scaleT>& scaleTensor, const AscendAntiQuantParam& para)
{
     __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t tailRow = rowNum % para.groupSize;

    AntiQuantPerGroupForRowFp4VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, para, rowNum, tailRow);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerGroupForColFp8VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb, const AscendAntiQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint32_t sreg = para.n;
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<float> f32SrcVreg;
    MicroAPI::RegTensor<float> f32ScaleVreg;
    MicroAPI::RegTensor<float> f32OffsetVreg;
    for (uint16_t i = 0; i < rowNum; ++i) {
        sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            GetPerGroupScaleAndOffset<scaleT, config>(scaleUb + i * scaleK, offsetUb + i * scaleK,
                j * vecLen, para, scaleVreg, offsetVreg);
            MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(srcVreg, srcUb + i * para.n + j * vecLen);
            MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32SrcVreg, srcVreg, preg);
            ConvertToF32ScaleAndOffset<scaleT, config>(scaleVreg, offsetVreg, preg, f32ScaleVreg, f32OffsetVreg);
            if constexpr (config.hasOffset) {
                if constexpr (SupportType<scaleT, float>()) {
                    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(f32SrcVreg, f32SrcVreg, offsetVreg, preg);
                } else {
                    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(f32SrcVreg, f32SrcVreg, f32OffsetVreg, preg);
                }
            }
            if constexpr (SupportType<scaleT, float>()) {
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32SrcVreg, f32SrcVreg, scaleVreg, preg);
            } else {
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32SrcVreg, f32SrcVreg, f32ScaleVreg, preg);
            }
            StoreF32Res<dstT>((dstUb + i * para.n + j * vecLen), f32SrcVreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantPerGroupForColFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                  const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
                                                  const AscendAntiQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    AntiQuantPerGroupForColFp8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_callee__ inline void AntiQuantPerGroupForRowB8TailBlock(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
                                                          __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb,
                                                          uint16_t repeat, uint16_t tailRow, uint32_t n, uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<scaleT> vreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            LoadNormScaleAndOffset<scaleT, config>((scaleUb + j * vecLen),
                (offsetUb + j * vecLen), scaleVreg, offsetVreg);
            preg = MicroAPI::UpdateMask<scaleT>(sreg);
            LoadSrc<scaleT, srcT>(srcUb + i * n + j * vecLen, srcVreg);
            ConvertSrc<scaleT, srcT>(vreg, srcVreg, preg);
            AddOffsetIfExist<scaleT, config>(vreg, offsetVreg, preg);
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, scaleVreg, preg);
            if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_NORM_B16>(dstUb + i * n + j * vecLen, vreg, preg);
            } else {
                if constexpr (SupportType<dstT, float>()) {
                    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * n + j * vecLen, vreg, preg);
                } else {
                    MicroAPI::RegTensor<dstT> tempVreg;
                    MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(tempVreg, vreg, preg);
                    MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * n + j * vecLen, tempVreg, preg);
                }
            }
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerGroupForRowB8VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb, const AscendAntiQuantParam para,
    uint16_t rowNum, uint16_t tailRow)
{
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint32_t vecLen = VECTOR_REG_WIDTH / sizeof(scaleT);
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
                LoadNormScaleAndOffset<scaleT, config>((scaleUb + i * para.n + k * vecLen),
                    (offsetUb + i * para.n + k * vecLen), scaleVreg, offsetVreg);
                preg = MicroAPI::UpdateMask<scaleT>(sreg);
                LoadSrc<scaleT, srcT>(srcUb + (i * para.groupSize + j) * para.n + k * vecLen, srcVreg);
                ConvertSrc<scaleT, srcT>(vreg, srcVreg, preg);
                AddOffsetIfExist<scaleT, config>(vreg, offsetVreg, preg);
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(vreg, vreg, scaleVreg, preg);
                if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                    MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_NORM_B16>(dstUb + (i * para.groupSize + j) * para.n + k * vecLen, vreg, preg);
                } else {
                    if constexpr (SupportType<dstT, float>()) {
                        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + (i * para.groupSize + j) * para.n + k * vecLen, vreg, preg);
                    } else {
                        MicroAPI::RegTensor<dstT> tempVreg;
                        MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(tempVreg, vreg, preg);
                        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + (i * para.groupSize + j) * para.n + k * vecLen, tempVreg, preg);
                    }
                }
            }
        }
    }
    AntiQuantPerGroupForRowB8TailBlock<dstT, srcT, scaleT, config>(
        dstUb + mainRowGroup * para.groupSize * para.n, srcUb + mainRowGroup * para.groupSize * para.n,
        scaleUb + mainRowGroup * para.n, offsetUb + mainRowGroup * para.n, repeat, tailRow, para.n, vecLen);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantPerGroupForRowB8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                 const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
                                                 const AscendAntiQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t tailRow = rowNum % para.groupSize;
    AntiQuantPerGroupForRowB8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para, rowNum, tailRow);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_callee__ inline void AntiQuantPerGroupForRowFp8TailBlock(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
                                                           __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb,
                                                           uint16_t repeat, uint16_t tailRow, uint32_t n, uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<float> f32ScaleVreg;
    MicroAPI::RegTensor<float> f32OffsetVreg;
    MicroAPI::RegTensor<float> f32SrcVreg;
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            LoadNormScaleAndOffset<scaleT, config>((scaleUb + j * vecLen),
                (offsetUb + j * vecLen), scaleVreg, offsetVreg);
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                srcVreg, srcUb + i * n + j * vecLen);
            MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32SrcVreg, srcVreg, preg);
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

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__simd_vf__ inline void AntiQuantPerGroupForRowFp8VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, __local_mem__ scaleT* offsetUb, const AscendAntiQuantParam para,
    uint16_t rowNum, uint16_t tailRow)
{
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = CeilDivision(para.n, vecLen);

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<float> f32Svreg;
    MicroAPI::RegTensor<float> f32Ovreg;
    MicroAPI::RegTensor<float> f32SrcVreg;
    for (uint16_t i = 0; i < mainRowGroup; ++i) {
        for (uint16_t j = 0; j < static_cast<uint16_t>(para.groupSize); ++j) {
            uint32_t sreg = para.n;
            for (uint16_t k = 0; k < repeat; ++k) {
                preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                LoadNormScaleAndOffset<scaleT, config>((scaleUb + i * para.n + k * vecLen),
                    (offsetUb + i * para.n + k * vecLen), scaleVreg, offsetVreg);
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    srcVreg, (srcUb + (i * para.groupSize + j) * para.n + k * vecLen));
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(f32SrcVreg, srcVreg, preg);
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
    AntiQuantPerGroupForRowFp8TailBlock<dstT, srcT, scaleT, config>(
        dstUb + mainRowGroup * para.groupSize * para.n, srcUb + mainRowGroup * para.groupSize * para.n,
        scaleUb + mainRowGroup * para.n, offsetUb + mainRowGroup * para.n, repeat, tailRow, para.n, vecLen);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AntiQuantPerGroupForRowFp8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                  const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
                                                  const AscendAntiQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    __local_mem__ scaleT* offsetUb = (__local_mem__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t tailRow = rowNum % para.groupSize;
    AntiQuantPerGroupForRowFp8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para, rowNum, tailRow);
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AscendAntiQuantPerToken(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                               const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                               const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    if constexpr (config.isTranspose) {
        if constexpr (SupportType<srcT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
            AntiQuantPerTokenTransposeForFp8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor,
                offsetTensor, para);
        } else {
            AntiQuantPerTokenTransposeForB8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor,
                offsetTensor, para);
        }
        return;
    }
    if constexpr (SupportType<srcT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        AntiQuantPerTokenForFp8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor,
            offsetTensor, para);
    } else {
        AntiQuantPerTokenForB8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor,
            offsetTensor, para);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AscendAntiQuantPerGroupForCol(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                     const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                                     const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    if constexpr (SupportType<srcT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        AntiQuantPerGroupForColFp8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor,
            offsetTensor, para);
    } else if constexpr (SupportType<srcT, fp4x2_e1m2_t, fp4x2_e2m1_t>()) {
        // fp4 dosen't count offset
        AntiQuantPerGroupForColFp4<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, para);
    } else {
        AntiQuantPerGroupForColB8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor,
            offsetTensor, para);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig& config>
__aicore__ inline void AscendAntiQuantPerGroupForRow(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                     const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                                     const LocalTensor<scaleT>& offsetTensor, const AscendAntiQuantParam& para)
{
    if constexpr (SupportType<srcT, fp8_e4m3fn_t, fp8_e5m2_t>()) {
        AntiQuantPerGroupForRowFp8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor,
            offsetTensor, para);
    } else if constexpr (SupportType<srcT, fp4x2_e1m2_t, fp4x2_e2m1_t>()) {
        // fp4 dosen't count offset
        AntiQuantPerGroupForRowFp4<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, para);
    } else {
        AntiQuantPerGroupForRowB8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor,
            offsetTensor, para);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendAntiQuantConfig &config,
    const AscendAntiQuantPolicy &policy>
__aicore__ inline void AscendAntiQuantImpl(const LocalTensor<dstT> &dstTensor, const LocalTensor<srcT> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const LocalTensor<scaleT> &scaleTensor,
    const LocalTensor<scaleT> &offsetTensor, const AscendAntiQuantParam &para)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<dstT, bfloat16_t, half, float>(),
        "AscendAntiQuant only support bfloat16_t/half/float output dtype");
    static_assert(SupportType<scaleT, bfloat16_t, half, float, fp8_e8m0_t>(),
        "AscendAntiQuant only support bfloat16_t/half/float scale/offset dtype");
    static_assert((policy == AscendAntiQuantPolicy::PER_TOKEN || policy == AscendAntiQuantPolicy::PER_GROUP),
        "unsupported policy for AscendAntiQuant in current device!");
    ASCENDC_ASSERT((para.calCount <= srcTensor.GetSize() && para.calCount <= dstTensor.GetSize() && para.calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]",
            para.calCount, srcTensor.GetSize(), dstTensor.GetSize());
    });
    ASCENDC_ASSERT(
        (para.calCount % para.n == 0), { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of n!"); });
    if constexpr (policy == AscendAntiQuantPolicy::PER_TOKEN) {
        static_assert(SupportType<srcT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t>(),
            "AscendAntiQuant PerToken only support int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t input dtype");
        AscendAntiQuantPerToken<dstT, srcT, scaleT, config>(
            dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
    } else if constexpr (policy == AscendAntiQuantPolicy::PER_GROUP) {
        static_assert(
            SupportType<srcT, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t, fp4x2_e1m2_t, fp4x2_e2m1_t>(),
            "AscendAntiQuant PerGroup only support "
            "int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t/fp4x2_e1m2_t/fp4x2_e2m1_t input dtype");
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
__aicore__ inline void AscendAntiQuantImplCommon(const LocalTensor<DstType> &dst, const LocalTensor<SrcType> &src,
    const LocalTensor<DstType> &offset, const LocalTensor<DstType> &scale, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const uint32_t k, const AntiQuantShapeInfo& shapeInfo = {})
{
    AntiQuantPerchannelImpl<SrcType, DstType, isTranspose>(dst, src, offset, scale, sharedTmpBuffer, k, shapeInfo);
}

template <typename SrcType, typename DstType, bool isTranspose>
__aicore__ inline void AscendAntiQuantImplCommon(const LocalTensor<DstType> &dst, const LocalTensor<SrcType> &src,
    const DstType offset, const DstType scale, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t k,
    const AntiQuantShapeInfo& shapeInfo = {})
{
    AntiQuantPertensorImpl<SrcType, DstType>(dst, src, offset, scale, sharedTmpBuffer, k, shapeInfo);
}
}  // namespace AscendC
#endif  // IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_C310_IMPL_H
