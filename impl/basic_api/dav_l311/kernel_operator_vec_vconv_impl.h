/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_operator_vec_vconv_impl.h
 * \brief AscendC l311 support vector cast api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
namespace AscendC {

// micro adaptor
template <typename T, typename U, RoundMode roundMode, Mode mode, SatMode satMode, PartMode partMode>
__aicore__ inline void CastAdaptor(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask) {
    if constexpr (roundMode != RoundMode::CAST_NONE && satMode != SatMode::UNKNOWN && partMode != PartMode::UNKNOWN) {
        return Cast<T, U, roundMode, mode, satMode, partMode>(dstReg, srcReg, mask);
    }
    else if constexpr (roundMode != RoundMode::CAST_NONE && satMode != SatMode::UNKNOWN && partMode == PartMode::UNKNOWN) {
        return Cast<T, U, roundMode, mode, satMode>(dstReg, srcReg, mask);
    }
    else if constexpr (roundMode != RoundMode::CAST_NONE && satMode == SatMode::UNKNOWN && partMode != PartMode::UNKNOWN) {
        return Cast<T, U, roundMode, mode, partMode>(dstReg, srcReg, mask);
    }
    else if constexpr (roundMode == RoundMode::CAST_NONE && satMode != SatMode::UNKNOWN && partMode != PartMode::UNKNOWN) {
        return Cast<T, U, mode, satMode, partMode>(dstReg, srcReg, mask);
    }
    else if constexpr (roundMode == RoundMode::CAST_NONE && satMode == SatMode::UNKNOWN && partMode != PartMode::UNKNOWN) {
        return Cast<T, U, mode, partMode>(dstReg, srcReg, mask);
    }
    else if constexpr (roundMode != RoundMode::CAST_NONE && satMode == SatMode::UNKNOWN && partMode == PartMode::UNKNOWN) {
        return Cast<T, U, roundMode, mode>(dstReg, srcReg, mask);
    }
    else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport pattern of CastAdaptor"); });
    }
}

template <typename T, typename U, RoundMode roundMode, Mode mode, SatMode satMode, PPMode ppMode>
__aicore__ inline void CastAdaptor(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask) {
    if constexpr (roundMode != RoundMode::CAST_NONE && satMode != SatMode::UNKNOWN && ppMode != PPMode::UNKNOWN) {
        return Cast<T, U, roundMode, mode, satMode, ppMode>(dstReg, srcReg, mask);
    }
    if constexpr (roundMode == RoundMode::CAST_NONE && satMode != SatMode::UNKNOWN && ppMode != PPMode::UNKNOWN) {
        return Cast<T, U, mode, satMode, ppMode>(dstReg, srcReg, mask);
    }
    else if constexpr (roundMode == RoundMode::CAST_NONE && satMode == SatMode::UNKNOWN && ppMode != PPMode::UNKNOWN) {
        return Cast<T, U, mode, ppMode>(dstReg, srcReg, mask);
    }
    else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport pattern of CastAdaptor"); });
    }
}

// For Cast L2
#define CAST_LOWER_HALF(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)                                             \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType* dst,                                                            \
        __ubuf__ srcType* src, const uint32_t count)                                                                                 \
    {                                                                                                                                   \
        __VEC_SCOPE__                                                                                                                   \
        {                                                                                                                               \
            uint32_t len = count;                                                                                                    \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(dstType));                                                        \
            uint16_t repeatTime = CeilDivision(count, sregLower);                                                                   \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; i++) {                                                                      \
                RegTensor<srcType> input0;                                                                                              \
                RegTensor<srcType> input1;                                                                                              \
                RegTensor<dstType> mid0;                                                                                                \
                RegTensor<dstType> mid1;                                                                                                \
                RegTensor<dstType> output;                                                                                              \
                MaskReg p0 = CreatePredicate<dstType>(len);                                                                             \
                DataCopy(input0, src, i * sregLower);                                                                                   \
                DataCopy(input1, src + (sregLower >> 1), i * sregLower);                                                                \
                DeInterleave(input0, input1, input0, input1);                                                                           \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::EVEN>(mid0, input0, p0);                                \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::ODD>(mid1, input1, p0);                                 \
                Or(output, mid0, mid1, p0);                                                                                             \
                DataCopy(dst, output, i * sregLower, p0);                                                                               \
            }                                                                                                                           \
        }                                                                                                                               \
    }                                                                                                                                   \

// For Cast L2
#define CAST_UPPER_HALF(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)                             \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType* dst,                                            \
        __ubuf__ srcType* src, const uint32_t count)                                                                 \
    {                                                                                                                   \
        __VEC_SCOPE__                                                                                                   \
        {                                                                                                               \
            uint32_t len = count;                                                                                    \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(srcType));                                        \
            uint16_t repeatTime = CeilDivision(count, sregLower);                                                   \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; i++) {                                                      \
                RegTensor<srcType> input;                                                                               \
                RegTensor<dstType> output0;                                                                             \
                RegTensor<dstType> output1;                                                                             \
                RegTensor<dstType> output_even;                                                                         \
                RegTensor<dstType> output_odd;                                                                          \
                MaskReg p0 = CreatePredicate<srcType>(len);                                                             \
                MaskReg p1;                                                                                             \
                MaskReg p2;                                                                                             \
                DataCopy(input, src, i * sregLower);                                                                    \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::EVEN>(output_even, input, p0);          \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::ODD>(output_odd, input, p0);            \
                Interleave(output0, output1, output_even, output_odd);                                                  \
                PredicateUnPack<HiloPart::Lower>(p1, p0);                                                               \
                PredicateUnPack<HiloPart::Higher>(p2, p0);                                                              \
                DataCopy(dst, output0, i * sregLower, p1);                                                              \
                DataCopy(dst + (sregLower >> 1), output1, i * sregLower, p2);                                           \
            }                                                                                                           \
        }                                                                                                               \
    }                                                                                                                   \

// For Cast L2
#define CAST_LOWER_QUATER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)                           \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType* dst,                                            \
        __ubuf__ srcType* src, const uint32_t count)                                                                 \
    {                                                                                                                   \
        __VEC_SCOPE__                                                                                                   \
        {                                                                                                               \
            uint32_t len = count;                                                                                    \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(dstType));                                        \
            uint16_t repeatTime = CeilDivision(count, sregLower);                                                   \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; i++) {                                                      \
                RegTensor<srcType> input0;                                                                              \
                RegTensor<srcType> input1;                                                                              \
                RegTensor<srcType> input2;                                                                              \
                RegTensor<srcType> input3;                                                                              \
                RegTensor<dstType> output;                                                                              \
                RegTensor<dstType> mid0;                                                                                \
                RegTensor<dstType> mid1;                                                                                \
                RegTensor<dstType> mid2;                                                                                \
                RegTensor<dstType> mid3;                                                                                \
                MaskReg p0 = CreatePredicate<dstType>(len);                                                             \
                DataCopy(input0, src, i * sregLower);                                                                   \
                DataCopy(input1, src + (sregLower >> 2)*1, i * sregLower);                                              \
                DataCopy(input2, src + (sregLower >> 2)*2, i * sregLower);                                              \
                DataCopy(input3, src + (sregLower >> 2)*3, i * sregLower);                                              \
                DeInterleave(input0, input1, input0, input1);                                                           \
                DeInterleave(input2, input3, input2, input3);                                                           \
                DeInterleave(input0, input2, input0, input2);                                                           \
                DeInterleave(input1, input3, input1, input3);                                                           \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::ZERO>(mid0, input0, p0);                  \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::ONE>(mid1, input1, p0);                   \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::TWO>(mid2, input2, p0);                   \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::THREE>(mid3, input3, p0);                 \
                Or(mid0, mid0, mid1, p0);                                                                               \
                Or(mid2, mid2, mid3, p0);                                                                               \
                Or(output, mid0, mid2, p0);                                                                             \
                DataCopy(dst, output, i * sregLower, p0);                                                               \
            }                                                                                                           \
        }                                                                                                               \
    }                                                                                                                   \

// For Cast L2
#define CAST_UPPER_QUATER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)                           \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType* dst,                                            \
        __ubuf__ srcType* src, const uint32_t count)                                                                 \
    {                                                                                                                   \
        __VEC_SCOPE__                                                                                                   \
        {                                                                                                               \
            uint32_t len = count;                                                                                    \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(srcType));                                        \
            uint16_t repeatTime = CeilDivision(count, sregLower);                                                   \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; i++) {                                                      \
                RegTensor<srcType> input;                                                                               \
                RegTensor<dstType> output0;                                                                             \
                RegTensor<dstType> output1;                                                                             \
                RegTensor<dstType> output2;                                                                             \
                RegTensor<dstType> output3;                                                                             \
                RegTensor<dstType> mid0;                                                                                \
                RegTensor<dstType> mid1;                                                                                \
                RegTensor<dstType> mid2;                                                                                \
                RegTensor<dstType> mid3;                                                                                \
                RegTensor<dstType> mid4;                                                                                \
                RegTensor<dstType> mid5;                                                                                \
                RegTensor<dstType> mid6;                                                                                \
                RegTensor<dstType> mid7;                                                                                \
                MaskReg p0 = CreatePredicate<srcType>(len);                                                             \
                MaskReg p1;                                                                                             \
                MaskReg p2;                                                                                             \
                MaskReg p11;                                                                                            \
                MaskReg p12;                                                                                            \
                MaskReg p21;                                                                                            \
                MaskReg p22;                                                                                            \
                DataCopy(input, src, i * sregLower);                                                                    \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::ZERO>(mid0, input, p0);                   \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::ONE>(mid1, input, p0);                    \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::TWO>(mid2, input, p0);                    \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::THREE>(mid3, input, p0);                  \
                Interleave(mid4, mid5, mid0, mid2);                                                                     \
                Interleave(mid6, mid7, mid1, mid3);                                                                     \
                Interleave(output0, output1, mid4, mid6);                                                               \
                Interleave(output2, output3, mid5, mid7);                                                               \
                PredicateUnPack<HiloPart::Lower>(p1, p0);                                                               \
                PredicateUnPack<HiloPart::Lower>(p11, p1);                                                              \
                PredicateUnPack<HiloPart::Higher>(p12, p1);                                                             \
                PredicateUnPack<HiloPart::Higher>(p2, p0);                                                              \
                PredicateUnPack<HiloPart::Lower>(p21, p2);                                                              \
                PredicateUnPack<HiloPart::Higher>(p22, p2);                                                             \
                DataCopy(dst, output0, i * sregLower, p11);                                                             \
                DataCopy(dst + (sregLower >> 2)*1, output1, i * sregLower, p12);                                        \
                DataCopy(dst + (sregLower >> 2)*2, output2, i * sregLower, p21);                                        \
                DataCopy(dst + (sregLower >> 2)*3, output3, i * sregLower, p22);                                        \
            }                                                                                                           \
        }                                                                                                               \
    }                                                                                                                   \

// For Cast L2
#define CAST_EQUAL(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)                                  \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType* dst,                                            \
        __ubuf__ srcType* src, const uint32_t count)                                                                 \
    {                                                                                                                   \
        __VEC_SCOPE__                                                                                                   \
        {                                                                                                               \
            uint32_t len = count;                                                                                    \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(srcType));                                        \
            uint16_t repeatTime = CeilDivision(count, sregLower);                                                   \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; i++) {                                                      \
                RegTensor<srcType> input;                                                                               \
                RegTensor<dstType> output;                                                                              \
                MaskReg p0 = CreatePredicate<dstType>(len);                                                             \
                DataCopy(input, src, i * sregLower);                                                                    \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::UNKNOWN>(output, input, p0);            \
                DataCopy(dst, output, i * sregLower, p0);                                                               \
            }                                                                                                           \
        }                                                                                                               \
    }                                                                                                                   \

// For Truncate L2
#define CAST_TRUNCATE(dType, rndStr, rndMode, mode)                                                                     \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dType* dst,                                              \
        __ubuf__ dType* src, const uint32_t count)                                                                   \
    {                                                                                                                   \
        __VEC_SCOPE__                                                                                                   \
        {                                                                                                               \
            uint32_t len = count;                                                                                    \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(dType));                                          \
            uint16_t repeatTime = CeilDivision(count, sregLower);                                                   \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; i++) {                                                      \
                RegTensor<dType> input;                                                                                 \
                RegTensor<dType> output;                                                                                \
                MaskReg p0 = CreatePredicate<dType>(len);                                                               \
                DataCopy(input, src, i * sregLower);                                                                    \
                Truncate<dType, rndMode, mode>(output, input, p0);                                                      \
                DataCopy(dst, output, i * sregLower, p0);                                                               \
            }                                                                                                           \
        }                                                                                                               \
    }

#define LV0_LOAD_UPPER_HALF(srcType)                                                                            \
    DataCopy<srcType, PostLiteral::POST_MODE_UPDATE>(vreg0, src, strideConfig0, strideOffset0, preg);           \
    Interleave(vreg0, vregTmp, vreg0, vregTmp)

#define LV0_LOAD_UPPER_QUATER(srcType)                                                                      \
    DataCopy<srcType, PostLiteral::POST_MODE_UPDATE>(vreg0, src, strideConfig0, strideOffset0, preg);       \
    Interleave(vreg0, vregTmp, vreg0, vregTmp);                                                             \
    Interleave(vreg0, vregTmp, vreg0, vregTmp)

#define LV0_LOAD_LOWER(srcType)                                                                             \
    DataCopy<srcType, PostLiteral::POST_MODE_UPDATE>(vreg0, src, strideConfig0, strideOffset0, preg)

#define LV0_LOAD_EQUAL(srcType)                                                                             \
    DataCopy<srcType, PostLiteral::POST_MODE_UPDATE>(vreg0, src, strideConfig0, strideOffset0, preg)

#define LV0_STORE_UPPER(dstType)                                                                            \
    DataCopy<dstType, PostLiteral::POST_MODE_UPDATE>(dst, vreg1, strideConfig1, strideOffset1, preg_dst)

#define LV0_STORE_LOWER_HALF(dstType)                                                                           \
    DeInterleave(vreg1, vregTmp, vreg1, vregTmp);                                                               \
    DataCopy<dstType, PostLiteral::POST_MODE_UPDATE>(dst, vreg1, strideConfig1, strideOffset1, preg_dst)

#define LV0_STORE_LOWER_QUATER(dstType)                                                                         \
    DeInterleave(vreg1, vregTmp, vreg1, vregTmp);                                                               \
    DeInterleave(vreg1, vregTmp, vreg1, vregTmp);                                                               \
    DataCopy<dstType, PostLiteral::POST_MODE_UPDATE>(dst, vreg1, strideConfig1, strideOffset1, preg_dst)

#define LV0_STORE_EQUAL(dstType)                                                                                \
    DataCopy<dstType, PostLiteral::POST_MODE_UPDATE>(dst, vreg1, strideConfig1, strideOffset1, preg)

#define BIT_GET_MASK_UPPER_HALF(srcType, dstType)                           \
    MaskReg preg_dst = MovePredicate<dstType>();                            \
    MaskReg preg;                                                           \
    PredicatePack<HiloPart::Lower>(preg, preg_dst);                         \
    RegTensor<srcType> vregTmp

#define BIT_GET_MASK_UPPER_QUATER(srcType, dstType)                         \
    MaskReg preg_dst = MovePredicate<dstType>();                            \
    MaskReg pregTmp;                                                        \
    MaskReg preg;                                                           \
    PredicatePack<HiloPart::Lower>(pregTmp, preg_dst);                      \
    PredicatePack<HiloPart::Lower>(preg, pregTmp);                          \
    RegTensor<srcType> vregTmp

#define BIT_GET_MASK_LOWER_HALF(srcType, dstType)                           \
    MaskReg preg = MovePredicate<srcType>();                                \
    MaskReg preg_dst;                                                       \
    PredicatePack<HiloPart::Lower>(preg_dst, preg);                         \
    RegTensor<dstType> vregTmp

#define BIT_GET_MASK_LOWER_QUATER(srcType, dstType)                         \
    MaskReg preg = MovePredicate<srcType>();                                \
    MaskReg pregTmp;                                                        \
    MaskReg preg_dst;                                                       \
    PredicatePack<HiloPart::Lower>(pregTmp, preg);                          \
    PredicatePack<HiloPart::Lower>(preg_dst, pregTmp);                      \
    RegTensor<dstType> vregTmp

#define BIT_GET_MASK_EQUAL(srcType, dstType)                                \
    MaskReg preg = MovePredicate<srcType>()

#define COUNTER_GET_MASK_UPPER_HALF(srcType, dstType)                       \
    uint32_t sreg = (uint32_t)mask;                                         \
    MaskReg preg = CreatePredicate<srcType>(sreg);                          \
    MaskReg preg_dst;                                                       \
    PredicateUnPack<HiloPart::Lower>(preg_dst, preg);                       \
    RegTensor<srcType> vregTmp

#define COUNTER_GET_MASK_UPPER_QUATER(srcType, dstType)                     \
    uint32_t sreg = (uint32_t)mask;                                         \
    MaskReg preg = CreatePredicate<srcType>(sreg);                          \
    MaskReg pregTmp;                                                        \
    MaskReg preg_dst;                                                       \
    PredicateUnPack<HiloPart::Lower>(pregTmp, preg);                        \
    PredicateUnPack<HiloPart::Lower>(preg_dst, pregTmp);                    \
    RegTensor<srcType> vregTmp

#define COUNTER_GET_MASK_LOWER_HALF(srcType, dstType)                       \
    uint32_t sreg = (uint32_t)mask;                                         \
    MaskReg preg = CreatePredicate<srcType>(sreg);                          \
    MaskReg preg_dst;                                                       \
    PredicatePack<HiloPart::Lower>(preg_dst, preg);                         \
    RegTensor<dstType> vregTmp

#define COUNTER_GET_MASK_LOWER_QUATER(srcType, dstType)                     \
    uint32_t sreg = (uint32_t)mask;                                         \
    MaskReg preg = CreatePredicate<srcType>(sreg);                          \
    MaskReg pregTmp;                                                        \
    MaskReg preg_dst;                                                       \
    PredicatePack<HiloPart::Lower>(pregTmp, preg);                          \
    PredicatePack<HiloPart::Lower>(preg_dst, pregTmp);                      \
    RegTensor<dstType> vregTmp

#define COUNTER_GET_MASK_EQUAL(srcType, dstType)                            \
    uint32_t sreg = (uint32_t)mask;                                         \
    MaskReg preg = CreatePredicate<srcType>(sreg)

#define CAST_TO_EQUAL(dstType, srcType, rndMode, satMode, mode)                                     \
    CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::UNKNOWN>(vreg1, vreg0, preg)

#define LOWER_TO_HALF(dstType, srcType, rndMode, satMode, mode)                                     \
    CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::EVEN>(vreg1, vreg0, preg)

#define UPPER_TO_HALF(dstType, srcType, rndMode, satMode, mode)                                     \
    CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::EVEN>(vreg1, vreg0, preg_dst)

#define LOWER_TO_QUATER(dstType, srcType, rndMode, satMode, mode)                                   \
    CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::ZERO>(vreg1, vreg0, preg)

#define UPPER_TO_QUATER(dstType, srcType, rndMode, satMode, mode)                                   \
    CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::ZERO>(vreg1, vreg0, preg_dst)

#define TRUNCATE_ROUND(dstType, srcType, rndMode, satMode, mode)                                    \
    Truncate<dstType, rndMode, mode>(vreg1, vreg0, preg)
// common vf function of Cast::Level 0
#define CAST_LV0_VF(srcType, dstType, srcBits, dstBits,                                                         \
    rndMode, satMode, mode, getMask, loadFunc, castFunc, storeFunc)                                             \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        RegTensor<srcType> vreg0;                                                                               \
        RegTensor<dstType> vreg1;                                                                               \
        getMask(srcType, dstType);                                                                              \
        uint32_t strideConfig0 = (uint32_t)repeatParams.srcBlkStride;                                           \
        uint32_t strideConfig1 = (uint32_t)repeatParams.dstBlkStride;                                           \
        uint32_t strideOffset0 = (uint32_t)repeatParams.srcRepStride;                                           \
        uint32_t strideOffset1 = (uint32_t)repeatParams.dstRepStride;                                           \
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {                                                  \
            loadFunc(srcType);                                                                                  \
            castFunc(dstType, srcType, rndMode, satMode, mode);                                                 \
            storeFunc(dstType);                                                                                 \
        }                                                                                                       \
    }

// Cast::Level 0 - mask bit mode
#define REGISTER_CAST_BIT(srcType, dstType, srcBits, dstBits,                                               \
    rndStr, rndMode, satMode, mode, getMask, loadFunc, castFunc, storeFunc)                                 \
    template <typename T = dstType, typename U = srcType, bool isSetMask = true>                            \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType* dst, __ubuf__ srcType* src,         \
        const uint64_t mask[2], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                 \
    {                                                                                                       \
        if constexpr (isSetMask) {                                                                          \
            if constexpr (srcBits > dstBits) {                                                              \
                SetVectorMask<srcType>(mask[1], mask[0]);                                                   \
            }                                                                                               \
            else {                                                                                          \
                SetVectorMask<dstType>(mask[1], mask[0]);                                                   \
            }                                                                                               \
        }                                                                                                   \
        CAST_LV0_VF(srcType, dstType, srcBits, dstBits, rndMode,                                            \
            satMode, mode, getMask, loadFunc, castFunc, storeFunc);                                         \
    }

// Cast::Level 0 - mask counter mode
#define REGISTER_CAST_COUNTER(srcType, dstType, srcBits, dstBits,                                           \
    rndStr, rndMode, satMode, mode, getMask, loadFunc, castFunc, storeFunc)                                 \
    template <typename T = dstType, typename U = srcType, bool isSetMask = true>                            \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType* dst, __ubuf__ srcType* src,         \
        const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                    \
    {                                                                                                       \
        CAST_LV0_VF(srcType, dstType, srcBits, dstBits, rndMode,                                            \
            satMode, mode, getMask, loadFunc, castFunc, storeFunc);                                         \
    }

#define REGISTER_CAST_LOWER_HALF(rndStr, rndMode, srcType, dstType,                                     \
        srcBits, dstBits, satMode, mode)                                                                \
    CAST_LOWER_HALF(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode);                \
    REGISTER_CAST_BIT(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode,               \
        BIT_GET_MASK_LOWER_HALF, LV0_LOAD_LOWER, LOWER_TO_HALF, LV0_STORE_LOWER_HALF);                  \
    REGISTER_CAST_COUNTER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode,           \
        COUNTER_GET_MASK_LOWER_HALF, LV0_LOAD_LOWER, LOWER_TO_HALF, LV0_STORE_LOWER_HALF)

#define REGISTER_CAST_UPPER_HALF(rndStr, rndMode, srcType, dstType,                                     \
        srcBits, dstBits, satMode, mode)                                                                \
    CAST_UPPER_HALF(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode);                \
    REGISTER_CAST_BIT(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode,               \
        BIT_GET_MASK_UPPER_HALF, LV0_LOAD_UPPER_HALF, UPPER_TO_HALF, LV0_STORE_UPPER);                  \
    REGISTER_CAST_COUNTER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode,           \
        COUNTER_GET_MASK_UPPER_HALF, LV0_LOAD_UPPER_HALF, UPPER_TO_HALF, LV0_STORE_UPPER)

#define REGISTER_CAST_LOWER_QUATER(rndStr, rndMode, srcType, dstType,                                   \
        srcBits, dstBits, satMode, mode)                                                                \
    CAST_LOWER_QUATER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode);              \
    REGISTER_CAST_BIT(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode,               \
        BIT_GET_MASK_LOWER_QUATER, LV0_LOAD_LOWER, LOWER_TO_QUATER, LV0_STORE_LOWER_QUATER);            \
    REGISTER_CAST_COUNTER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode,           \
        COUNTER_GET_MASK_LOWER_QUATER, LV0_LOAD_LOWER, LOWER_TO_QUATER, LV0_STORE_LOWER_QUATER)

#define REGISTER_CAST_UPPER_QUATER(rndStr, rndMode, srcType, dstType,                                   \
        srcBits, dstBits, satMode, mode)                                                                 \
    CAST_UPPER_QUATER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode);              \
    REGISTER_CAST_BIT(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode,               \
        BIT_GET_MASK_UPPER_QUATER, LV0_LOAD_UPPER_QUATER, UPPER_TO_QUATER, LV0_STORE_UPPER);            \
    REGISTER_CAST_COUNTER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode,           \
        COUNTER_GET_MASK_UPPER_QUATER, LV0_LOAD_UPPER_QUATER, UPPER_TO_QUATER, LV0_STORE_UPPER)

#define REGISTER_CAST_EQUAL(rndStr, rndMode, srcType, dstType,                                          \
        srcBits, dstBits, satMode, mode)                                                                \
    CAST_EQUAL(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode);                     \
    REGISTER_CAST_BIT(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode,               \
        BIT_GET_MASK_EQUAL, LV0_LOAD_EQUAL, CAST_TO_EQUAL, LV0_STORE_EQUAL);                            \
    REGISTER_CAST_COUNTER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode,           \
        COUNTER_GET_MASK_EQUAL, LV0_LOAD_EQUAL, CAST_TO_EQUAL, LV0_STORE_EQUAL)

#define REGISTER_CAST_TRUNCATE(rndStr, rndMode, srcType, dstType,                                       \
        srcBits, dstBits, satMode, mode)                                                                \
    CAST_TRUNCATE(srcType, rndStr, rndMode, mode);                                                      \
    REGISTER_CAST_BIT(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode,               \
        BIT_GET_MASK_EQUAL, LV0_LOAD_EQUAL, TRUNCATE_ROUND, LV0_STORE_EQUAL);                           \
    REGISTER_CAST_COUNTER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode,           \
        COUNTER_GET_MASK_EQUAL, LV0_LOAD_EQUAL, TRUNCATE_ROUND, LV0_STORE_EQUAL)

#define REGISTER_CAST_LV2_NOT_SUPPORTED(rndStr, srcType, dstType)                                       \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType* dst, __ubuf__ srcType* src,     \
        const uint32_t count)                                                                        \
        {                                                                                               \
            ASCENDC_ASSERT((false), {                                                                   \
                KERNEL_LOG(KERNEL_ERROR,                                                                \
                    "rndStr from srcType to dstType not supported!");                                   \
            });                                                                                         \
        }

#define REGISTER_CAST_COUNTER_NOT_SUPPORTED(rndStr, srcType, dstType)                                   \
    template <typename T = dstType, typename U = srcType, bool isSetMask = true>                        \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType* dst, __ubuf__ srcType* src,     \
        const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)                \
        {                                                                                               \
            ASCENDC_ASSERT((false), {                                                                   \
                KERNEL_LOG(KERNEL_ERROR,                                                                \
                    "rndStr from srcType to dstType not supported!");                                   \
            });                                                                                         \
        }

#define REGISTER_CAST_BIT_NOT_SUPPORTED(rndStr, srcType, dstType)                                       \
    template <typename T = dstType, typename U = srcType, bool isSetMask = true>                        \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType* dst, __ubuf__ srcType* src,     \
        const uint64_t mask[2], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)             \
        {                                                                                               \
            ASCENDC_ASSERT((false), {                                                                   \
                KERNEL_LOG(KERNEL_ERROR,                                                                \
                    "rndStr from srcType to dstType not supported!");                                   \
            });                                                                                         \
        }

#define REGISTER_CAST_NOT_SUPPORTED(rndStr, srcType, dstType)                                          \
    REGISTER_CAST_LV2_NOT_SUPPORTED(rndStr, srcType, dstType);                                         \
    REGISTER_CAST_COUNTER_NOT_SUPPORTED(rndStr, srcType, dstType);                                     \
    REGISTER_CAST_BIT_NOT_SUPPORTED(rndStr, srcType, dstType)


// ROUND GROUP 0
// support CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC, CAST_ODD
#define REGISTER_CAST_ROUND_GROUP0(sizeMode, srcType, dstType,                                    \
        srcBits, dstBits, satMode, mode)                                                          \
    REGISTER_CAST_##sizeMode(CastRound, RoundMode::CAST_ROUND, srcType, dstType,                  \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_##sizeMode(CastRint, RoundMode::CAST_RINT, srcType, dstType,                    \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_##sizeMode(CastFloor, RoundMode::CAST_FLOOR, srcType, dstType,                  \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_##sizeMode(CastTrunc, RoundMode::CAST_TRUNC, srcType, dstType,                  \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_##sizeMode(CastCeil, RoundMode::CAST_CEIL, srcType, dstType,                    \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_##sizeMode(CastNone, RoundMode::CAST_RINT, srcType, dstType,                    \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_##sizeMode(CastOdd, RoundMode::CAST_ODD, srcType, dstType,                      \
        srcBits, dstBits, satMode, mode)

// ROUND GROUP 1
// support CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC
// not support CAST_ODD
#define REGISTER_CAST_ROUND_GROUP1(sizeMode, srcType, dstType,                                    \
        srcBits, dstBits, satMode, mode)                                                          \
    REGISTER_CAST_##sizeMode(CastRound, RoundMode::CAST_ROUND, srcType, dstType,                  \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_##sizeMode(CastRint, RoundMode::CAST_RINT, srcType, dstType,                    \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_##sizeMode(CastFloor, RoundMode::CAST_FLOOR, srcType, dstType,                  \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_##sizeMode(CastTrunc, RoundMode::CAST_TRUNC, srcType, dstType,                  \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_##sizeMode(CastCeil, RoundMode::CAST_CEIL, srcType, dstType,                    \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_##sizeMode(CastNone, RoundMode::CAST_RINT, srcType, dstType,                    \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_NOT_SUPPORTED(CastOdd, srcType, dstType)


// ROUND GROUP 2
// support CAST_NONE
#define REGISTER_CAST_ROUND_GROUP2(sizeMode, srcType, dstType,                                    \
        srcBits, dstBits, satMode, mode)                                                          \
    REGISTER_CAST_NOT_SUPPORTED(CastRound, srcType, dstType);                                     \
    REGISTER_CAST_NOT_SUPPORTED(CastRint, srcType, dstType);                                      \
    REGISTER_CAST_NOT_SUPPORTED(CastFloor, srcType, dstType);                                     \
    REGISTER_CAST_NOT_SUPPORTED(CastTrunc, srcType, dstType);                                     \
    REGISTER_CAST_NOT_SUPPORTED(CastCeil, srcType, dstType);                                      \
    REGISTER_CAST_##sizeMode(CastNone, RoundMode::CAST_NONE, srcType, dstType,                    \
        srcBits, dstBits, satMode, mode);                                                         \
    REGISTER_CAST_NOT_SUPPORTED(CastOdd, srcType, dstType)


REGISTER_CAST_ROUND_GROUP1(UPPER_HALF, half, int32_t, 16, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP0(LOWER_HALF, float, half, 32, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(LOWER_HALF, float, int16_t, 32, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(LOWER_HALF, half, int8_t, 16, 8, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(LOWER_HALF, half, uint8_t, 16, 8, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(EQUAL, half, int16_t, 16, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(EQUAL, float, int32_t, 32, 32, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, half, float, 16, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, uint8_t, half, 8, 16, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, int8_t, half, 8, 16, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, int16_t, float, 16, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, uint8_t, uint16_t, 8, 16, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, int8_t, int16_t, 8, 16, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, uint16_t, uint32_t, 16, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, int16_t, uint32_t, 16, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, int16_t, int32_t, 16, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_QUATER, int8_t, int32_t, 8, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_QUATER, uint8_t, uint32_t, 8, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_HALF, uint16_t, uint8_t, 16, 8, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_HALF, int16_t, uint8_t, 16, 8, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_HALF, uint32_t, uint16_t, 32, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_HALF, uint32_t, int16_t, 32, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_HALF, int32_t, uint16_t, 32, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_HALF, int32_t, int16_t, 32, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_QUATER, int32_t, uint8_t, 32, 8, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_QUATER, uint32_t, uint8_t, 32, 8, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(EQUAL, int16_t, half, 16, 16, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(EQUAL, int32_t, float, 32, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(TRUNCATE, half, half, 16, 16, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(TRUNCATE, float, float, 32, 32, SatMode::UNKNOWN, Mode::ZEROING);

// Cast::Level 2
template <typename T, typename U>
__aicore__ inline void CastImpl(__ubuf__ T* dst, __ubuf__ U* src, const RoundMode& roundMode,
    const uint32_t count)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            CastIntrinsicsImplCastRint(dst, src, count);
            break;
        case RoundMode::CAST_FLOOR:
            CastIntrinsicsImplCastFloor(dst, src, count);
            break;
        case RoundMode::CAST_CEIL:
            CastIntrinsicsImplCastCeil(dst, src, count);
            break;
        case RoundMode::CAST_ROUND:
            CastIntrinsicsImplCastRound(dst, src, count);
            break;
        case RoundMode::CAST_TRUNC:
            CastIntrinsicsImplCastTrunc(dst, src, count);
            break;
        case RoundMode::CAST_ODD:
            CastIntrinsicsImplCastOdd(dst, src, count);
            break;
        case RoundMode::CAST_NONE:
            CastIntrinsicsImplCastNone(dst, src, count);
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

// Cast::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void CastImpl(__ubuf__ T* dst, __ubuf__ U* src, const RoundMode& roundMode,
    const uint64_t mask[2], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_NONE:
            CastIntrinsicsImplCastNone<T, U, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            break;
        case RoundMode::CAST_ROUND:
            CastIntrinsicsImplCastRound<T, U, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            break;
        case RoundMode::CAST_RINT:
            CastIntrinsicsImplCastRint<T, U, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            break;
        case RoundMode::CAST_FLOOR:
            CastIntrinsicsImplCastFloor<T, U, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            break;
        case RoundMode::CAST_TRUNC:
            CastIntrinsicsImplCastTrunc<T, U, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            break;
        case RoundMode::CAST_ODD:
            CastIntrinsicsImplCastOdd<T, U, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            break;
        case RoundMode::CAST_CEIL:
            CastIntrinsicsImplCastCeil<T, U, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

// Cast::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void CastImpl(__ubuf__ T* dst, __ubuf__ U* src, const RoundMode& roundMode,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            CastIntrinsicsImplCastRint(dst, src, mask, repeatTime, repeatParams);
            break;
        case RoundMode::CAST_FLOOR:
            CastIntrinsicsImplCastFloor(dst, src, mask, repeatTime, repeatParams);
            break;
        case RoundMode::CAST_CEIL:
            CastIntrinsicsImplCastCeil(dst, src, mask, repeatTime, repeatParams);
            break;
        case RoundMode::CAST_ROUND:
            CastIntrinsicsImplCastRound(dst, src, mask, repeatTime, repeatParams);
            break;
        case RoundMode::CAST_TRUNC:
            CastIntrinsicsImplCastTrunc(dst, src, mask, repeatTime, repeatParams);
            break;
        case RoundMode::CAST_ODD:
            CastIntrinsicsImplCastOdd(dst, src, mask, repeatTime, repeatParams);
            break;
        case RoundMode::CAST_NONE:
            CastIntrinsicsImplCastNone(dst, src, mask, repeatTime, repeatParams);
            break;
        default:
            ASCENDC_ASSERT((false),
                { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

template <typename T, typename U, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ T* dst, __ubuf__ U* src,
    const uint32_t count)
{
    ASCENDC_ASSERT((false), "CastDeq is not supported");
}

template <typename T, typename U, bool isSetMask = true, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ T* dst, __ubuf__ U* src,
    const uint64_t mask[2], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT((false), "CastDeq is not supported");
}

template <typename T, typename U, bool isSetMask = true, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ T* dst, __ubuf__ U* src,
    const int32_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT((false), "CastDeq is not supported");
}

// AddReluCast::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AddReluCastImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    const uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT((false), "AddReluCast is not supported");
}

// AddReluCast::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AddReluCastImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    const uint64_t mask[2], uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT((false), "AddReluCast is not supported");
}

// AddReluCast::Level 2
template <typename T, typename U>
__aicore__ inline void AddReluCastImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    const uint32_t count)
{
    ASCENDC_ASSERT((false), "AddReluCast is not supported");
}

// SubReluCast::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void SubReluCastImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    const uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT((false), "SubReluCast is not supported");
}

// SubReluCast::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void SubReluCastImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    const uint64_t mask[2], uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT((false), "SubReluCast is not supported");
}

// SubReluCast::Level 2
template <typename T, typename U>
__aicore__ inline void SubReluCastImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    const uint32_t count)
{
    ASCENDC_ASSERT((false), "SubReluCast is not supported");
}

__aicore__ inline void SetDeqScaleImpl(float scale, int16_t offset, bool signMode)
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported");
}

template <typename T>
__aicore__ inline void SetDeqScaleImpl(const LocalTensor<T>& vdeq, const VdeqInfo& vdeqInfo)
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported");
}

template<typename T>
__aicore__ inline void SetDeqScaleImpl(T config)
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
