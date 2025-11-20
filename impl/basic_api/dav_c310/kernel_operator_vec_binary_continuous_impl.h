/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {
namespace CastParam {
constexpr MicroAPI::CastTrait s322floatCastTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait float2halfCastTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                     MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait mulAddDstTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
}

template <typename T, typename RegType, auto func>
__simd_vf__ inline void BinaryContinuousImplTemplate(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    const int32_t calCount)
{
    RegType src0Reg;
    RegType src1Reg;
    RegType dstReg;
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg mask;
    constexpr uint32_t repeatStride = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T) * RegType::trait.REG_NUM);
    uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, repeatStride));
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<T, RegType::trait>(sreg);
        MicroAPI::DataCopy(src0Reg, src0 + i * repeatStride);
        MicroAPI::DataCopy(src1Reg, src1 + i * repeatStride);
        func(dstReg, src0Reg, src1Reg, mask);
        MicroAPI::DataCopy(dst + i * repeatStride, dstReg, mask);
    }
}

template <typename T, typename U, typename RegTypeT, typename RegTypeU, auto func>
__simd_vf__ inline void BinaryContinuousImplTemplate(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ U* src1,
    const int32_t calCount)
{
    RegTypeT src0Reg;
    RegTypeU src1Reg;
    RegTypeT dstReg;
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg mask;
    constexpr uint32_t repeatStride = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T) * RegTypeT::trait.REG_NUM);
    uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, repeatStride));
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<T, RegTypeT::trait>(sreg);
        MicroAPI::DataCopy(src0Reg, src0 + i * repeatStride);
        MicroAPI::DataCopy(src1Reg, src1 + i * repeatStride);
        func(dstReg, src0Reg, src1Reg, mask);
        MicroAPI::DataCopy(dst + i * repeatStride, dstReg, mask);
    }
}

template <typename T>
__aicore__ inline void AddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, uint16_t, int16_t, bfloat16_t,
        uint32_t, int32_t, float, int64_t, uint64_t, complex32, complex64>()),
        "current data type is not supported on current device!");
    if constexpr (SupportBytes<T, 8>() || SupportType<T, complex32>()) {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(dst, src0, src1, calCount);
    } else {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T>,
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T>>>(dst, src0, src1, calCount);
    }
}

template <typename T>
__aicore__ inline void SubImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, uint16_t, int16_t, bfloat16_t,
        uint32_t, int32_t, float, int64_t, uint64_t, complex32, complex64>()),
        "current data type is not supported on current device!");
    if constexpr (SupportBytes<T, 8>() || SupportType<T, complex32>()) {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
            MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(dst, src0, src1, calCount);
    } else {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T>,
            MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T>>>(dst, src0, src1, calCount);
    }
}

/* **************************************************************************************************
 * Mul                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void MulImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float,
        int64_t, uint64_t, complex32, complex64>()),
        "current data type is not supported on current device!");
    if constexpr (SupportBytes<T, 8>() || SupportType<T, complex32>()) {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(dst, src0, src1, calCount);
    } else {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T>,
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T>>>(dst, src0, src1, calCount);
    }
}

/* **************************************************************************************************
 * Div                                             *
 * ************************************************************************************************* */
// Div::Level 2
template <typename T, const DivConfig& config = DEFAULT_DIV_CONFIG>
__aicore__ inline void DivImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    static_assert((SupportType<T, uint16_t, int16_t, uint32_t, int32_t, half, float,
        int64_t, uint64_t, complex32, complex64>()),
        "current data type is not supported on current device!");
    if constexpr (config.algo == DivAlgo::INTRINSIC || config.algo == DivAlgo::PRECISION_1ULP_FTZ_TRUE) {
        if constexpr (SupportBytes<T, 8>() || SupportType<T, complex32>()) {
            BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
                MicroAPI::Div<T, MicroAPI::MaskMergeMode::ZEROING,
                MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(dst, src0, src1, calCount);
        } else {
            BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T>,
                MicroAPI::Div<T, MicroAPI::MaskMergeMode::ZEROING,
                MicroAPI::RegTensor<T>>>(dst, src0, src1, calCount);
        }
    } else if constexpr (config.algo == DivAlgo::DIFF_COMPENSATION || config.algo == DivAlgo::PRECISION_0ULP_FTZ_TRUE) {
        static constexpr MicroAPI::DivSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, true, DivAlgo::PRECISION_0ULP_FTZ_TRUE };
        if constexpr (SupportBytes<T, 8>()) {
            constexpr auto func = MicroAPI::Div<T, &mode, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
            BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, func>(dst, src0, src1, calCount);
        } else {
            constexpr auto func = MicroAPI::Div<T, &mode, MicroAPI::RegTensor<T>>;
            BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T>, func>(dst, src0, src1, calCount);
        }
    } else if constexpr (config.algo == DivAlgo::PRECISION_0ULP_FTZ_FALSE) {
        static constexpr MicroAPI::DivSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, false, DivAlgo::PRECISION_0ULP_FTZ_FALSE };
        if constexpr (SupportBytes<T, 8>()) {
            constexpr auto func = MicroAPI::Div<T, &mode, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
            BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, func>(dst, src0, src1, calCount);
        } else {
            constexpr auto func = MicroAPI::Div<T, &mode, MicroAPI::RegTensor<T>>;
            BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T>, func>(dst, src0, src1, calCount);
        }
    } else if constexpr (config.algo == DivAlgo::PRECISION_1ULP_FTZ_FALSE) {
        static constexpr MicroAPI::DivSpecificMode mode = { MicroAPI::MaskMergeMode::ZEROING, false, DivAlgo::PRECISION_1ULP_FTZ_FALSE };
        if constexpr (SupportBytes<T, 8>()) {
            constexpr auto func = MicroAPI::Div<T, &mode, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
            BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, func>(dst, src0, src1, calCount);
        } else {
            constexpr auto func = MicroAPI::Div<T, &mode, MicroAPI::RegTensor<T>>;
            BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T>, func>(dst, src0, src1, calCount);
        }
    }
}

/* **************************************************************************************************
 * Max                                             *
 * ************************************************************************************************* */
// Max::Level 2
template <typename T>
__aicore__ inline void MaxImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t,
        float, int64_t, uint64_t>()), "current data type is not supported on current device!");
    if constexpr (SupportBytes<T, 8>()) {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
            MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(dst, src0, src1, calCount);
    } else {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T>,
            MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(dst, src0, src1, calCount);
    }
}

/* **************************************************************************************************
 * Min                                             *
 * ************************************************************************************************* */
// Min::Level 2
template <typename T>
__aicore__ inline void MinImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t,
        float, int64_t, uint64_t>()), "current data type is not supported on current device!");
    if constexpr (SupportBytes<T, 8>()) {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
            MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(dst, src0, src1, calCount);
    } else {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T>,
            MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(dst, src0, src1, calCount);
    }
}

/* **************************************************************************************************
 * And                                             *
 * ************************************************************************************************* */
// And::Level 2
template <typename T>
__aicore__ inline void AndImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, int16_t, uint16_t, uint32_t, int32_t, int64_t, uint64_t>()),
        "current data type is not supported on current device!");
    if constexpr (SupportBytes<T, 8>()) {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
            MicroAPI::And<T, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(dst, src0, src1, calCount);
    } else {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T>,
            MicroAPI::And<T, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T>>>(dst, src0, src1, calCount);
    }
}


/* **************************************************************************************************
 * Or                                             *
 * ************************************************************************************************* */
// Or::Level 2
template <typename T>
__aicore__ inline void OrImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, int16_t, uint16_t, uint32_t, int32_t, int64_t, uint64_t>()),
        "current data type is not supported on current device!");
    if constexpr (SupportBytes<T, 8>()) {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
            MicroAPI::Or<T, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(dst, src0, src1, calCount);
    } else {
        BinaryContinuousImplTemplate<T, MicroAPI::RegTensor<T>,
            MicroAPI::Or<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(dst, src0, src1, calCount);
    }
}


// AddRelu::Level 2
template <typename T>
__simd_vf__ inline void AddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t calCount)
{
    static_assert(SupportType<T, half, float, int16_t, uint64_t, int64_t>(), "Failed to check dtype in AddRelu, "
        "current api support dtype combination is src and dst both: half / float / int16_t/ uint64_t/ int64_t.");
    const T scalarValue = 0;
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t sregLower = static_cast<uint32_t>(B64_DATA_NUM_PER_REPEAT * 2);
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vDstReg;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vSrcReg0;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vSrcReg1;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::DataCopy(vSrcReg0, src0 + i * sregLower);
            MicroAPI::DataCopy(vSrcReg1, src1 + i * sregLower);
            MicroAPI::Add(vDstReg, vSrcReg0, vSrcReg1, mask);
            MicroAPI::Maxs(vDstReg, vDstReg, scalarValue, mask);
            MicroAPI::DataCopy(dst + i * sregLower, vDstReg, mask);
        }
    } else {
        constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
        MicroAPI::RegTensor<T> dstReg;
        MicroAPI::RegTensor<T> src0Reg;
        MicroAPI::RegTensor<T> src1Reg;
        MicroAPI::MaskReg preg;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            preg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy<T>(src0Reg, src0 + i * sregLower);
            MicroAPI::DataCopy<T>(src1Reg, src1 + i * sregLower);
            MicroAPI::Add<T>(dstReg, src0Reg, src1Reg, preg);
            MicroAPI::Maxs<T>(dstReg, dstReg, scalarValue, preg);
            MicroAPI::DataCopy<T>(dst + i * sregLower, dstReg, preg);
        }
    }
}
/* **************************************************************************************************
 * ShiftLeft                                             *
 * ************************************************************************************************* */
// ShiftLeft::Level 2
template <typename T, typename U>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ U *src1, const int32_t& calCount)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<int64_t, int64_t>, Tuple<uint64_t, int64_t>, Tuple<int32_t, int32_t>,
        Tuple<uint32_t, int32_t>, Tuple<int16_t, int16_t>, Tuple<uint16_t, int16_t>, Tuple<int8_t, int8_t>,
        Tuple<uint8_t, int8_t>>(),
        "Failed to check dtype in ShiftLeft, current api support dtype combination is src0: int64_t, src1: int64_t; "
        "src0: uint64_t, src1: int64_t; src0: int32_t, src1: int32_t; src0: uint32_t, src1: int32_t; src0: int16_t, "
        "src1: int16_t; src0: uint16_t, src1: int16_t; "
        "src0: int8_t, src1: int8_t; src0: uint8_t, src1: int8_t.");
    if constexpr (SupportBytes<T, 8>()) {
        BinaryContinuousImplTemplate<T, U, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
            MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo>,
            MicroAPI::ShiftLeft<T, U, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo>>>(
            dst, src0, src1, calCount);
    } else {
        BinaryContinuousImplTemplate<T, U, MicroAPI::RegTensor<T>, MicroAPI::RegTensor<U>,
            MicroAPI::ShiftLeft<T, U, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>,
            MicroAPI::RegTensor<U>>>(dst, src0, src1, calCount);
    }
}

/* **************************************************************************************************
 * ShiftRight                                             *
 * ************************************************************************************************* */
// ShiftRight::Level 2
template <typename T, typename U>
__aicore__ inline void ShiftRightImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ U *src1, const int32_t& calCount)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<int64_t, int64_t>, Tuple<uint64_t, int64_t>, Tuple<int32_t, int32_t>,
        Tuple<uint32_t, int32_t>, Tuple<int16_t, int16_t>, Tuple<uint16_t, int16_t>, Tuple<int8_t, int8_t>,
        Tuple<uint8_t, int8_t>>(),
        "Failed to check dtype in ShiftRight, current api support dtype combination is src0: int64_t, src1: int64_t; "
        "src0: uint64_t, src1: int64_t; src0: int32_t, src1: int32_t; src0: uint32_t, src1: int32_t; src0: int16_t, "
        "src1: int16_t; src0: uint16_t, src1: int16_t; "
        "src0: int8_t, src1: int8_t; src0: uint8_t, src1: int8_t.");
    if constexpr (SupportBytes<T, 8>()) {
        BinaryContinuousImplTemplate<T, U, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
            MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo>,
            MicroAPI::ShiftRight<T, U, MicroAPI::MaskMergeMode::ZEROING,
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo>>>(
            dst, src0, src1, calCount);
    } else {
        BinaryContinuousImplTemplate<T, U, MicroAPI::RegTensor<T>, MicroAPI::RegTensor<U>,
            MicroAPI::ShiftRight<T, U, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>,
            MicroAPI::RegTensor<U>>>(dst, src0, src1, calCount);
    }
}

/* **************************************************************************************************
 * FusedMulAdd                                             *
 * ************************************************************************************************* */
template <typename T, bool isSetMask = true>
__simd_vf__ inline void FusedMulAddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t calCount)
{
    static_assert(SupportType<T, half, float, bfloat16_t, uint64_t, int64_t>(), "Failed to check dtype in FusedMulAdd,"
        "current api support dtype combination is src and dst both: half / float / bfloat16_t / uint64_t / int64_t.");
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t sregLower = static_cast<uint32_t>(B64_DATA_NUM_PER_REPEAT * 2);
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vDstReg0;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vDstReg1;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vSrcReg0;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vSrcReg1;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::DataCopy(vSrcReg0, src0 + i * sregLower);
            MicroAPI::DataCopy(vSrcReg1, src1 + i * sregLower);
            MicroAPI::DataCopy(vDstReg0, dst + i * sregLower);
            MicroAPI::Mul(vDstReg1, vSrcReg0, vDstReg0, mask);
            MicroAPI::Add(vDstReg0, vDstReg1, vSrcReg1, mask);
            MicroAPI::DataCopy(dst + i * sregLower, vDstReg0, mask);
        }
    } else {
        constexpr uint32_t repeatStride = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, repeatStride));
        MicroAPI::RegTensor<T> src0Reg;
        MicroAPI::RegTensor<T> src1Reg;
        MicroAPI::RegTensor<T> dstReg;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy(src0Reg, src0 + i * repeatStride);
            MicroAPI::DataCopy(src1Reg, src1 + i * repeatStride);
            MicroAPI::DataCopy(dstReg, dst + i * repeatStride);
            MicroAPI::FusedMulDstAdd(dstReg, src0Reg, src1Reg, mask);
            MicroAPI::DataCopy(dst + i * repeatStride, dstReg, mask);
        }
    }
}

/* **************************************************************************************************
 * FusedMulAddRelu                                             *
 * ************************************************************************************************* */
// FusedMulAddRelu::Level 2
template <typename T>
__simd_vf__ inline void FusedMulAddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t calCount)
{
    static_assert(SupportType<T, half, float, uint64_t, int64_t>(), "Failed to check dtype in FusedMulAddRelu, current "
        "api support dtype combination is src and dst both: half / float / uint64_t / int64_t.");
    const T scalarValue = 0;
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t sregLower = static_cast<uint32_t>(B64_DATA_NUM_PER_REPEAT * 2);
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vDstReg0;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vDstReg1;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vSrcReg0;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vSrcReg1;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::DataCopy(vSrcReg0, src0 + i * sregLower);
            MicroAPI::DataCopy(vSrcReg1, src1 + i * sregLower);
            MicroAPI::DataCopy(vDstReg0, dst + i * sregLower);
            MicroAPI::Mul(vDstReg1, vSrcReg0, vDstReg0, mask);
            MicroAPI::Add(vDstReg0, vDstReg1, vSrcReg1, mask);
            MicroAPI::Maxs(vDstReg0, vDstReg0, scalarValue, mask);
            MicroAPI::DataCopy(dst + i * sregLower, vDstReg0, mask);
        }
    } else {
        const uint32_t repeatStride = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, repeatStride));
        MicroAPI::RegTensor<T> src0Reg;
        MicroAPI::RegTensor<T> src1Reg;
        MicroAPI::RegTensor<T> dstReg;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy(src0Reg, src0 + i * repeatStride);
            MicroAPI::DataCopy(src1Reg, src1 + i * repeatStride);
            MicroAPI::DataCopy(dstReg, dst + i * repeatStride);
            MicroAPI::FusedMulDstAdd(dstReg, src0Reg, src1Reg, mask);
            MicroAPI::Maxs(dstReg, dstReg, scalarValue, mask);
            MicroAPI::DataCopy(dst + i * repeatStride, dstReg, mask);
        }
    }
}
/* **************************************************************************************************
 * MulAddDst                                             *
 * ************************************************************************************************* */
// MulAddDst::Level 2
template <typename T, typename U>
__simd_vf__ inline void MulAddDstImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, const int32_t calCount)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<float, half>,
        Tuple<uint64_t, uint64_t>, Tuple<int64_t, int64_t>>(), "Failed to check dtype in MulAddDst, current api "
        "support dtype combination is src: half, dst: half / float; src: float, dst: float; src: uint64_t, dst: "
        "uint64_t; src: int64_t, dst: int64_t.");
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t sregLower = static_cast<uint32_t>(B64_DATA_NUM_PER_REPEAT * 2);
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vDstReg0;
        MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo> vSrcReg0;
        MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo> vSrcReg1;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::DataCopy(vSrcReg0, src0 + i * sregLower);
            MicroAPI::DataCopy(vSrcReg1, src1 + i * sregLower);
            MicroAPI::DataCopy(vDstReg0, dst + i * sregLower);
            MicroAPI::MulAddDst(vDstReg0, vSrcReg0, vSrcReg1, mask);
            MicroAPI::DataCopy(dst + i * sregLower, vDstReg0, mask);
        }
    } else {
        constexpr uint16_t numPerRep = VECTOR_REG_WIDTH / sizeof(T);
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, numPerRep));
        MicroAPI::RegTensor<U> src0Reg, src1Reg;
        MicroAPI::RegTensor<T> dstReg;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy(src0Reg, src0 + i * numPerRep);
            MicroAPI::DataCopy(src1Reg, src1 + i * numPerRep);
            MicroAPI::DataCopy(dstReg, dst + i * numPerRep);
            MicroAPI::MulAddDst(dstReg, src0Reg, src1Reg, mask);
            MicroAPI::DataCopy(dst + i * numPerRep, dstReg, mask);
        }
    }
}

__simd_vf__ inline void MulAddDstImpl(__ubuf__ float* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    const int32_t calCount)
{
    uint32_t sregB32 = static_cast<uint32_t>(calCount);     // updated when float calculation
    constexpr uint16_t numPerRep = VECTOR_REG_WIDTH / sizeof(float);     // each repeat 64 half->float to calculate
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, numPerRep));
    MicroAPI::RegTensor<half> src0Reg, src1Reg;
    MicroAPI::RegTensor<float> dstReg, castReg1, castReg2;
    MicroAPI::MaskReg maskB32;                              // updated when float calculation
    for (uint16_t i = 0; i < repeatTime; ++i) {
        maskB32 = MicroAPI::UpdateMask<float>(sregB32);
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(src0Reg, src0 + i * numPerRep); // 64 half
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(src1Reg, src1 + i * numPerRep); // 64 half
        MicroAPI::Cast<float, half, CastParam::mulAddDstTrait>(castReg1, src0Reg, maskB32);           // 64 float
        MicroAPI::Cast<float, half, CastParam::mulAddDstTrait>(castReg2, src1Reg, maskB32);           // 64 float
        MicroAPI::DataCopy(dstReg, dst + i * numPerRep);
        MicroAPI::MulAddDst(dstReg, castReg1, castReg2, maskB32);
        MicroAPI::DataCopy(dst + i * numPerRep, dstReg, maskB32);
    }
}

/* **************************************************************************************************
 * SubRelu                                             *
 * ************************************************************************************************* */
// SubRelu::Level 2
template <typename T>
__simd_vf__ inline void SubReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t calCount)
{
    static_assert(SupportType<T, half, float, int16_t, uint64_t, int64_t>(), "Failed to check dtype in SubRelu, "
        "current api support dtype combination is src and dst both: half / float / int16_t / uint64_t / int64_t.");
    uint32_t sreg = static_cast<uint32_t>(calCount);
    const T scalarValue = 0;
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t sregLower = static_cast<uint32_t>(B64_DATA_NUM_PER_REPEAT * 2);
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vDstReg;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vSrcReg0;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> vSrcReg1;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::DataCopy(vSrcReg0, src0 + i * sregLower);
            MicroAPI::DataCopy(vSrcReg1, src1 + i * sregLower);
            MicroAPI::Sub(vDstReg, vSrcReg0, vSrcReg1, mask);
            MicroAPI::Maxs(vDstReg, vDstReg, scalarValue, mask);
            MicroAPI::DataCopy(dst + i * sregLower, vDstReg, mask);
        }
    } else {
        constexpr uint16_t numPerRep = VECTOR_REG_WIDTH / sizeof(T);
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, numPerRep));
        MicroAPI::RegTensor<T> dstReg, src0Reg, src1Reg;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy(src0Reg, src0 + i * numPerRep);
            MicroAPI::DataCopy(src1Reg, src1 + i * numPerRep);
            MicroAPI::Sub(dstReg, src0Reg, src1Reg, mask);
            MicroAPI::Maxs(dstReg, dstReg, scalarValue, mask);
            MicroAPI::DataCopy(dst + i * numPerRep, dstReg, mask);
        }
    }
}

/* **************************************************************************************************
 * AddDeqRelu                                             *
 * ************************************************************************************************* */
// AddDeqRelu::Level 2
__simd_vf__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const int32_t calCount)
{
    const float scalarValue = 0.;
    constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(int32_t));
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    MicroAPI::RegTensor<half> dstReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<int32_t> src0Reg;
    MicroAPI::RegTensor<int32_t> src1Reg;
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg preg;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        preg = MicroAPI::UpdateMask<int32_t>(sreg);
        MicroAPI::DataCopy<int32_t>(src0Reg, src0 + i * sregLower);
        MicroAPI::DataCopy<int32_t>(src1Reg, src1 + i * sregLower);
        MicroAPI::Add<int32_t>(src0Reg, src0Reg, src1Reg, preg);
        MicroAPI::Cast<float, int32_t, CastParam::s322floatCastTrait>(tmpReg, src0Reg, preg);
        MicroAPI::Muls<float>(tmpReg, tmpReg, static_cast<float>(DEQ_SHIFT_RIGHT_17_BIT), preg);
        MicroAPI::Muls<float>(tmpReg, tmpReg, static_cast<float>(g_deqValue), preg);
        MicroAPI::Muls<float>(tmpReg, tmpReg, static_cast<float>(DEQ_SHIFT_LEFT_17_BIT), preg);
        MicroAPI::Maxs<float>(tmpReg, tmpReg, scalarValue, preg);
        MicroAPI::Cast<half, float, CastParam::float2halfCastTrait>(dstReg, tmpReg, preg);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * sregLower, dstReg, preg);
    }
}

/* **************************************************************************************************
 * Prelu                                             *
 * ************************************************************************************************* */
// Prelu::Level 2
template <typename T>
__simd_vf__ inline void PreluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint32_t calCount)
{
    static_assert(SupportType<T, half, float>(),  "Failed to check dtype in Prelu, current api support "
        "dtype is half, float.");
    constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::RegTensor<T> vDstReg0;
    MicroAPI::RegTensor<T> vSrcReg0;
    MicroAPI::RegTensor<T> vSrcReg1;
    MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy(vSrcReg0, src0 + i * sregLower);
        MicroAPI::DataCopy(vSrcReg1, src1 + i * sregLower);
        MicroAPI::Prelu(vDstReg0, vSrcReg0, vSrcReg1, mask);
        MicroAPI::DataCopy(dst + i * sregLower, vDstReg0, mask);
    }
}

/* **************************************************************************************************
 * Mull                                            *
 * ************************************************************************************************* */
// Mull::Level 2
template <typename T>
__simd_vf__ inline void MullImpl(
    __ubuf__ T *dst0, __ubuf__ T *dst1, __ubuf__ T *src0, __ubuf__ T *src1, const uint32_t calCount)
{
    static_assert(SupportType<T, uint32_t, int32_t>(), "Failed to check dtype in Mull, current api support "
        "dtype is uint32_t, int32_t");
    constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::RegTensor<T> vDstReg0;
    MicroAPI::RegTensor<T> vDstReg1;
    MicroAPI::RegTensor<T> vSrcReg0;
    MicroAPI::RegTensor<T> vSrcReg1;
    MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy(vSrcReg0, src0 + i * sregLower);
        MicroAPI::DataCopy(vSrcReg1, src1 + i * sregLower);
        MicroAPI::Mull(vDstReg0, vDstReg1, vSrcReg0, vSrcReg1, mask);
        MicroAPI::DataCopy(dst0 + i * sregLower, vDstReg0, mask);
        MicroAPI::DataCopy(dst1 + i * sregLower, vDstReg1, mask);
    }
}


/* **************************************************************************************************
 * FusedAbsSub                                            *
 * ************************************************************************************************* */
// FusedAbsSub::Level 2
template <typename T>
__simd_vf__ inline void FusedAbsSubImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint32_t calCount)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in FusedAbsSub, current api support "
        "dtype is src and dst both: half, float.");
    constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::RegTensor<T> vDstReg0;
    MicroAPI::RegTensor<T> vSrcReg0;
    MicroAPI::RegTensor<T> vSrcReg1;
    MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy(vSrcReg0, src0 + i * sregLower);
        MicroAPI::DataCopy(vSrcReg1, src1 + i * sregLower);
        MicroAPI::FusedAbsSub(vDstReg0, vSrcReg0, vSrcReg1, mask);
        MicroAPI::DataCopy(dst + i * sregLower, vDstReg0, mask);
    }
}

/* **************************************************************************************************
 * FusedExpSub                                        *
 * ************************************************************************************************* */
// FusedExpSub::Level 2
template <typename T, typename U>
__simd_vf__ inline void FusedExpSubImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1, const uint32_t calCount)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<float, half>, Tuple<float, float>>(), "Failed to check dtype in "
        "FusedExpSub, current api support dtype combination is src : half / float, dst: float.");
    constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(float));
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::RegTensor<T> vDstReg0;
    MicroAPI::RegTensor<U> vSrcReg0;
    MicroAPI::RegTensor<U> vSrcReg1;
    MicroAPI::MaskReg mask;
    if constexpr (IsSameType<U, half>::value) {
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(vSrcReg0, src0 + i * sregLower);
            MicroAPI::DataCopy<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(vSrcReg1, src1 + i * sregLower);
            MicroAPI::FusedExpSub(vDstReg0, vSrcReg0, vSrcReg1, mask);
            MicroAPI::DataCopy(dst + i * sregLower, vDstReg0, mask);
        }
    } else if constexpr (IsSameType<U, float>::value) {
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy(vSrcReg0, src0 + i * sregLower);
            MicroAPI::DataCopy(vSrcReg1, src1 + i * sregLower);
            MicroAPI::FusedExpSub(vDstReg0, vSrcReg0, vSrcReg1, mask);
            MicroAPI::DataCopy(dst + i * sregLower, vDstReg0, mask);
        }
    }
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
