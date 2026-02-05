/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H

#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_operator_vec_binary_continuous_impl.h"
#include "kernel_operator_vec_template_impl.h"
#include "micro_api/kernel_micro_intf.h"

namespace AscendC {
template <typename T, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true, const DivConfig& config = DEFAULT_DIV_CONFIG>
__aicore__ inline void DivImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, uint16_t, int16_t, uint32_t, int32_t, half, float>()),
        "current data type is not supported on current device!");
    if constexpr (config.algo == DivAlgo::INTRINSIC || config.algo == DivAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPI::Div<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (config.algo == DivAlgo::DIFF_COMPENSATION || config.algo == DivAlgo::PRECISION_0ULP_FTZ_TRUE) {
        static constexpr AscendC::MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, true, DivAlgo::PRECISION_0ULP_FTZ_TRUE};
        constexpr auto func = MicroAPI::Div<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (config.algo == DivAlgo::PRECISION_0ULP_FTZ_FALSE) {
        static constexpr AscendC::MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false, DivAlgo::PRECISION_0ULP_FTZ_FALSE};
        constexpr auto func = MicroAPI::Div<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (config.algo == DivAlgo::PRECISION_1ULP_FTZ_FALSE) {
        static constexpr AscendC::MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false, DivAlgo::PRECISION_1ULP_FTZ_FALSE};
        constexpr auto func = MicroAPI::Div<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t, uint32_t, int32_t>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::And<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void OrImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t, uint32_t, int32_t>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Or<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true, const DivConfig& config = DEFAULT_DIV_CONFIG>
__aicore__ inline void DivImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, uint16_t, int16_t, uint32_t, int32_t, half, float>()),
        "current data type is not supported on current device!");
    if constexpr (config.algo == DivAlgo::INTRINSIC || config.algo == DivAlgo::PRECISION_1ULP_FTZ_TRUE) {
        constexpr auto func = MicroAPI::Div<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (config.algo == DivAlgo::DIFF_COMPENSATION || config.algo == DivAlgo::PRECISION_0ULP_FTZ_TRUE) {
        static constexpr AscendC::MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, true, DivAlgo::PRECISION_0ULP_FTZ_TRUE};
        constexpr auto func = MicroAPI::Div<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (config.algo == DivAlgo::PRECISION_0ULP_FTZ_FALSE) {
        static constexpr AscendC::MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false, DivAlgo::PRECISION_0ULP_FTZ_FALSE};
        constexpr auto func = MicroAPI::Div<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (config.algo == DivAlgo::PRECISION_1ULP_FTZ_FALSE) {
        static constexpr AscendC::MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false, DivAlgo::PRECISION_1ULP_FTZ_FALSE};
        constexpr auto func = MicroAPI::Div<T, &mode, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t, uint32_t, int32_t>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::And<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void OrImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t, uint32_t, int32_t>()),
        "current data type is not supported on current device!");
    constexpr auto func = MicroAPI::Or<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

/* **************************************************************************************************
 * AddRelu                                             *
 * ************************************************************************************************* */
// AddRelu::Level 0
namespace MicroAPIAddRelu {
template <typename T, typename RegT>
__aicore__ inline void AddRelu(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MicroAPI::MaskReg &mask)
{
    MicroAPI::Add(dstReg, srcReg0, srcReg1, mask);
    MicroAPI::Maxs(dstReg, dstReg, (T)0, mask);
}
} // namespace MicroAPIAddRelu

template <typename T, bool isSetMask = true>
__aicore__ inline void AddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t>(), "Failed to check dtype in AddRelu, current api support "
        "dtype combination is src and dst both: half / float / int16_t.");
    constexpr auto func = MicroAPIAddRelu::AddRelu<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t>(), "Failed to check dtype in AddRelu, current api support "
        "dtype combination is src and dst both: half / float / int16_t.");
    constexpr auto func = MicroAPIAddRelu::AddRelu<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

/* **************************************************************************************************
 * FusedMulAdd                                             *
 * ************************************************************************************************* */
// FusedMulAdd::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float, bfloat16_t>(),
        "Failed to check dtype in FusedMulAdd, current api support dtype "
        "combination is src and dst both: half/float/bfloat16_t.");
    constexpr auto func = MicroAPI::FusedMulDstAdd<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true, Internal::BinaryFuncMode::DST_SRC_INPUT>(dst, src0, src1,
        mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float, bfloat16_t>(),
        "Failed to check dtype in FusedMulAdd, current api support dtype "
        "combination is src and dst both: half/float/bfloat16_t.");
    constexpr auto func = MicroAPI::FusedMulDstAdd<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false, Internal::BinaryFuncMode::DST_SRC_INPUT>(dst, src0, src1,
        nullptr, mask, repeatTime, repeatParams);
}

/* **************************************************************************************************
 * FusedMulAddRelu                                             *
 * ************************************************************************************************* */
// FusedMulAddRelu::Level 0
namespace MicroAPIFusedMulAddRelu {
template <typename T, typename RegT>
__aicore__ inline void FusedMulAddRelu(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MicroAPI::MaskReg &mask)
{
    MicroAPI::FusedMulDstAdd(dstReg, srcReg0, srcReg1, mask);
    MicroAPI::Maxs(dstReg, dstReg, (T)0, mask);
}
} // namespace MicroAPIFusedMulAddRelu
template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in FusedMulAddRelu, current api support dtype "
        "combination is src and dst both: half / float.");
    constexpr auto func = MicroAPIFusedMulAddRelu::FusedMulAddRelu<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true, Internal::BinaryFuncMode::DST_SRC_INPUT>(dst, src0, src1,
        mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in FusedMulAddRelu, current api support dtype "
        "combination is src and dst both: half / float.");
    constexpr auto func = MicroAPIFusedMulAddRelu::FusedMulAddRelu<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false, Internal::BinaryFuncMode::DST_SRC_INPUT>(dst, src0, src1,
        nullptr, mask, repeatTime, repeatParams);
}
/* **************************************************************************************************
 * MulAddDst                                             *
 * ************************************************************************************************* */
// MulAddDst::Level 0
namespace MicroAPIMulAddDst {
template <typename T, typename U, typename RegT, typename RegU>
__aicore__ inline void MulAddDst(RegT &dstReg, RegU &srcReg0, RegU &srcReg1, MicroAPI::MaskReg &mask)
{
    if constexpr (std::is_same<T, U>::value) {
        MicroAPI::MulAddDst(dstReg, srcReg0, srcReg1, mask);
    } else {
        MicroAPI::RegTensor<half> fp16RegTemp;
        MicroAPI::RegTensor<float> castReg1, castReg2;
        // the first 64 half is needed to do muladddst in each repeat
        MicroAPI::UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
            (MicroAPI::RegTensor<uint32_t> &)fp16RegTemp, (MicroAPI::RegTensor<uint16_t> &)srcReg0);
        MicroAPI::Cast<float, half, CastParam::mulAddDstTrait>(castReg1, fp16RegTemp, mask);
        MicroAPI::UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
            (MicroAPI::RegTensor<uint32_t> &)fp16RegTemp, (MicroAPI::RegTensor<uint16_t> &)srcReg1);
        MicroAPI::Cast<float, half, CastParam::mulAddDstTrait>(castReg2, fp16RegTemp, mask);
        MicroAPI::MulAddDst(dstReg, castReg1, castReg2, mask);
    }
}
} // namespace MicroAPIMulAddDst

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void MulAddDstImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<float, half>>(), "Failed to "
        "check dtype in MulAddDst, current api support dtype combination is src: half, dst: half / float; src: float, "
        "dst: float.");
    constexpr auto func = MicroAPIMulAddDst::MulAddDst<T, U, MicroAPI::RegTensor<T>, MicroAPI::RegTensor<U>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true, Internal::BinaryFuncMode::DST_SRC_INPUT>(dst, src0, src1,
        mask, 0, repeatTime, repeatParams);
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void MulAddDstImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<float, half>>(), "Failed to "
        "check dtype in MulAddDst, current api support dtype combination is src: half, dst: half / float; src: float, "
        "dst: float.");
    constexpr auto func = MicroAPIMulAddDst::MulAddDst<T, U, MicroAPI::RegTensor<T>, MicroAPI::RegTensor<U>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false, Internal::BinaryFuncMode::DST_SRC_INPUT>(dst, src0, src1,
        nullptr, mask, repeatTime, repeatParams);
}

/* **************************************************************************************************
 * SubRelu                                             *
 * ************************************************************************************************* */
// SubRelu::Level 0
namespace MicroAPISubRelu {
template <typename T, typename RegT>
__aicore__ inline void SubRelu(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MicroAPI::MaskReg &mask)
{
    MicroAPI::Sub(dstReg, srcReg0, srcReg1, mask);
    MicroAPI::Maxs(dstReg, dstReg, (T)0, mask);
}
} // namespace MicroAPISubRelu

// SubRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void SubReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t>(), "Failed to check dtype in SubRelu, current api support dtype "
        "combination is src and dst both: half / float / int16_t.");
    constexpr auto func = MicroAPISubRelu::SubRelu<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t>(), "Failed to check dtype in SubRelu, current api support dtype "
        "combination is src and dst both: half / float / int16_t.");
    constexpr auto func = MicroAPISubRelu::SubRelu<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

/* **************************************************************************************************
 * AddDeqRelu                                             *
 * ************************************************************************************************* */
namespace MicroAPIAddDeqRelu {
template <typename T, typename RegT, typename RegU>
__aicore__ inline void AddDeqRelu(RegU &dstReg, RegT &srcReg0, RegT &srcReg1, MicroAPI::MaskReg &mask)
{
    // max(float(srcReg0 + srcReg1) * (1/131072) * g_deqValue * 131072, 0)
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::Add(srcReg0, srcReg0, srcReg1, mask);
    MicroAPI::Cast<float, int32_t, CastParam::s322floatCastTrait>(tmpReg, srcReg0, mask);
    MicroAPI::Muls(tmpReg, tmpReg, static_cast<float>(DEQ_SHIFT_RIGHT_17_BIT), mask);
    MicroAPI::Muls(tmpReg, tmpReg, static_cast<float>(Internal::g_deqValue), mask);
    MicroAPI::Muls(tmpReg, tmpReg, static_cast<float>(DEQ_SHIFT_LEFT_17_BIT), mask);
    MicroAPI::Maxs(tmpReg, tmpReg, (T)0, mask);
    MicroAPI::Cast<half, float, CastParam::float2halfCastTrait>(dstReg, tmpReg, mask);
    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t> &)dstReg,
        (MicroAPI::RegTensor<uint32_t> &)dstReg);
}
} // namespace MicroAPIAddDeqRelu
template <bool isSetMask = true>
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const uint64_t mask[], const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    constexpr auto func =
        MicroAPIAddDeqRelu::AddDeqRelu<float, MicroAPI::RegTensor<int32_t>, MicroAPI::RegTensor<half>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

template <bool isSetMask = true>
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    constexpr auto func =
        MicroAPIAddDeqRelu::AddDeqRelu<float, MicroAPI::RegTensor<int32_t>, MicroAPI::RegTensor<half>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
