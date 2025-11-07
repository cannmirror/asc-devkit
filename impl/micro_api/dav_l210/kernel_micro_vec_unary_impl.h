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
 * \file kernel_micro_vec_unary_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_SINGLE_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_SINGLE_IMPL_H
#include "kernel_micro_common_impl.h"

namespace AscendC {
namespace MicroAPI {

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void AbsImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, int8_t, int16_t, int32_t, half, float>(),
        "current data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Abs api only supported Mode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vabs(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ReluImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, int32_t, half, float>(),
        "current data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Relu api only supported Mode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vrelu(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ExpImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Exp api only supported Mode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vexp(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void SqrtImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Sqrt api only supported Mode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vsqrt(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void RecImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Sqrt api only supported Mode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vrec(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void RsqrtImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Rsqrt api only supported Mode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vrsqrt(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void LogImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Log api only supported Mode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vln(dstReg, srcReg, mask, modeValue);
}

template <MaskMergeMode mode = MaskMergeMode::ZEROING>
__aicore__ inline void LogXImpl(RegTensor<half> &dstReg, RegTensor<half> &srcReg,
    MaskReg &mask, const float lnXReciprocal)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "LogX is not supported on current device!"); });
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void Log2Impl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Log2 is not supported on current device!"); });
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void Log10Impl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Log10 is not supported on current device!"); });
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void NegImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, int8_t, int16_t, int32_t, half, float>(),
        "current data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Neg api only supported Mode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vneg(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void NotImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(
        SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Not api only supported Mode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vnot(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, typename SrcT = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING,
    typename RegT, typename RegSrcT>
__aicore__ inline void CountBitImpl(RegT &dstReg, RegSrcT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    using ActualSrcT = typename RegSrcT::ActualT;
    static_assert(std::is_same_v<SrcT, DefaultType> || std::is_same_v<SrcT, ActualSrcT>, "SrcT type is not correct!");
    static_assert(SupportType<ActualT, int8_t, int16_t, int32_t>(),
        "current data type is not supported on current device!");
    static_assert(SupportType<ActualSrcT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current CountBit api only supported Mode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vbcnt(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void CountLeadingSignBitsImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current CountLeadingSignBits api only supported Mode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vcls(dstReg, srcReg, mask, modeValue);
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_SINGLE_IMPL_H