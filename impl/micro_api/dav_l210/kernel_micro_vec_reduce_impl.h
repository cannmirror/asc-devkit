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
 * \file kernel_micro_vec_reduce_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_REDUCE_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_REDUCE_IMPL_H

#include "kernel_micro_common_impl.h"
namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, typename U = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING,
    typename DstRegT, typename SrcRegT>
__aicore__ inline void ReduceSumImpl(DstRegT &dstReg, SrcRegT srcReg, MaskReg mask)
{
    using ActualDstRegT = typename DstRegT::ActualT;
    using ActualSrcRegT = typename SrcRegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualDstRegT>, "T type is not correct!");
    static_assert(std::is_same_v<U, DefaultType> || std::is_same_v<U, ActualSrcRegT>, "U type is not correct!");
    static_assert((SupportType<Tuple<ActualDstRegT, ActualSrcRegT>, Tuple<int16_t, int8_t>, Tuple<uint16_t, uint8_t>, Tuple<int32_t, int16_t>,
        Tuple<uint32_t, uint16_t>, Tuple<uint32_t, uint32_t>,  Tuple<int32_t, int32_t>,
        Tuple<half, half>, Tuple<float, float>>()),
        "ReduceSum unsupport this datatype on current device");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vcadd(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ReduceMaxImpl(RegT &dstReg, RegT srcReg, MaskReg mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert((SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>()),
        "ReduceMax unsupport this datatype on current device");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vcmax(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ReduceMinImpl(RegT &dstReg, RegT srcReg, MaskReg mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert((SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>()),
        "ReduceMin unsupport this datatype on current device");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current ReduceMin api only supported Mode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vcmin(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ReduceSumWithDataBlockImpl(RegT &dstReg, RegT srcReg, MaskReg mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "FusedMulsCast is not supported on this device!"); });
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ReduceMaxWithDataBlockImpl(RegT &dstReg, RegT srcReg, MaskReg mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "FusedMulsCast is not supported on this device!"); });
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ReduceMinWithDataBlockImpl(RegT &dstReg, RegT srcReg, MaskReg mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "FusedMulsCast is not supported on this device!"); });
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void PairReduceSumImpl(RegT &dstReg, RegT srcReg, MaskReg mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "FusedMulsCast is not supported on this device!"); });
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_REDUCE_IMPL_H