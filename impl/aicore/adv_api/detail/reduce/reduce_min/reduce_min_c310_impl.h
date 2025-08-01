/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICORE_ADV_API_DETAIL_REDUCE_REDUCE_MIN_REDUCE_MIN_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_REDUCE_REDUCE_MIN_REDUCE_MIN_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../reduce_common_util_impl.h"
#include "../reduce_common_util_c310_impl.h"
#include "../reduce_common_ar_reuse_align_c310_impl.h"
#include "../reduce_common_ra_reuse_align_c310_impl.h"
#include "../reduce_common_ar_ra_reuse_unalign_c310_impl.h"
#include "../../common/check.h"
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
template <class T>
__aicore__ inline void ReduceMinARB64ReuseSourceCompute(
    __ubuf__ T* dstAddr, __ubuf__ T* srcAddr, const uint32_t srcShape[])
{
    if ((srcShape[1] * sizeof(T)) % 32 == 0) {
        ReduceARReuseSource<T, MicroAPI::RegTraitNumTwo,
            MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>,
            MicroAPI::ReduceMin<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(
            dstAddr, srcAddr, srcShape[0], srcShape[1]);
    } else {
        ReduceARReuseSourceUnAligned<T, MicroAPI::RegTraitNumTwo,
            MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>,
            MicroAPI::ReduceMin<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(
            dstAddr, srcAddr, srcShape[0], srcShape[1]);
    }
}

template <class T>
__aicore__ inline void ReduceMinARReuseSourceCompute(
    __ubuf__ T* dstAddr, __ubuf__ T* srcAddr, const uint32_t srcShape[])
{
    using castType = typename ReduceOpInternal::ExtractReduceCastType<T>::CastT;
    if ((srcShape[1] * sizeof(T)) % 32 == 0) {
        ReduceARReuseSource<T, MicroAPI::RegTraitNumOne,
            MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>,
            MicroAPI::ReduceMin<castType, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<castType>>>(
            dstAddr, srcAddr, srcShape[0], srcShape[1]);
    } else {
        if constexpr (SupportBytes<T, 1>()) {
            constexpr uint32_t ONE_STRIDE = 32768;
            uint16_t repeatTimes = (srcShape[0] + ONE_STRIDE - 1) >> 15;
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                uint32_t dimA = ONE_STRIDE < srcShape[0] - ONE_STRIDE * i ? ONE_STRIDE : srcShape[0] - ONE_STRIDE * i;
                ReduceARReuseSourceUnAligned<T, MicroAPI::RegTraitNumOne,
                    MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>,
                    MicroAPI::ReduceMin<castType, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<castType>>>(
                    dstAddr + ONE_STRIDE * i, srcAddr + ONE_STRIDE * i * srcShape[1], dimA, srcShape[1]);
            }
        } else {
            ReduceARReuseSourceUnAligned<T, MicroAPI::RegTraitNumOne,
                MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>,
                MicroAPI::ReduceMin<castType, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<castType>>>(
                dstAddr, srcAddr, srcShape[0], srcShape[1]);
        }
    }
}

template <class T>
__aicore__ inline void ReduceMinRAB64ReuseSourceCompute(
    __ubuf__ T* dstAddr, __ubuf__ T* srcAddr, const uint32_t srcShape[])
{
    if ((srcShape[1] * sizeof(T)) % 32 == 0) {
        ReduceRAB64ReuseSource<T, MicroAPI::RegTraitNumTwo,
            MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(
            dstAddr, srcAddr, srcShape[1], srcShape[0]);
    } else {
        ReduceRAReuseSourceUnAlignedB64<T, MicroAPI::RegTraitNumTwo,
            MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>>(
            dstAddr, srcAddr, srcShape[1], srcShape[0]);
    }
}

template <class T>
__aicore__ inline void ReduceMinRAReuseSourceCompute(
    __ubuf__ T* dstAddr, __ubuf__ T* srcAddr, const uint32_t srcShape[])
{
    if ((srcShape[1] * sizeof(T)) % 32 == 0) {
        ReduceRAReuseSource<T, MicroAPI::RegTraitNumOne,
            MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
            dstAddr, srcAddr, srcShape[1], srcShape[0]);
    } else {
        ReduceRAReuseSourceUnAligned<T, MicroAPI::RegTraitNumOne,
            MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>>(
            dstAddr, srcAddr, srcShape[1], srcShape[0]);
    }
}

template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMinCompute(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t srcShape[], bool srcInnerPad)
{
    CheckTensorPos<T>(dst, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "ReduceMin");
    CheckTensorPos<T>(src, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "ReduceMin");
    static_assert(SupportType<T, int8_t, uint8_t, int16_t, uint16_t, half, bfloat16_t, int32_t, uint32_t, float,
                      int64_t, uint64_t>(),
        "ReduceMin only support "
        "int8_t/uint8_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float/int64_t/uint64_t data type on current "
        "device!");
    static_assert(std::is_same_v<pattern, Pattern::Reduce::AR> || std::is_same_v<pattern, Pattern::Reduce::RA>,
        "ReduceMin only support AR and RA pattern on current device!");
    static_assert(isReuseSource, "ReduceMin only support isReuseSource true on current device!");
    ASCENDC_ASSERT(
        (!srcInnerPad), { KERNEL_LOG(KERNEL_ERROR, "ReduceMin only support srcInnerPad false on current device!!"); });

    __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
    __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
    if constexpr (std::is_same_v<pattern, Pattern::Reduce::AR>) {
        if constexpr (SupportBytes<T, 8>()) {
            ReduceMinARB64ReuseSourceCompute<T>(dstAddr, srcAddr, srcShape);
        } else {
            ReduceMinARReuseSourceCompute<T>(dstAddr, srcAddr, srcShape);
        }
    } else {
        if constexpr (SupportBytes<T, 8>()) {
            ReduceMinRAB64ReuseSourceCompute<T>(dstAddr, srcAddr, srcShape);
        } else {
            ReduceMinRAReuseSourceCompute<T>(dstAddr, srcAddr, srcShape);
        }
    }
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_REDUCE_REDUCE_MIN_REDUCE_MIN_C310_IMPL_H