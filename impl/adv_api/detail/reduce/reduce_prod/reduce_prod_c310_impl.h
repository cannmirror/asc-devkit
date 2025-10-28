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
 
#ifndef IMPL_REDUCE_REDUCE_PROD_REDUCE_PROD_C310_IMPL_H_
#define IMPL_REDUCE_REDUCE_PROD_REDUCE_PROD_C310_IMPL_H_

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "../reduce_common_util_impl.h"
#include "../reduce_common_util_c310_impl.h"
#include "../reduce_common_ar_reuse_align_c310_impl.h"
#include "../reduce_common_ra_reuse_align_c310_impl.h"
#include "../../api_check/kernel_api_check.h"
#include "../../common/check.h"

namespace AscendC {
namespace Internal {
template <typename T>
__aicore__ inline void ReduceProd(MicroAPI::RegTensor<T>& dst, MicroAPI::RegTensor<T> src, MicroAPI::MaskReg mask)
{
    MicroAPI::RegTensor<T> tempLow;
    MicroAPI::Duplicate(tempLow, 1);
    MicroAPI::Select(tempLow, src, tempLow, mask);

    MicroAPI::RegTensor<T> tempHigh;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

    if constexpr(sizeof(T) == 1) {
        MicroAPI::MaskReg maskHigh = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL128>();
        MicroAPI::MaskNot(maskHigh, maskHigh, maskAll);
        MicroAPI::GatherMask(tempHigh, tempLow, maskHigh);
        MicroAPI::Mul(tempLow, tempLow, tempHigh, maskAll);
    }

    if constexpr(sizeof(T) <= 2) {
        MicroAPI::MaskReg maskHigh = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL64>();
        MicroAPI::MaskNot(maskHigh, maskHigh, maskAll);
        MicroAPI::GatherMask(tempHigh, tempLow, maskHigh);
        MicroAPI::Mul(tempLow, tempLow, tempHigh, maskAll);
    }

    MicroAPI::MaskReg maskHigh32 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL32>();
    MicroAPI::MaskNot(maskHigh32, maskHigh32, maskAll);
    MicroAPI::GatherMask(tempHigh, tempLow, maskHigh32);
    MicroAPI::Mul(tempLow, tempLow, tempHigh, maskAll);

    MicroAPI::MaskReg maskHigh16 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL16>();
    MicroAPI::MaskNot(maskHigh16, maskHigh16, maskAll);
    MicroAPI::GatherMask(tempHigh, tempLow, maskHigh16);
    MicroAPI::Mul(tempLow, tempLow, tempHigh, maskAll);

    MicroAPI::MaskReg maskHigh8 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL8>();
    MicroAPI::MaskNot(maskHigh8, maskHigh8, maskAll);
    MicroAPI::GatherMask(tempHigh, tempLow, maskHigh8);
    MicroAPI::Mul(tempLow, tempLow, tempHigh, maskAll);

    MicroAPI::MaskReg maskHigh4 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL4>();
    MicroAPI::MaskNot(maskHigh4, maskHigh4, maskAll);
    MicroAPI::GatherMask(tempHigh, tempLow, maskHigh4);
    MicroAPI::Mul(tempLow, tempLow, tempHigh, maskAll);

    MicroAPI::MaskReg maskHigh2 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL2>();
    MicroAPI::MaskNot(maskHigh2, maskHigh2, maskAll);
    MicroAPI::GatherMask(tempHigh, tempLow, maskHigh2);
    MicroAPI::Mul(tempLow, tempLow, tempHigh, maskAll);

    MicroAPI::MaskReg maskHigh1 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskNot(maskHigh1, maskHigh1, maskAll);
    MicroAPI::GatherMask(tempHigh, tempLow, maskHigh1);
    MicroAPI::Mul(dst, tempLow, tempHigh, maskAll);
}

template <class T, bool isReuseSource = false>
__aicore__ inline void ReduceProdARImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, __ubuf__ T *tmpAddr,
    uint32_t dimA, uint32_t dimR)
{
    constexpr uint16_t vlSize = GetVecLen() / sizeof(T);
    if (dimR <= vlSize) {
        ReduceARReuseSourceLessThanVL<T, MicroAPI::RegTraitNumOne, vlSize,
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>,
            ReduceProd<T>>(dstAddr, srcAddr, dimA, dimR);
    } else {
        ReduceAROverVLImpl<T, MicroAPI::RegTraitNumOne, vlSize,
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>,
            ReduceProd<T>, isReuseSource>(dstAddr, srcAddr, tmpAddr, dimA, dimR);
    }
}

template <typename T, typename pattern, bool isReuseSource = false>
__aicore__ inline void ReduceProdImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t srcShape[], bool srcInnerPad)
{
    CHECK_FUNC_HIGHLEVEL_API(ReduceProd, (T, pattern), (dst, src, sharedTmpBuffer, srcShape, srcInnerPad, srcShape[1]));

    CheckTensorPos<T>(dst, Hardware::UB, "dst", "VECIN / VECCALC / VECOUT", "ReduceProd");
    CheckTensorPos<T>(src, Hardware::UB, "src", "VECIN / VECCALC / VECOUT", "ReduceProd");
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", 
        "VECIN / VECCALC / VECOUT", "ReduceProd");
    static_assert(SupportType<T, float>(), "ReduceProd only support float data type on current device!");
    static_assert(std::is_same_v<pattern, Pattern::Reduce::AR> || std::is_same_v<pattern, Pattern::Reduce::RA>,
        "ReduceProd only support AR and RA pattern on current device!");

    __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
    __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
    LocalTensor<T> tmpBuf = sharedTmpBuffer.ReinterpretCast<T>();
    __ubuf__ T* tmpAddr = (__ubuf__ T*)tmpBuf.GetPhyAddr();
    if constexpr (std::is_same_v<pattern, Pattern::Reduce::AR>) {
        ReduceProdARImpl<T, isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[0], srcShape[1]);
    } else {
        ReduceRAImpl<T, MicroAPI::RegTraitNumOne,
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>,
            isReuseSource>(dstAddr, srcAddr, tmpAddr, srcShape[1], srcShape[0]);
    }
}
} // namespace Internal
} // namespace AscendC
#endif // IMPL_REDUCE_REDUCE_PROD_REDUCE_PROD_C310_IMPL_H_