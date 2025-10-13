/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IMPL_REDUCE_REDUCE_COMMON_UTIL_C310_IMPL_H
#define IMPL_REDUCE_REDUCE_COMMON_UTIL_C310_IMPL_H

#include "kernel_operator_intf.h"
#include "kernel_tensor.h"
#include "reduce_common_util_impl.h"

namespace AscendC {
namespace ReduceOpInternal {
// Calculate the sum of two points based on count. The main block size is ex.2000->1024 900->512.
__aicore__ inline uint32_t CalculateRMainBlock(uint32_t count)
{
    constexpr int32_t TWO_BITS = 2;
    constexpr int32_t FOUR_BITS = 4;
    constexpr int32_t EIGHT_BITS = 8;
    constexpr int32_t SIXTEEN_BITS = 16;
    count |= count >> 1;
    count |= count >> TWO_BITS;
    count |= count >> FOUR_BITS;
    count |= count >> EIGHT_BITS;
    count |= count >> SIXTEEN_BITS;
    return (count + 1) >> 1;
}

constexpr uint16_t BASE_FOLD = 4;
constexpr uint16_t BASE_FOLD_B64 = 3;
constexpr uint16_t FLOD_THREE = 3;
constexpr uint16_t FLOD_TWO = 2;
constexpr uint16_t FLOD_ONE = 1;
constexpr uint32_t U16_STRIDE = 65535;
constexpr uint16_t REGULAR_FOLD_NUM = 2;
constexpr uint16_t NO_REUSE_FOLD_NUM = 2;

static constexpr MicroAPI::CastTrait CastTraitBF16F32 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING };
static constexpr MicroAPI::CastTrait CastTraitF32BF16 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };
static constexpr MicroAPI::CastTrait CastTraitB8F16 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };
static constexpr MicroAPI::CastTrait CastTraitF16B8 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

__aicore__ inline uint32_t CalculateMainR(int64_t dimR, bool isAR, uint16_t vlSize)
{
    constexpr uint16_t MAX_OFFSET = 16;
    int64_t mainR = 1;
    for (uint16_t i = 1; i < MAX_OFFSET; i++) {
        if ((dimR >> i) == 0) {
            break;
        }
        mainR = 1 << i;
    }

    if (isAR && dimR < vlSize) {
        mainR = dimR;
    }
    return static_cast<uint32_t>(mainR);
}

__aicore__ inline uint16_t CalculateFolds(const uint16_t count)
{
    constexpr uint16_t MAX_OFFSET = 16;
    uint16_t folds = 0;
    uint16_t base = count;
    for (uint16_t i = 1; i < MAX_OFFSET; i++) {
        if ((base >> i) == 0) {
            break;
        }
        folds++;
    }
    return folds;
}

template <class T, class U, const MicroAPI::RegTrait &Trait, const MicroAPI::CastTrait &CastTraitUppper,
    const MicroAPI::CastTrait &CastTraitLower, auto Binaryfunc, auto Reducefunc>
__aicore__ inline void ReduceARCastfoldOneToThree(MicroAPI::RegTensor<T, Trait> &vreg0,
    MicroAPI::RegTensor<T, Trait> &vreg1, MicroAPI::MaskReg &fullMask)
{
    MicroAPI::RegTensor<U, Trait> vreg0CastB32;
    MicroAPI::RegTensor<U, Trait> vreg1CastB32;
    MicroAPI::RegTensor<T, Trait> vreg2;
    MicroAPI::RegTensor<T, Trait> vreg3;
    MicroAPI::RegTensor<T, Trait> vreg4;
    Duplicate(vreg4, static_cast<T>(0), fullMask);
    Interleave(vreg2, vreg3, vreg0, vreg4);
    Binaryfunc(vreg0, vreg2, vreg3, fullMask);
    MicroAPI::Cast<U, T, CastTraitUppper>(vreg0CastB32, vreg0, fullMask);
    Reducefunc(vreg1CastB32, vreg0CastB32, fullMask);
    MicroAPI::Cast<T, U, CastTraitLower>(vreg1, vreg1CastB32, fullMask);
}

template <typename T> struct ExtractReduceCastType {
    using CastT = T;
};

template <> struct ExtractReduceCastType<bfloat16_t> {
    using CastT = float;
};

template <> struct ExtractReduceCastType<uint8_t> {
    using CastT = half;
};

template <> struct ExtractReduceCastType<int8_t> {
    using CastT = half;
};

template <uint32_t size = sizeof(uint8_t)> struct ExtractUnsignedTypeBySize {
    using T = uint8_t;
};

template <> struct ExtractUnsignedTypeBySize<sizeof(uint16_t)> {
    using T = uint16_t;
};

template <> struct ExtractUnsignedTypeBySize<sizeof(uint32_t)> {
    using T = uint32_t;
};
} // namespace ReduceOpInternal
} // namespace AscendC
#endif // IMPL_REDUCE_REDUCE_COMMON_UTIL_C310_IMPL_H