/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sort_common_utils.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_SORT_SORT_COMMON_UTILS_H
#define AICORE_ADV_API_DETAIL_SORT_SORT_COMMON_UTILS_H

namespace AscendC {
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
namespace Internal {
template <uint32_t size = sizeof(uint8_t)>
struct ExtractTypeBySize {
    using T = uint8_t;
};

template <>
struct ExtractTypeBySize<sizeof(uint16_t)> {
    using T = uint16_t;
};

template <>
struct ExtractTypeBySize<sizeof(uint32_t)> {
    using T = uint32_t;
};

template <>
struct ExtractTypeBySize<sizeof(uint64_t)> {
    using T = uint64_t;
};
} // namespace Internal
#endif

namespace MicroAPI {
namespace Internal {
template <typename T, typename U>
__aicore__ inline void TwiddleIn(RegTensor<U>& dst, RegTensor<U>& src, MaskReg& maskReg)
{
    RegTensor<U> highBit;
    Duplicate(highBit, 1ul << (sizeof(T) * 8 - 1), maskReg);
    if constexpr (SupportType<T, float, bfloat16_t, half>()) {
        RegTensor<U> retAnd;
        RegTensor<U> selSrc;
        Duplicate(selSrc, -1ul, maskReg);
        And(retAnd, src, highBit, maskReg);
        MaskReg cmpMask;
        CompareScalar<U, CMPMODE::NE>(cmpMask, retAnd, 0, maskReg);
        Select(highBit, selSrc, highBit, cmpMask);
    }
    Xor(dst, src, highBit, maskReg);
}

template <typename T, typename U>
__aicore__ inline void TwiddleOut(RegTensor<U>& dst, RegTensor<U>& src, MaskReg& maskReg)
{
    RegTensor<U> highBit;
    Duplicate(highBit, 1ul << (sizeof(T) * 8 - 1), maskReg);
    if constexpr (SupportType<T, float, bfloat16_t, half>()) {
        RegTensor<U> retAnd;
        RegTensor<U> selSrc;
        Duplicate(selSrc, -1ul, maskReg);
        And(retAnd, src, highBit, maskReg);
        MaskReg cmpMask;
        CompareScalar<U, CMPMODE::EQ>(cmpMask, retAnd, 0, maskReg);
        Select(highBit, selSrc, highBit, cmpMask);
    }
    Xor(dst, src, highBit, maskReg);
}

} // namespace Internal
} // namespace MicroAPI

} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_SORT_SORT_COMMON_UTILS_H