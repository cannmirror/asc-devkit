/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_operator_vec_duplicate_impl.h
 * \brief AscendC l310 eff support vector duplicate api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
#include <type_traits>
#include "kernel_operator_common_impl.h"

namespace AscendC {
template <typename T> constexpr __aicore__ inline void CheckDuplicateSupportedType()
{
    static_assert(std::is_same<T, half>::value || std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, float>::value,
        "Duplicate instr only support half/int16_t/uint16_t/int32_t/uint32_t/float type in this version");
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dst, const T scalarValue, uint64_t mask,
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported in this version!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ half* dst, const half& scalarValue, uint64_t mask,
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}


template <typename T = float, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ float* dst, const float& scalarValue, uint64_t mask,
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ int16_t* dst, const int16_t& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ uint16_t* dst, const uint16_t& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ int32_t* dst, const int32_t& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ uint32_t* dst, const uint32_t& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dst, const T scalarValue, uint64_t mask[2],
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported in this version!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ half* dst, const half& scalarValue, uint64_t mask[2],
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ float* dst, const float& scalarValue, uint64_t mask[2],
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ int16_t* dst, const int16_t& scalarValue,
    uint64_t mask[2], const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ uint16_t* dst, const uint16_t& scalarValue,
    uint64_t mask[2], const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ int32_t* dst, const int32_t& scalarValue,
    uint64_t mask[2], const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ uint32_t* dst, const uint32_t& scalarValue,
    uint64_t mask[2], const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dst, const T scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported in this version!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ half* dst, const half& scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ float* dst, const float& scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ int16_t* dst, const int16_t& scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ uint16_t* dst, const uint16_t& scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ int32_t* dst, const int32_t& scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ uint32_t* dst, const uint32_t& scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support on this version!"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
