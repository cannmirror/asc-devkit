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
 * \file kernel_operator_scalar_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_SCALAR_INTERFACE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SCALAR_INTERFACE_IMPL_H
#include "kernel_tensor.h"
#include "kernel_scalar.h"

namespace AscendC {
template <int countValue>
__aicore__ inline int64_t GetBitCount(uint64_t valueIn)
{
    return GetBitCountImpl<countValue>(valueIn);
}

__aicore__ inline int64_t CountLeadingZero(uint64_t valueIn)
{
    return CountLeadingZeroImpl(valueIn);
}

template <int countValue>
__aicore__ inline int64_t ScalarGetCountOfValue(uint64_t valueIn)
{
    return GetBitCount<countValue>(valueIn);
}

__aicore__ inline int64_t ScalarCountLeadingZero(uint64_t valueIn)
{
    return CountLeadingZero(valueIn);
}

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
template <typename T>
__aicore__ inline void GetUintDivMagicAndShift(T& magic, T& shift, T divisor)
{
    return GetUintDivMagicAndShiftImpl(magic, shift, divisor);
}
#endif

__aicore__ inline int64_t CountBitsCntSameAsSignBit(int64_t valueIn)
{
    return CountBitsCntSameAsSignBitImpl(valueIn);
}

template <int countValue>
__aicore__ inline int64_t GetSFFValue(uint64_t valueIn)
{
    return GetSFFValueImpl<countValue>(valueIn);
}

template <int countValue>
__aicore__ inline int64_t ScalarGetSFFValue(uint64_t valueIn)
{
    return GetSFFValue<countValue>(valueIn);
}

template <typename T, typename U, RoundMode roundMode>
__aicore__ inline U Cast(T valueIn)
{
    return CastImpl<T, U, roundMode>(valueIn);
}

template <typename T, typename U, RoundMode roundMode>
__aicore__ inline U ScalarCast(T valueIn)
{
    return ScalarCastImpl<T, U, roundMode>(valueIn);
}

#if __NPU_ARCH__ == 2201 || (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
template <typename T>
__aicore__ inline void WriteGmByPassDCache(__gm__ T* addr, T value)
{
    return WriteGmByPassDCacheImpl(addr, value);
}

template <typename T>
__aicore__ inline T ReadGmByPassDCache(__gm__ T* addr)
{
    return ReadGmByPassDCacheImpl(addr);
}
#endif
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SCALAR_INTERFACE_IMPL_H
