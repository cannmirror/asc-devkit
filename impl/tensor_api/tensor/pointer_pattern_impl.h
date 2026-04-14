/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file pointer_pattern_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_POINTER_PATTERN_IMPL_H
#define IMPL_TENSOR_API_TENSOR_POINTER_PATTERN_IMPL_H

#include "impl/tensor_api/tensor/pointer_mem_impl.h"

namespace AscendC {
namespace Te {

template <typename T, typename = void>
struct IsMemPtrIterator : Std::false_type {};

template <typename T>
struct IsMemPtrIterator<T, void_t<decltype(*Std::declval<T&>())>> : Std::true_type {};

template <typename PtrPattern, typename Iterator>
__aicore__ inline auto MakeHardwareMemPtr(Iterator iter)
{
    return HardwareMemPtrV2<PtrPattern, Iterator>{iter};
}

template <typename PtrPattern, typename T>
struct LocationMemPtrType {
    static_assert(!Std::is_same_v<PtrPattern, PtrPattern>,
        "MakeLocationMemPtr/MakeMemPtr byteOffset overload does not support this Location.");
};

template <typename T>
struct LocationMemPtrType<Location::UB, T> {
    using type = __ubuf__ T*;
};

template <typename T>
struct LocationMemPtrType<Location::L1, T> {
    using type = __cbuf__ T*;
};

template <typename T>
struct LocationMemPtrType<Location::L0A, T> {
    using type = __ca__ T*;
};

template <typename T>
struct LocationMemPtrType<Location::L0B, T> {
    using type = __cb__ T*;
};

template <typename T>
struct LocationMemPtrType<Location::L0C, T> {
    using type = __cc__ T*;
};

template <typename T>
struct LocationMemPtrType<Location::Bias, T> {
    using type = __biasbuf__ T*;
};

template <typename T>
struct LocationMemPtrType<Location::Fixbuf, T> {
    using type = __fbuf__ T*;
};

template <typename PtrPattern, typename TraitType, typename Arg>
__aicore__ inline auto MakeLocationMemPtr(const Arg& arg)
{
    using T = typename TraitType::type;
    using Pointer = typename LocationMemPtrType<PtrPattern, T>::type;
    return MakeHardwareMemPtr<PtrPattern>(reinterpret_cast<Pointer>(get_imm(0) + arg));
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_POINTER_PATTERN_IMPL_H
