/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "impl/tensor_api/tensor/pointer_pattern_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

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
    return HardwareMemPtr<PtrPattern, Iterator>{iter};
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
struct LocationMemPtrType<Location::BIAS, T> {
    using type = __biasbuf__ T*;
};

template <typename T>
struct LocationMemPtrType<Location::FIXBUF, T> {
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

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
