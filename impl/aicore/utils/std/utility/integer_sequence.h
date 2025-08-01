//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/*!
 * \file integer_sequence.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_UTILITY_INTEGER_SEQUENCE_H
#define AICORE_UTILS_STD_UTILITY_INTEGER_SEQUENCE_H

namespace AscendC {
namespace Std {

#include <cstddef>

namespace Impl {
constexpr size_t MaxIntegerSequenceSize = 64;
constexpr size_t SpiltSize = 2;
}; // namespace Impl

template <class T, T... Ns>
struct IntegerSequence {
    using type = IntegerSequence;
    using valueType = T;
    static_assert((0 <= sizeof...(Ns) && sizeof...(Ns) <= Impl::MaxIntegerSequenceSize),
        "Std::index_sequence size must be within [0,64].");
    ASCENDC_HOST_AICORE inline static constexpr size_t size()
    {
        return sizeof...(Ns);
    }
};

namespace Impl {

template <class T, class Seq0, class Seq1>
struct MergeSeq {};

template <class T, T... Ns0, T... Ns1>
struct MergeSeq<T, IntegerSequence<T, Ns0...>, IntegerSequence<T, Ns1...>>
    : IntegerSequence<T, Ns0..., (sizeof...(Ns0) + Ns1)...> {};

template <class T, size_t N>
struct MakeIntegerSequence : Impl::MergeSeq<T, typename MakeIntegerSequence<T, N / SpiltSize>::type,
                                 typename MakeIntegerSequence<T, N - N / SpiltSize>::type> {};

template <class T>
struct MakeIntegerSequence<T, 0> : IntegerSequence<T> {};

template <class T>
struct MakeIntegerSequence<T, 1> : IntegerSequence<T, 0> {};

}; // namespace Impl

template <class T, T N>
using MakeIntegerSequenceNoChecked = typename Impl::MakeIntegerSequence<T, N>::type;

template <class T, T N>
struct MakeIntegerSequenceChecked {
    static_assert(0 <= N && N <= Impl::MaxIntegerSequenceSize, "Std::make_index_sequence must be within [0,64].");
    using type = MakeIntegerSequenceNoChecked<T, 0 <= N ? N : 0>;
};

template <class T, T N>
using MakeIntegerSequence = typename MakeIntegerSequenceChecked<T, N>::type;

} // namespace Std
} // namespace AscendC
#endif // AICORE_UTILS_STD_UTILITY_INTEGER_SEQUENCE_H
