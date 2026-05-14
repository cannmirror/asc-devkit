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
    "impl/tensor_api/algorithm/copy_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file copy_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_ALGORITHM_COPY_IMPL_H
#define IMPL_TENSOR_API_ALGORITHM_COPY_IMPL_H

#include "impl/tensor_api/atom/copy_atom_impl.h"

namespace AscendC {
namespace Te {

template <typename Tp, const Tp& traits, typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params)
{
    atomCopy.template Call<traits>(params...);
}

template <typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params)
{
    atomCopy.Call(params...);
}

using CopyDispatchSet = TupleMap<
    Std::tuple<Std::tuple<Location::L1, Location::GM>, CopyAtom<CopyGM2L1>>,
    Std::tuple<Std::tuple<Location::UB, Location::GM>, CopyAtom<CopyGM2UB>>,
    Std::tuple<Std::tuple<Location::GM, Location::UB>, CopyAtom<CopyUB2GM>>,
    Std::tuple<Std::tuple<Location::L1, Location::UB>, CopyAtom<CopyUB2L1>>,
    Std::tuple<Std::tuple<Location::UB, Location::L1>, CopyAtom<CopyL12UB>>,
    Std::tuple<Std::tuple<Location::L0A, Location::L1>, CopyAtom<CopyL12L0A>>,
    Std::tuple<Std::tuple<Location::L0B, Location::L1>, CopyAtom<CopyL12L0B>>,
    Std::tuple<Std::tuple<Location::BIAS, Location::L1>, CopyAtom<CopyL12BT>>,
    Std::tuple<Std::tuple<Location::FIXBUF, Location::L1>, CopyAtom<CopyL12FB>>,
    Std::tuple<Std::tuple<Location::GM, Location::L0C>, CopyAtom<CopyL0C2GM>>,
    Std::tuple<Std::tuple<Location::UB, Location::L0C>, CopyAtom<CopyL0C2UB>>>;

template <typename T, typename U, Std::enable_if_t<IsAttrTensorV<T> && IsAttrTensorV<U>, int> = 0,
    typename... Params>
__aicore__ inline void
Copy(const T& dst, const U& src, const Params& ...params)
{
    using DstLocation = GetMemLocation<T>;
    using SrcLocation = GetMemLocation<U>;
    using CopyAtomType = typename CopyDispatchSet::template Get<Std::tuple<DstLocation, SrcLocation>>;
    static_assert(!Std::is_same_v<CopyAtomType, EmptyValue>, "Unsupported Copy dst/src location combination.");
    CopyAtomType{}.Call(dst, src, params...);
}

template <typename... Args>
__aicore__ inline auto MakeCopy(const Args& ...traits) {
    return CopyAtom<CopyTraits<Args...>>{};
}

}
}

#endif // IMPL_TENSOR_API_ALGORITHM_COPY_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
