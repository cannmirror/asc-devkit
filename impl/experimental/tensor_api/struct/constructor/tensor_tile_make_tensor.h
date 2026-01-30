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
* \file tensor_tile_make_tensor.h
* \brief
*/
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_TENSOR_H
#define IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_TENSOR_H

#include "impl/experimental/tensor_api/struct/definition/tensor_tile_tensor.h"

namespace AscendC {
namespace TileInternal
{
template <typename T, typename = void>
struct HasDereference : Std::false_type {};

template <typename T>
struct HasDereference<T, void_t<decltype(*Std::declval<T&>())>> : Std::true_type {};

template <typename T>
struct MakeLocalTensor {
template <typename Arg0, typename... Args>
__aicore__ inline constexpr auto operator()(const Arg0& arg0, const Args&... args) const {
    if constexpr (HasDereference<Arg0>::value) {
    using Engine = ViewEngine<Arg0>;
    if constexpr (sizeof...(Args) == 1 && (is_layout<Args>::value && ...)) {
        return LocalTensor<TensorAttribute<Engine, Args...>>{Engine{arg0}, args...};
    } else {
        return LocalTensor<TensorAttribute<Engine, decltype(MakeLayout(args...))>>{Engine{arg0}, MakeLayout(args...)};
    }
    }
}
};

template <typename Iterator, typename... Args>
__aicore__ inline constexpr auto MakeTensorImpl(const Iterator& iter, const Args&... args)
{
    static_assert(HasDereference<Iterator>::value, "Expected iterator iter in MakeLocalTensor(iter, args...)");
    static_assert(!(HasDereference<Args>::value && ...), "Expected layout args... in MakeLocalTensor(iter, args...)");
    return MakeLocalTensor<Iterator>{}(iter, args...);
}
}
} // namespace AscendC

#endif // IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_TENSOR_H