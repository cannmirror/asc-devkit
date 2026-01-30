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
* \file tensor_tile_pointer.h
* \brief
*/
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_TENSOR_TILE_POINTER_H
#define IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_TENSOR_TILE_POINTER_H

#include "impl/experimental/tensor_api/utils/tensor_tile_utils.h"

namespace AscendC {
namespace TileInternal{

template <typename Iterator, typename DerivedType>
struct IterAdaptor
{
    using iterator     = Iterator;
    using reference    = typename TileInternal::IterRef<iterator>::type; // T&
    using elementType = typename TileInternal::IterEle<iterator>::type; // rm_ref
    using valueType   = typename TileInternal::IterVal<iterator>::type; // rm_cvf

    __aicore__ inline constexpr IterAdaptor(iterator ptr = {}) : ptr(ptr) {}

    __aicore__ inline constexpr reference operator*() const {
        return *ptr;
    }

    template <typename Index>
    __aicore__ inline constexpr reference operator[](const Index& i) const {
        return ptr[i];
    }

    template <typename Index>
    __aicore__ inline constexpr DerivedType operator+(const Index& i) const {
        return {ptr + i};
    }

    __aicore__ inline constexpr iterator Get() const {
        return ptr; 
    }

    __aicore__ inline constexpr friend bool operator==(const DerivedType& x, const DerivedType& y) {
        return x.ptr == y.ptr;
    }

    __aicore__ inline constexpr friend bool operator!=(const DerivedType& x, const DerivedType& y) {
        return x.ptr != y.ptr;
    }

    __aicore__ inline constexpr friend bool operator< (const DerivedType& x, const DerivedType& y) {
        return x.ptr <  y.ptr; 
    }

    __aicore__ inline constexpr friend bool operator<=(const DerivedType& x, const DerivedType& y) {
        return x.ptr <= y.ptr; 
    }

    __aicore__ inline constexpr friend bool operator> (const DerivedType& x, const DerivedType& y) {
        return x.ptr >  y.ptr; 
    }

    __aicore__ inline constexpr friend bool operator>=(const DerivedType& x, const DerivedType& y) {
        return x.ptr >= y.ptr; 
    }
private:
    iterator ptr; // u8*
};

}
} // namespace AscendC

#endif // IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_TENSOR_TILE_POINTER_H