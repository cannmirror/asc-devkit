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
* \file hardware_pointer.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_HARDWARE_POINTER_H
#define IMPL_TENSOR_API_TENSOR_HARDWARE_POINTER_H

#include "include/experimental/tensor_api/utils/utils.h"

// pointer_impl.h
namespace AscendC {
namespace TensorInternal{

template <typename Iterator, typename DerivedType>
struct IterAdaptor
{
    using iterator     = Iterator;
    using reference    = typename TensorInternal::IterRef<iterator>::type; // T&
    using elementType = typename TensorInternal::IterEle<iterator>::type; // rm_ref
    using valueType   = typename TensorInternal::IterVal<iterator>::type; // rm_cvf

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

namespace AscendC {
namespace TensorInternal {
template <Hardware hPos, typename Pointer>
struct HardwareMemPtr : IterAdaptor<Pointer, HardwareMemPtr<hPos, Pointer>> {
    using IterAdaptor<Pointer, HardwareMemPtr<hPos, Pointer>>::IterAdaptor;
    static constexpr const Hardware hardPos = hPos;
};

// is hardware mem
template <Hardware hardPos, typename Pointer, typename = void>
struct IsHardwareMem : Std::false_type {};

template <Hardware hardPos, typename Pointer>
struct IsHardwareMem<hardPos, HardwareMemPtr<hardPos, Pointer>> : Std::true_type {};

template <Hardware hardPos, typename Pointer>
struct IsHardwareMem<hardPos, Pointer, TensorInternal::void_t<typename Pointer::iterator>> : IsHardwareMem<hardPos, typename Pointer::iterator> {};

template <Hardware hardPos, typename Pointer>
constexpr bool IsHardwareMemV = IsHardwareMem<hardPos, Pointer>::value;

}
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_HARDWARE_POINTER_H