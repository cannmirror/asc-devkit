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
* \file pointer_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_POINTER_IMPL_H
#define IMPL_TENSOR_API_TENSOR_POINTER_IMPL_H

#include "impl/experimental/tensor_api/utils/utils_impl.h"

namespace AscendC {
namespace Te {

// struct hardware_pointer
template <typename Iterator, typename DerivedType>
struct IterAdaptor
{
    using iterator     = Iterator;
    using reference    = typename IterRef<iterator>::type; // T&
    using elementType = typename IterEle<iterator>::type; // rm_ref
    using valueType   = typename IterVal<iterator>::type; // rm_cvf

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
struct IsHardwareMem<hardPos, Pointer, void_t<typename Pointer::iterator>> : IsHardwareMem<hardPos, typename Pointer::iterator> {};

template <Hardware hardPos, typename Pointer>
constexpr bool IsHardwareMemV = IsHardwareMem<hardPos, Pointer>::value;

// struct engine
template <typename Iterator>
struct ViewEngine
{
    using iterator     = Iterator;
    using reference    = typename IterRef<iterator>::type; // T&
    using elementType = typename IterEle<iterator>::type; // rm_ref
    using valueType   = typename IterVal<iterator>::type; // rm_cvf
    __aicore__ inline constexpr iterator const& Begin() const {
        return storage;
    }

    __aicore__ inline constexpr iterator& Begin() {
        return storage;
    }
    __aicore__ inline constexpr ViewEngine(iterator storage = {}) : storage(storage) {}
private:
    iterator storage;
};

template <typename Iterator>
struct ConstViewEngine
{
    using iterator     = Iterator;
    using reference    = typename IterRef<iterator>::type; // T&
    using elementType = typename IterEle<iterator>::type; // rm_ref
    using valueType   = typename IterVal<iterator>::type; // rm_cvf

    __aicore__ inline constexpr iterator const& Begin() const {
        return storage;
    }
    __aicore__ inline constexpr ConstViewEngine(iterator storage = {}) : storage(storage) {}
private:
    iterator storage;
};

// pointer.h
template <Hardware hardPos, typename Iterator>
__aicore__ inline constexpr auto MakeMemPtr(Iterator iter) 
{
    if constexpr (IsHardwareMem<hardPos, Iterator>::value) {
        return iter;
    } else {
        return HardwareMemPtr<hardPos, Iterator>{iter};
    }
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeGMmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::GM, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeUBmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::UB, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL1memPtr(Iterator iter) {
    return MakeMemPtr<Hardware::L1, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0AmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::L0A, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0BmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::L0B, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0CmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::L0C, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeBiasmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::BIAS, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeFixbufmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::FIXBUF, Iterator>(iter);
}

template <typename T, typename U>
__aicore__ inline constexpr auto MakeUBmemPtr(const U& byteOffset) {
    return MakeUBmemPtr(reinterpret_cast<__ubuf__ T*>(get_imm(0) + byteOffset));
}

template <typename T, typename U>
__aicore__ inline constexpr auto MakeL1memPtr(const U& byteOffset) {
    return MakeL1memPtr(reinterpret_cast<__cbuf__ T*>(get_imm(0) + byteOffset));
}

template <typename T, typename U>
__aicore__ inline constexpr auto MakeL0AmemPtr(const U& byteOffset) {
    return MakeL0AmemPtr(reinterpret_cast<__ca__ T*>(get_imm(0) + byteOffset));
}

template <typename T, typename U>
__aicore__ inline constexpr auto MakeL0BmemPtr(const U& byteOffset) {
    return MakeL0BmemPtr(reinterpret_cast<__cb__ T*>(get_imm(0) + byteOffset));
}

template <typename T, typename U>
__aicore__ inline constexpr auto MakeL0CmemPtr(const U& byteOffset) {
    return MakeL0CmemPtr(reinterpret_cast<__cc__ T*>(get_imm(0) + byteOffset));
}

template <typename T, typename U>
__aicore__ inline constexpr auto MakeBiasmemPtr(const U& byteOffset) {
    return MakeBiasmemPtr(reinterpret_cast<__biasbuf__ T*>(get_imm(0) + byteOffset));
}

template <typename T, typename U>
__aicore__ inline constexpr auto MakeFixbufmemPtr(const U& byteOffset) {
    return MakeFixbufmemPtr(reinterpret_cast<__fbuf__ T*>(get_imm(0) + byteOffset));
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_POINTER_IMPL_H