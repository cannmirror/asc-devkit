/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LLVM_ADT_SMALL_VECTOR_H
#define LLVM_ADT_SMALL_VECTOR_H

namespace llvm {
template <typename T>
class SmallVectorTemplateCommon {
protected:
    void *BeginX;
    size_t Size = 0, Capacity;
public:
    SmallVectorTemplateCommon(size_t Size) {}

    size_t size() const { return Size; }
    size_t capacity() const { return Capacity; }

    using iterator = T *;
    using const_iterator = const T *;
    iterator begin() { return (iterator)this->BeginX; }
    const_iterator begin() const { return (const_iterator)this->BeginX; }
    iterator end() { return begin() + size(); }
    const_iterator end() const { return begin() + size(); }

    using pointer = T *;
    pointer data() { return pointer(begin()); }
};

template <typename T>
class SmallVectorTemplateBase : public SmallVectorTemplateCommon<T> {
public:
    SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

    void push_back(const T &Elt) {}

    void push_back(T &&Elt) {}

    void pop_back() {
    }
};

template <typename T>
class SmallVectorImpl : public SmallVectorTemplateBase<T> {
public:
    SmallVectorImpl(unsigned N) : SmallVectorTemplateBase<T>(N) {}
};

template <typename T, unsigned N>
class SmallVector : public SmallVectorImpl<T> {
public:
    SmallVector() : SmallVectorImpl<T>(N) {}
};
}
#endif