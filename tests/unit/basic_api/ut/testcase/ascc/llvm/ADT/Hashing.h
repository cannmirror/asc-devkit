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
#ifndef LLVM_ADT_HASHING_H
#define LLVM_ADT_HASHING_H

#include <string>
#include <functional>
#include <cstddef>

#include "asc_utils.h"
#include "asc_ast_device_consumer.h"

namespace llvm {

inline size_t hash_value(const std::string& s) {
    return s.size();
}

template <typename T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T1, typename T2, typename T3, typename T4>
inline std::size_t hash_combine(const T1& v1, const T2& v2, const T3& v3, const T4& v4) {
    std::size_t seed = 0;
    hash_combine(seed, v1);
    hash_combine(seed, v2);
    hash_combine(seed, v3);
    hash_combine(seed, v4);
    return seed;
}

template<typename T, typename Enable = void>
struct DenseMapInfo {};
}
#endif