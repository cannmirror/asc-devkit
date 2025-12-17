/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef LLVM_ADT_ARRAY_REF_H
#define LLVM_ADT_ARRAY_REF_H
#include <vector>

namespace llvm {
template<typename T>
class ArrayRef {
public:
    ArrayRef() = default;

    template<typename A>
    ArrayRef(const std::vector<T, A> &Vec) : Data(Vec.data()), Length(Vec.size()) {}
    

    const T *Data = nullptr;
    size_t Length = 0;
};

  template<typename T>
  ArrayRef<T> makeArrayRef(const std::vector<T> &Vec) {
    return Vec;
  }

  enum class NoneType { None = 1 };
  const NoneType None = NoneType::None;
}
#endif