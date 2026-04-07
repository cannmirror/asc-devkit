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
* \file get_pattern.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_GET_PATTERN_H
#define IMPL_TENSOR_API_TENSOR_GET_PATTERN_H

#include "impl/experimental/tensor_api/tensor/layout_definition.h"
#include "impl/experimental/tensor_api/tensor/local_tensor_impl.h"

namespace AscendC {
namespace Te {

template <typename T, typename = void>
struct PatternTrait;

template <typename T>
struct PatternTrait<T, Std::enable_if_t<is_layout_v<T>>> {
    using type = typename Std::remove_cvref_t<T>::tag;
};

template <typename T>
struct PatternTrait<T, Std::enable_if_t<IsTileTensorV<T>>> {
    using type = typename PatternTrait<decltype(Std::declval<const T&>().Layout())>::type;
};

template <typename T>
using PatternTraitT = typename PatternTrait<T>::type;

}
}
#endif 