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
 * \file format.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_UTILS_FORMAT_H
#define INCLUDE_TENSOR_API_UTILS_FORMAT_H

#include "impl/experimental/tensor_api/detail/utils/format_impl.h"

namespace AscendC {
namespace Te {

template <typename T, typename U>
constexpr bool VerifyingDataCopyTemplate =
    ((IsTileTensorV<U> && IsTileTensorV<T>) ||
    (IsTileTensorV<U> && IsTileTensorV<T>));

template <typename T, typename U, typename Coord>
constexpr bool VerifyingDataCopyTemplateWithCoord = Std::is_tuple_v<Coord> && VerifyingDataCopyTemplate<T, U>;

template <typename T, typename U>
constexpr bool VerifyingLoadDataTemplate = IsTileTensorV<U> && IsTileTensorV<T>;

template <typename T, typename U, typename Coord>
constexpr bool VerifyingLoadDataTemplateWithCoord = Std::is_tuple_v<Coord> && VerifyingLoadDataTemplate<T, U>;

template <typename T, typename U, typename S>
static constexpr bool VerifyingMmadTemplate = (IsTileTensorV<T> && IsTileTensorV<U> 
    && IsTileTensorV<S>);

template <typename T, typename U, typename S, typename V>
static constexpr bool VerifyingMmadWithBiasTemplate = (IsTileTensorV<T> && IsTileTensorV<U> 
    && IsTileTensorV<S> && IsTileTensorV<V>);

template <typename T, typename U>
static constexpr bool VerifyingFixpipeTemplate = (IsTileTensorV<T> && IsTileTensorV<U>);

template <typename T, typename U, typename V>
static constexpr bool VerifyingFixpipeQuantTemplate = (IsTileTensorV<T> && IsTileTensorV<U>
    && (IsTileTensorV<V> || Std::is_same_v<V, uint64_t>)); 

template <typename T, typename U, typename Coord>
constexpr bool VerifyingFixpipeTemplateWithCoord = Std::is_tuple_v<Coord> && VerifyingFixpipeTemplate<T, U>;

template <typename T, typename U, typename V, typename Coord>
constexpr bool VerifyingFixpipeQuantTemplateWithCoord = Std::is_tuple_v<Coord> && VerifyingFixpipeQuantTemplate<T, U, V>;

} // namespace Te
} // namespace AscendC

#endif // INCLUDE_TENSOR_API_UTILS_FORMAT_H
