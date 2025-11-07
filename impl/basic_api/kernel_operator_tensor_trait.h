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

/*!
 * \file kernel_operator_tensor_trait.h
 * \brief
 */
#ifndef TIKCFW_IMPL_KERNEL_OPERATOR_TENSOR_TRAIT_H
#define TIKCFW_IMPL_KERNEL_OPERATOR_TENSOR_TRAIT_H

#include "kernel_operator_coord.h"
#include "kernel_event.h"

namespace AscendC {

/* \brief the TensorTrait of tensor;
 * \note this struct contains primitive type of tensor;
 * info:
 * LiteType: the tensor's primitive type
 * If the layou type is used, the type must be specified and cannot be implicitly deduced.
 */
template <typename T, TPosition pos = TPosition::GM, typename LayoutType = Layout<Shape<>, Stride<>>>
struct TensorTrait {
    using LiteType = T;
    using LiteLayoutType = LayoutType;
    static constexpr const TPosition tPos = pos;
public:
    __aicore__ inline TensorTrait(const LayoutType& t = {})
    {
        static_assert(is_layout_v<LayoutType>, "TensorTrait without layout instantiation!");
        this->layout_ = t;
    }

    __aicore__ inline LayoutType& GetLayout()
    {
        static_assert(is_layout_v<LayoutType>, "TensorTrait without layout instantiation!");
        return layout_;
    }

    __aicore__ inline const LayoutType& GetLayout() const
    {
        static_assert(is_layout_v<LayoutType>, "TensorTrait without layout instantiation!");
        return layout_;
    }

    __aicore__ inline void SetLayout(const LayoutType& t)
    {
        static_assert(is_layout_v<LayoutType>, "TensorTrait without layout instantiation!");
        this->layout_ = t;
    }

    __aicore__ inline decltype(auto) GetShape() const
    {
        static_assert(is_layout_v<LayoutType>, "TensorTrait without layout instantiation!");
        return layout_.GetShape();
    }

    __aicore__ inline decltype(auto) GetStride() const
    {
        static_assert(is_layout_v<LayoutType>, "TensorTrait without layout instantiation!");
        return layout_.GetStride();
    }
private:
    LayoutType layout_ = {};
};

template <typename T, TPosition pos, typename LayoutType>
__aicore__ inline constexpr auto MakeTensorTrait(const LayoutType& t)
{
    static_assert(is_layout_v<LayoutType>, "Input parameters does not contain the layout type!");
    return TensorTrait<T, pos, LayoutType>(t);
}

template <typename T>
struct is_tensorTrait : Std::false_type {};

template <typename T, TPosition pos, typename ShapeType, typename StrideType>
struct is_tensorTrait<TensorTrait<T, pos, Layout<ShapeType, StrideType>>> : Std::true_type {};

template <typename T>
constexpr bool is_tensorTrait_v = is_tensorTrait<T>::value;

} // namespace AscendC
#endif
