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
* \file tensor_tile_tensor.h
* \brief
*/
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_TENSOR_TILE_TENSOR_H
#define IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_TENSOR_TILE_TENSOR_H

#include "impl/experimental/tensor_api/struct/definition/tensor_tile_engine.h"
#include "impl/experimental/tensor_api/struct/definition/tensor_tile_layout.h"

namespace AscendC {

template <typename EngineType, typename LayoutType>
struct TensorAttribute {};

template <typename T>
struct LocalTensor {};

template <typename EngineType, typename LayoutType>
struct LocalTensor<TensorAttribute<EngineType, LayoutType>> {
    using iterator = typename EngineType::iterator;
    using valueType = typename EngineType::valueType;
    using elementType = typename EngineType::elementType;
    using reference = typename EngineType::reference;

    using engineType  = EngineType;
    using layoutType  = LayoutType;

    Std::tuple<layoutType, engineType> rep;

    __aicore__ inline constexpr LocalTensor() {}
    __aicore__ inline constexpr LocalTensor(const EngineType& engine, const LayoutType& layout) : rep(layout, engine) {}

    static constexpr int rank  = LayoutType::rank; // tuple size

    __aicore__ inline constexpr decltype(auto) Tensor() const {
        return *this;
    }

    __aicore__ inline constexpr decltype(auto) Engine() const {
        return Std::get<1>(rep);
    }

    __aicore__ inline constexpr decltype(auto) Engine() {
        return Std::get<1>(rep);
    }

    __aicore__ inline constexpr decltype(auto) Layout() const {
        return Std::get<0>(rep);
    }

    __aicore__ inline constexpr decltype(auto) Data() const {
        return Engine().Begin();
    }

    __aicore__ inline constexpr decltype(auto) Data() {
        return Engine().Begin();
    }

    __aicore__ inline constexpr decltype(auto) Shape() const {
        return Layout().Shape();
    }

    __aicore__ inline constexpr decltype(auto) Stride() const {
        return Layout().Stride();
    }

    __aicore__ inline constexpr auto Size() const {
        return Layout().Size();
    }

    __aicore__ inline constexpr auto Capacity() const {
        return Layout().Capacity();
    }

    template <typename Coord>
    __aicore__ inline constexpr decltype(auto) operator[](const Coord& coord) {
        return Data()[Layout()(coord)];
    }

    template <typename Coord>
    __aicore__ inline constexpr decltype(auto) operator[](const Coord& coord) const {
        return Data()[Layout()(coord)];
    }

    template <typename Coord>
    __aicore__ inline constexpr decltype(auto) operator()(const Coord& coord) {
        return Data()[Layout()(coord)];
    }

    template <typename Coord>
    __aicore__ inline constexpr decltype(auto) operator()(const Coord& coord) const {
        return Data()[Layout()(coord)];
    }

    template <typename Coord0, typename Coord1, typename... Coords>
    __aicore__ inline constexpr decltype(auto) operator()(const Coord0& c0, const Coord1& c1, const Coords&... cs) {
        return operator()(MakeCoord(c0,c1,cs...));
    }

    template <typename Coord0, typename Coord1, typename... Coords>
    __aicore__ inline constexpr decltype(auto) operator()(const Coord0& c0, const Coord1& c1, const Coords&... cs) const {
        return operator()(MakeCoord(c0,c1,cs...));
    }

    template <typename... Layouts>
    __aicore__ inline constexpr auto Compose(const Layouts&... layouts) {
        return MakeTensor(Data(), Layout().Compose(layouts...));
    }

    template <typename... Layouts>
    __aicore__ inline constexpr auto Compose(const Layouts&... layouts) const {
        return MakeTensor(Data(), Layout().Compose(layouts...));
    }

    template <typename... Layouts>
    __aicore__ inline constexpr auto Tile(const Layouts&... layouts) {
        return MakeTensor(Data(), Layout().Tile(layouts...));
    }

    template <typename... Layouts>
    __aicore__ inline constexpr auto Tile(const Layouts&... layouts) const {
        return MakeTensor(Data(), Layout().Tile(layouts...));
    }
};

template <typename T>
struct IsTileTensor : Std::false_type {};

template <typename Engine, typename Layout>
struct IsTileTensor<LocalTensor<TensorAttribute<Engine,Layout>>> : Std::true_type {};

template <typename T>
constexpr bool IsTileTensorV = IsTileTensor<T>::value;

} // namespace AscendC

#endif // IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_TENSOR_TILE_TENSOR_H