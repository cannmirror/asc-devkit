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
 * \file format_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_UTILS_FORMAT_IMPL_H
#define IMPL_TENSOR_API_UTILS_FORMAT_IMPL_H

#include "impl/experimental/tensor_api/tensor/pointer_impl.h"
#include "impl/experimental/tensor_api/tensor/local_tensor_impl.h"
#include "impl/experimental/tensor_api/utils/constant_impl.h"

namespace AscendC {
namespace Te {
struct ArchVersion {
    static constexpr uint32_t V3510 = 3510;
    static constexpr uint32_t V2201 = 2201;
};

struct GetArchVersion {
    __aicore__ inline constexpr uint32_t operator()() const {
#ifdef __NPU_ARCH__
        return __NPU_ARCH__;
#else
        return 0;
#endif
    }
};

constexpr uint32_t CURRENT_ARCH_VERSION = GetArchVersion{}();
template <typename T>
__aicore__ inline uint8_t GetCacheModeFromTensor(__gm__ T* src) {
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
        return static_cast<uint8_t>((reinterpret_cast<uint64_t>(src)) >> L2_CACHE_OFFSET);
    } else {
        return 0;
    }
}

template <typename T>
struct GetTypeFromFourDimTrait;

template <Hardware hPos, typename Pointer, typename Shape1, typename Shape2, typename Stride1, typename Stride2>
struct GetTypeFromFourDimTrait<LocalTensor<TensorAttribute<ViewEngine<HardwareMemPtr<hPos, Pointer>>, Layout<Shape<Shape1, Shape2>, Stride<Stride1, Stride2>>>>> {
    using ShapeRowsZeroDim = typename Std::tuple_element<0, Shape1>::type;
    using ShapeRowsOneDim = typename Std::tuple_element<1, Shape1>::type;
    using ShapeColumnsZeroDim = typename Std::tuple_element<0, Shape2>::type;
    using ShapeColumnsOneDim = typename Std::tuple_element<1, Shape2>::type;

    using StrideRowsZeroDim = typename Std::tuple_element<0, Stride1>::type;
    using StrideRowsOneDim = typename Std::tuple_element<1, Stride1>::type;
    using StrideColumnsZeroDim = typename Std::tuple_element<0, Stride2>::type;
    using StrideColumnsOneDim = typename Std::tuple_element<1, Stride2>::type;
};

enum class AttrInfo : uint8_t {SHAPE, STRIDE, ROW, COLUMN};

template <typename T, AttrInfo info1, AttrInfo info2, size_t dim> 
struct GetFourDimType;

template <typename T>
struct GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::ShapeRowsZeroDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 1> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::ShapeRowsOneDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::ShapeColumnsZeroDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 1> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::ShapeColumnsOneDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::StrideRowsZeroDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::StrideRowsOneDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::StrideColumnsZeroDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::StrideColumnsOneDim>;
};

template <typename T>
struct IsZZFormat {
private:
    __aicore__ inline static constexpr bool IsFractalZZFormat() {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;

        constexpr bool isShapeRight = Std::is_constant<FRACTAL_FIXED, ShapeRow0>::value 
            && Std::is_constant<C0_SIZE / sizeof(type), ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<C0_SIZE / sizeof(type), StrideRow0>::value 
            && Std::is_constant<1, StrideColumn0>::value;

        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalZZFormat();
};

template <typename T>
struct IsZNFormat {
private:
    __aicore__ inline static constexpr bool IsFractalZNFormat() {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;

        constexpr bool isShapeRight = Std::is_constant<C0_SIZE / sizeof(type), ShapeRow0>::value 
            && Std::is_constant<FRACTAL_FIXED, ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<1, StrideRow0>::value 
            && Std::is_constant<C0_SIZE / sizeof(type), StrideColumn0>::value;

        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalZNFormat();
};

template <typename T>
struct IsNZFormat {
private:
    __aicore__ inline static constexpr bool IsFractalNZFormat() {
        using type = typename T::elementType;
        // NZ shape (Int<16>, row) , (Int<C0Size>, column))
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;        
        // NZ stride (Int<C0Size>, N * C0Size + 16 * Int<C0Size>) , (Int<1>, row * (N * C0Size + 16 * Int<C0Size>)))
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;

        constexpr bool isStrideRight = Std::is_constant<C0_SIZE / sizeof(type), StrideRow0>::value 
            && Std::is_constant<1, StrideColumn0>::value;
        constexpr bool isShapeRight = Std::is_constant<FRACTAL_FIXED, ShapeRow0>::value 
            && Std::is_constant<C0_SIZE / sizeof(type), ShapeColumn0>::value;

        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalNZFormat();
};

template <typename T>
struct IsL0cNZFormat {
private:
    __aicore__ inline static constexpr bool IsFractalL0cNZFormat() {
        using type = typename T::elementType;
        // NZ shape (Int<16>, row) , (Int<C0Size>, column))
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;

        // NZ stride (Int<C0Size>, N * C0Size + 16 * Int<C0Size>) , (Int<1>, row * (N * C0Size + 16 * Int<C0Size>)))
        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;

        constexpr bool isShapeRight = Std::is_constant<FRACTAL_FIXED, ShapeRow0>::value 
            && Std::is_constant<FRACTAL_FIXED, ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<FRACTAL_FIXED, StrideRow0>::value 
            && Std::is_constant<1, StrideColumn0>::value;

        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalL0cNZFormat();
};

template <typename T>
struct IsNDFormat {
private:
    __aicore__ inline static constexpr bool IsFractalNDFormat() {
        // ND shape (Int<1>, row), (Int<1>, column)
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;

        // ND stride (Int<0>, N * column), (Int<0>, Int<1>))
        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;

        constexpr bool isShapeRight = Std::is_constant<1, ShapeRow0>::value && Std::is_constant<1, ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<0, StrideRow0>::value && Std::is_constant<0, StrideColumn0>::value
            && Std::is_constant<1, StrideColumn1>::value;

        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalNDFormat();
};

template <typename T>
struct IsDNFormat {
private:
    __aicore__ inline static constexpr bool IsFractalDNFormat() {
        // DN shape (Int<1>, row), (Int<1>, column)
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;

        // DN stride (Int<0>, Int<1>), (Int<0>, N * row))
        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;

        constexpr bool isShapeRight = Std::is_constant<1, ShapeRow0>::value && Std::is_constant<1, ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<0, StrideRow0>::value && Std::is_constant<1, StrideRow1>::value
            && Std::is_constant<0, StrideColumn0>::value;

        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalDNFormat();
};

template <typename T, AttrInfo info1, AttrInfo info2, size_t dim>
__aicore__ inline constexpr decltype(auto) GetEleFromLayout(const T& layout) {
    if constexpr (info1 == AttrInfo::SHAPE && info2 == AttrInfo::ROW) {
        return Std::get<dim>(Std::get<0>(layout.Shape()));
    } else if constexpr (info1 == AttrInfo::SHAPE && info2 == AttrInfo::COLUMN) {
        return Std::get<dim>(Std::get<1>(layout.Shape()));
    } else if constexpr (info1 == AttrInfo::STRIDE && info2 == AttrInfo::ROW) {
        return Std::get<dim>(Std::get<0>(layout.Stride()));
    } else if constexpr (info1 == AttrInfo::STRIDE && info2 == AttrInfo::COLUMN) {
        return Std::get<dim>(Std::get<1>(layout.Stride()));
    }        
}

template<typename T>
__aicore__ inline constexpr auto GetHardPos()
{
   return T::iterator::hardPos;
}
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_UTILS_FORMAT_IMPL_H
