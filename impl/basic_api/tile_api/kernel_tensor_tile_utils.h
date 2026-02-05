/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_tensor_tile_utils.h
 * \brief
 */
#ifndef IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_UTILS_H
#define IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_UTILS_H

#include "kernel_operator_tensor_trait.h"
#include "kernel_struct_data_copy.h"
#include "kernel_tensor.h"

namespace AscendC {
namespace TileInternal {

constexpr size_t TWO_DIM_DATA = 2;
constexpr size_t FOUR_DIM_DATA = 4;
constexpr size_t C0_SIZE = 32;
constexpr size_t FRACTAL_FIXED = 16;

struct ArchVersion {
    static constexpr uint32_t V3101 = 3101;
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

template <QuantMode_t Value, QuantMode_t... Args>
struct is_one_of_value : Std::false_type {};

template <QuantMode_t Value, QuantMode_t Head, QuantMode_t... Tail>
struct is_one_of_value<Value, Head, Tail...>
    : Std::bool_constant<(Value == Head) || is_one_of_value<Value, Tail...>::value> {};

template <QuantMode_t Value, QuantMode_t... Args>
inline constexpr bool is_one_of_value_v = is_one_of_value<Value, Args...>::value;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3101
#define SCALAR_QUANT_MODE QuantMode_t::DEQF16, QuantMode_t::QF322B8_PRE, QuantMode_t::REQ8,\
    QuantMode_t::QS322BF16_PRE, QuantMode_t::QF322F16_PRE, QuantMode_t::QF322BF16_PRE, QuantMode_t::QF322FP8_PRE,\
    QuantMode_t::QF322HIF8_PRE, QuantMode_t::QF322HIF8_PRE_HYBRID, QuantMode_t::QF322F32_PRE
#else
#define SCALAR_QUANT_MODE QuantMode_t::NoQuant
#endif

template <QuantMode_t quantPre>
using IsScalarQuantMode = is_one_of_value<quantPre, SCALAR_QUANT_MODE>;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3101
#define TILE_OP_INTERNAL_TENSOR_QUANT_MODE QuantMode_t::VDEQF16, QuantMode_t::VQF322B8_PRE, QuantMode_t::VREQ8,\
    QuantMode_t::VQS322BF16_PRE, QuantMode_t::VQF322F16_PRE, QuantMode_t::VQF322BF16_PRE, QuantMode_t::VQF322FP8_PRE,\
    QuantMode_t::VQF322HIF8_PRE, QuantMode_t::VQF322HIF8_PRE_HYBRID, QuantMode_t::VQF322F32_PRE
#else
#define TILE_OP_INTERNAL_TENSOR_QUANT_MODE QuantMode_t::NoQuant
#endif

template <QuantMode_t quantPre>
using IsVectorQuantMode = is_one_of_value<quantPre, TILE_OP_INTERNAL_TENSOR_QUANT_MODE>;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3101
#define TILE_OP_INTERNAL_DIRECT_QUANT_MODE QuantMode_t::F322F16, QuantMode_t::F322BF16
#else
#define TILE_OP_INTERNAL_DIRECT_QUANT_MODE QuantMode_t::NoQuant
#endif

template <QuantMode_t quantPre>
using IsDirectQuantMode = is_one_of_value<quantPre, TILE_OP_INTERNAL_DIRECT_QUANT_MODE>;

template <typename T>
__aicore__ inline auto AllocTempBuf(const T& calNSize)
{
    uint64_t deqTensorTempBuf = 0;
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
        constexpr uint32_t deqTensorAddrAlignValue = 128;
        uint16_t deqDataSize = DivCeil(calNSize * sizeof(uint64_t), deqTensorAddrAlignValue) * deqTensorAddrAlignValue;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        deqTensorTempBuf = reinterpret_cast<uint64_t>(ConstDefiner::Instance().hardwareCpuBufferMap.at(Hardware::FIXBUF) + 0);
#else
        deqTensorTempBuf = reinterpret_cast<uint64_t>(get_imm(0));
#endif
    }
    return deqTensorTempBuf;
}

template <typename T>
__aicore__ inline void FreeTempBuf(const T& deqTensorTempBuf)
{
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
        (void)(deqTensorTempBuf);
    }
}

template <typename T>
__aicore__ inline void SetFpc(const T& deqTensorTempBuf)
{
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
        uint64_t deqTensorAddr = (deqTensorTempBuf >> static_cast<uint64_t>(7)) << 8;
        set_fpc(deqTensorAddr);
    }
}

template <typename T>
__aicore__ inline uint8_t GetCacheModeFromTensor(const GlobalTensor<T>& tensor) {
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
        return ExtractCacheMode(tensor);
    } else {
        return 0;
    }
}

__aicore__ inline void InsertPipeFix() 
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3101
    pipe_barrier(PIPE_FIX);
#endif
}

// IsLocalTensorTrait IsGlobalTensorTrait
template <typename T>
struct IsGlobalTensorTrait : Std::false_type {};

template <typename T, TPosition pos, typename ShapeType, typename StrideType>
struct IsGlobalTensorTrait<GlobalTensor<TensorTrait<T, pos, Layout<ShapeType, StrideType>>>> : Std::true_type {};

template <typename T>
constexpr bool IsGlobalTensorTraitV = IsGlobalTensorTrait<T>::value;

template <typename T>
struct IsLocalTensorTrait : Std::false_type {};

template <typename T, TPosition pos, typename ShapeType, typename StrideType>
struct IsLocalTensorTrait<LocalTensor<TensorTrait<T, pos, Layout<ShapeType, StrideType>>>> : Std::true_type {};

template <typename T>
constexpr bool IsLocalTensorTraitV = IsLocalTensorTrait<T>::value;

// GetTypeFromTensor
template <typename T>
struct GetTypeFromTensor;

template <typename T, TPosition pos, typename Shape, typename Stride>
struct GetTypeFromTensor<LocalTensor<TensorTrait<T, pos, Layout<Shape, Stride>>>> {
     using TraitType = TensorTrait<T, pos, Layout<Shape, Stride>>;
};

template <typename T, TPosition pos, typename Shape, typename Stride>
struct GetTypeFromTensor<GlobalTensor<TensorTrait<T, pos, Layout<Shape, Stride>>>> {
     using TraitType = TensorTrait<T, pos, Layout<Shape, Stride>>;
};

template <typename T>
using GetTensorTraitType = Std::remove_cvref_t<typename GetTypeFromTensor<T>::TraitType>;

// IsIntegralConstant
template <typename T>
struct IsIntegralConstant : Std::false_type {};

template <size_t Value>
struct IsIntegralConstant<Std::Int<Value>> : Std::true_type {};

template <typename T>
constexpr bool IsIntegralConstantV = IsIntegralConstant<T>::value;

template <typename T>
struct GetTypeFromFourDimTrait;

template <typename T, TPosition pos, typename Shape1, typename Shape2, typename Stride1, typename Stride2>
struct GetTypeFromFourDimTrait<TensorTrait<T, pos, Layout<Shape<Shape1, Shape2>, Stride<Stride1, Stride2>>>> {
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
struct IsNZFormat {
private:
    __aicore__ inline static constexpr bool IsFractalNZFormat() {
        using type = typename T::LiteType;
        // NZ shape (Int<16>, row) , (Int<C0Size>, column))
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;

        // NZ stride (Int<C0Size>, N * C0Size + 16 * Int<C0Size>) , (Int<1>, row * (N * C0Size + 16 * Int<C0Size>)))
        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;

        constexpr bool isShapeRight = Std::is_constant<FRACTAL_FIXED, ShapeRow0>::value 
            && Std::is_constant<C0_SIZE / sizeof(type), ShapeColumn0>::value;
        constexpr bool isStrideRight = Std::is_constant<C0_SIZE / sizeof(type), StrideRow0>::value 
            && Std::is_constant<1, StrideColumn0>::value;

        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalNZFormat();
};

template <typename T>
struct IsL0cNZFormat {
private:
    __aicore__ inline static constexpr bool IsFractalL0cNZFormat() {
        using type = typename T::LiteType;
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
        return Std::get<dim>(Std::get<0>(layout.GetShape()));
    } else if constexpr (info1 == AttrInfo::SHAPE && info2 == AttrInfo::COLUMN) {
        return Std::get<dim>(Std::get<1>(layout.GetShape()));
    } else if constexpr (info1 == AttrInfo::STRIDE && info2 == AttrInfo::ROW) {
        return Std::get<dim>(Std::get<0>(layout.GetStride()));
    } else if constexpr (info1 == AttrInfo::STRIDE && info2 == AttrInfo::COLUMN) {
        return Std::get<dim>(Std::get<1>(layout.GetStride()));
    }        
}

// NZ
template <typename T>
__aicore__ inline decltype(auto) MakeLayoutForNZ(size_t row, size_t column)
{
    auto shape = MakeShape(MakeShape(Std::Int<FRACTAL_FIXED>{}, row / FRACTAL_FIXED),
        MakeShape(Std::Int<C0_SIZE / sizeof(T)>{}, column / (C0_SIZE / sizeof(T))));
    auto stride = MakeStride(MakeStride(Std::Int<C0_SIZE / sizeof(T)>{}, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>{}),
        MakeStride(Std::Int<1>{}, C0_SIZE / sizeof(T) * row));
    return MakeLayout(shape, stride);
}

template <typename T, size_t row, size_t column>
using NZShapeFormat = Shape<Shape<Std::Int<FRACTAL_FIXED>, Std::Int<row / FRACTAL_FIXED>>,
    Shape<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<column / (C0_SIZE / sizeof(T))>>>;

template <typename T, size_t row, size_t column>
using NZStrideFormat = Stride<Stride<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>>,
    Stride<Std::Int<1>, Std::Int<C0_SIZE / sizeof(T) * row>>>;

// ND
template <typename T>
__aicore__ inline decltype(auto) MakeLayoutForND(size_t row, size_t column)
{
    auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<1>{}, column));
    auto stride = MakeStride(MakeStride(Std::Int<0>{}, column), MakeStride(Std::Int<0>{},  Std::Int<1>{}));
    return MakeLayout(shape, stride);
}

template <typename T, size_t row, size_t column>
using NDShapeFormat = Shape<Shape<Std::Int<1>, Std::Int<row>>, Shape<Std::Int<1>, Std::Int<column>>>;

template <typename T, size_t row, size_t column>
using NDStrideFormat = Stride<Stride<Std::Int<0>, Std::Int<column>>, Stride<Std::Int<0>, Std::Int<1>>>;

// DN
template <typename T>
__aicore__ inline decltype(auto) MakeLayoutForDN(size_t row, size_t column)
{
    auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<1>{}, column));
    auto stride = MakeStride(MakeStride(Std::Int<0>{}, Std::Int<1>{}), MakeStride(Std::Int<0>{}, row));
    return MakeLayout(shape, stride);
}

template <typename T, size_t row, size_t column>
using DNShapeFormat = Shape<Shape<Std::Int<1>, Std::Int<row>>, Shape<Std::Int<1>, Std::Int<column>>>;

template <typename T, size_t row, size_t column>
using DNStrideFormat = Stride<Stride<Std::Int<0>, Std::Int<1>>, Stride<Std::Int<0>, Std::Int<row>>>;

// ZN
template <typename T>
__aicore__ inline decltype(auto) MakeLayoutForZN(size_t row, size_t  column)
{
    auto shape = MakeShape(MakeShape(Std::Int<C0_SIZE / sizeof(T)>{}, row / (C0_SIZE / sizeof(T))),
        MakeShape(Std::Int<FRACTAL_FIXED>{}, column / FRACTAL_FIXED));
    auto stride = MakeStride(MakeStride(Std::Int<1>{}, C0_SIZE / sizeof(T) * column),
        MakeStride(Std::Int<C0_SIZE / sizeof(T)>{}, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>{}));
    return MakeLayout(shape, stride);
}

template <typename T, size_t  row, size_t  column>
using ZNShapeFormat = Shape<Shape<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<row / (C0_SIZE / sizeof(T))>>,
    Shape<Std::Int<FRACTAL_FIXED>, Std::Int<column / FRACTAL_FIXED>>>;

template <typename T, size_t  row, size_t  column>
using ZNStrideFormat = Stride<Stride<Std::Int<1>, Std::Int<C0_SIZE / sizeof(T) * column>>,
    Stride<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>>>;

} // namespace TileInternal
} // namespace AscendC

#endif // IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_UTILS_H
