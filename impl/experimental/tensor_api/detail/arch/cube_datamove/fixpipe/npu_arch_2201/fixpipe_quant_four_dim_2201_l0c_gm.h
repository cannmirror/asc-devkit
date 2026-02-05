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
 * \file fixpipe_quant_four_dim_2201_l0c_gm.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_QUANT_FOUR_DIM_2201_L0C_GM_H
#define EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_QUANT_FOUR_DIM_2201_L0C_GM_H

#include "impl/experimental/tensor_api/detail/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_quant_nz2nd_four_dim_2201_l0c_gm.h"
#include "impl/experimental/tensor_api/detail/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_quant_nz2nz_four_dim_2201_l0c_gm.h"

namespace AscendC {
namespace TensorInternal {

enum class Format2201 : uint8_t { None, NZ, ND};
enum class QuantMode2201 : uint8_t { None, Scalar, Vector, Direct };

template <typename T>
__aicore__ inline constexpr Format2201 GetDataFormat2201()
{
    if constexpr (IsL0cNZFormat<T>::value) {
        return Format2201::NZ;
    } else if constexpr (IsNDFormat<T>::value) {
        return Format2201::ND;
    }
    return Format2201::None;
}

template <const FixpipeTrait& trait>
__aicore__ inline constexpr QuantMode2201 GetQuantMode2201()
{
    if constexpr (IsVectorQuantMode<trait.quantPre>()) {
        return QuantMode2201::Vector;
    } else if constexpr (IsScalarQuantMode<trait.quantPre>()) {
        return QuantMode2201::Scalar;
    } else if constexpr (IsDirectQuantMode<trait.quantPre>()) {
        return QuantMode2201::Direct;
    }
    return QuantMode2201::None;
}

class Format2201RegistorIgnore {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename S, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const S& quant, const Coord& coord) {}
};

template <Format2201 dstFormat, Format2201 srcFormat, QuantMode2201 quantMode>
struct Format2201Registor {
    using type = Format2201RegistorIgnore;
};

template <>
struct Format2201Registor<Format2201::NZ, Format2201::NZ, QuantMode2201::Direct> {
    using type = FixpipeNZ2NZ2201SimpleQuant;
};

template <>
struct Format2201Registor<Format2201::ND, Format2201::NZ, QuantMode2201::Direct> {
    using type = FixpipeNZ2ND2201SimpleQuant;
};

template <>
struct Format2201Registor<Format2201::NZ, Format2201::NZ, QuantMode2201::Scalar> {
    using type = FixpipeNZ2NZ2201SimpleQuant;
};

template <>
struct Format2201Registor<Format2201::ND, Format2201::NZ, QuantMode2201::Scalar> {
    using type = FixpipeNZ2ND2201SimpleQuant;
};

template <>
struct Format2201Registor<Format2201::NZ, Format2201::NZ, QuantMode2201::Vector> {
    using type = FixpipeNZ2NZ2201VectorQuant;
};

template <>
struct Format2201Registor<Format2201::ND, Format2201::NZ, QuantMode2201::Vector> {
    using type = FixpipeNZ2ND2201VectorQuant;
};

template <const FixpipeTrait& trait, typename T, typename U>
__aicore__ inline void CheckFixpipe2201QuantParams()
{
    using srcType = typename U::elementType;
    using dstType = typename T::elementType;
    using currentType = Std::tuple<srcType, dstType>;
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        if constexpr (trait.quantPre == QuantMode_t::NoQuant) {
            using quantDataType1 = Std::tuple<__cc__ float, __gm__ float>;
            using quantDataType2 = Std::tuple<__cc__ int32_t, __gm__ int32_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType1, quantDataType2>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::F322F16) {
            using quantDataType = Std::tuple<__cc__ float, __gm__ half>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::F322BF16) {
            using quantDataType = Std::tuple<__cc__ float, __gm__ bfloat16_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::DEQF16 || trait.quantPre == QuantMode_t::VDEQF16) {
            using quantDataType = Std::tuple<__cc__ int32_t, __gm__ half>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::QF322B8_PRE || trait.quantPre == QuantMode_t::VQF322B8_PRE) {
            using quantDataType1 = Std::tuple<__cc__ float, __gm__ int8_t>;
            using quantDataType2 = Std::tuple<__cc__ float, __gm__ uint8_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType1, quantDataType2>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::REQ8 || trait.quantPre == QuantMode_t::VREQ8) {
            using quantDataType1 = Std::tuple<__cc__ int32_t, __gm__ int8_t>;
            using quantDataType2 = Std::tuple<__cc__ int32_t, __gm__ uint8_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType1, quantDataType2>),
                "Failed to check quantPre value in Fixpipe");
        } 
    }
}

class FixpipeQuantFourDim2201L0C2GM {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename S, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const S& quant, const Coord& coord)
    {
        CheckFixpipe2201QuantParams<trait, T, U>();
        using FixpipeQuantCoordL0C2GM =
            typename Format2201Registor<GetDataFormat2201<T>(), GetDataFormat2201<U>(), GetQuantMode2201<trait>()>::type;
        FixpipeQuantCoordL0C2GM{}.template Run<trait, T, U, S, Coord>(dst, src, quant, coord);
    }
};
}  // namespace TensorInternal
}  // namespace AscendC

#endif  // EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_QUANT_FOUR_DIM_2201_L0C_GM_H
