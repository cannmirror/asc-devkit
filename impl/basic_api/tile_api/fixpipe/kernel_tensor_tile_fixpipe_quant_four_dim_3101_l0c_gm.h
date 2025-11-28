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
 * \file kernel_tensor_tile_fixpipe_quant_four_dim_3101_l0c_gm.h
 * \brief
 */
#ifndef IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_QUANT_FOUR_DIM_3101_L0C_GM_H
#define IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_QUANT_FOUR_DIM_3101_L0C_GM_H

#include "kernel_tensor_tile_fixpipe_quant_nz2nz_four_dim_3101_l0c_gm.h"
#include "kernel_tensor_tile_fixpipe_quant_nz2nd_four_dim_3101_l0c_gm.h"
#include "kernel_tensor_tile_fixpipe_quant_nz2dn_four_dim_3101_l0c_gm.h"

namespace AscendC {
namespace TileInternal {

enum class Format : uint8_t { None, NZ, ND, DN };
enum class QuantMode : uint8_t { None, Scalar, Vector, Direct };

template <typename T>
__aicore__ inline constexpr Format GetDataFormat()
{
    using traitType = GetTensorTraitType<T>;
    if constexpr (IsL0cNZFormat<traitType>::value) {
        return Format::NZ;
    } else if constexpr (IsNDFormat<traitType>::value) {
        return Format::ND;
    } else if constexpr (IsDNFormat<traitType>::value) {
        return Format::DN;
    }
    return Format::None;
}

template <const FixpipeTrait& trait>
__aicore__ inline constexpr QuantMode GetQuantMode()
{
    if constexpr (IsVectorQuantMode<trait.quantPre>()) {
        return QuantMode::Vector;
    } else if constexpr (IsScalarQuantMode<trait.quantPre>()) {
        return QuantMode::Scalar;
    } else if constexpr (IsDirectQuantMode<trait.quantPre>()) {
        return QuantMode::Direct;
    }
    return QuantMode::None;
}

class FormatRegistorIgnore {
public:
    template <typename T, typename U, typename V, const DataCopyTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant) {}
};

template <Format dstFormat, Format srcFormat, QuantMode quantMode>
struct FormatRegistor {
    using type = FormatRegistorIgnore;
};

template <>
struct FormatRegistor<Format::NZ, Format::NZ, QuantMode::Direct> {
    using type = FixpipeNZ2NZSimpleQuant;
};

template <>
struct FormatRegistor<Format::ND, Format::NZ, QuantMode::Direct> {
    using type = FixpipeNZ2NDSimpleQuant;
};

template <>
struct FormatRegistor<Format::DN, Format::NZ, QuantMode::Direct> {
    using type = FixpipeNZ2DNSimpleQuant;
};

template <>
struct FormatRegistor<Format::NZ, Format::NZ, QuantMode::Scalar> {
    using type = FixpipeNZ2NZSimpleQuant;
};

template <>
struct FormatRegistor<Format::ND, Format::NZ, QuantMode::Scalar> {
    using type = FixpipeNZ2NDSimpleQuant;
};

template <>
struct FormatRegistor<Format::DN, Format::NZ, QuantMode::Scalar> {
    using type = FixpipeNZ2DNSimpleQuant;
};

template <>
struct FormatRegistor<Format::NZ, Format::NZ, QuantMode::Vector> {
    using type = FixpipeNZ2NZVectorQuant;
};

template <>
struct FormatRegistor<Format::ND, Format::NZ, QuantMode::Vector> {
    using type = FixpipeNZ2NDVectorQuant;
};

template <>
struct FormatRegistor<Format::DN, Format::NZ, QuantMode::Vector> {
    using type = FixpipeNZ2DNVectorQuant;
};

template <typename T, typename U, const FixpipeTrait& trait>
__aicore__ inline void CheckFixpipeQuantParams()
{
    using srcType = typename GetTensorTraitType<U>::LiteType;
    using dstType = typename GetTensorTraitType<T>::LiteType;
    using currentType = Std::tuple<srcType, dstType>;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3101
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
        if constexpr (trait.quantPre == QuantMode_t::NoQuant) {
            using quantDataType1 = Std::tuple<float, float>;
            using quantDataType2 = Std::tuple<int32_t, int32_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType1, quantDataType2>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::F322F16 || trait.quantPre == QuantMode_t::QF322F16_PRE ||
                trait.quantPre == QuantMode_t::VQF322F16_PRE) {
            using quantDataType = Std::tuple<float, half>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::F322BF16 || trait.quantPre == QuantMode_t::QF322BF16_PRE ||
                trait.quantPre == QuantMode_t::VQF322BF16_PRE) {
            using quantDataType = Std::tuple<float, bfloat16_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::DEQF16 || trait.quantPre == QuantMode_t::VDEQF16) {
            using quantDataType = Std::tuple<int32_t, half>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::QF322B8_PRE || trait.quantPre == QuantMode_t::VQF322B8_PRE) {
            using quantDataType1 = Std::tuple<float, int8_t>;
            using quantDataType2 = Std::tuple<float, uint8_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType1, quantDataType2>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::REQ8 || trait.quantPre == QuantMode_t::VREQ8) {
            using quantDataType1 = Std::tuple<int32_t, int8_t>;
            using quantDataType2 = Std::tuple<int32_t, uint8_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType1, quantDataType2>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::QF322FP8_PRE || trait.quantPre == QuantMode_t::VQF322FP8_PRE) {
            using quantDataType = Std::tuple<float, fp8_e4m3fn_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::QF322HIF8_PRE || trait.quantPre == QuantMode_t::VQF322HIF8_PRE ||
                trait.quantPre == QuantMode_t::QF322HIF8_PRE_HYBRID ||  trait.quantPre == QuantMode_t::VQF322HIF8_PRE_HYBRID) {
            using quantDataType = Std::tuple<float, hifloat8_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::QS322BF16_PRE || trait.quantPre == QuantMode_t::VQS322BF16_PRE) {
            using quantDataType = Std::tuple<int32_t, bfloat16_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::QF322F32_PRE || trait.quantPre == QuantMode_t::VQF322F32_PRE) {
            using quantDataType = Std::tuple<float, float>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        }
    }
#endif
}

class FixpipeQuantFourDim3101L0C2GM {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant)
    {
        CheckFixpipeQuantParams<T, U, trait>();
        using FixpipeQuantL0C2GM =
            typename FormatRegistor<GetDataFormat<T>(), GetDataFormat<U>(), GetQuantMode<trait>()>::type;
        FixpipeQuantL0C2GM{}.template Run<T, U, V, trait>(dst, src, quant);
    }
};
}  // namespace TileInternal
}  // namespace AscendC

#endif  // IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_QUANT_FOUR_DIM_3101_L0C_GM_H
