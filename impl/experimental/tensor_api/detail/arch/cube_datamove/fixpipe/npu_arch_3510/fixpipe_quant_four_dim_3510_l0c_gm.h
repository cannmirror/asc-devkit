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
 * \file fixpipe_quant_four_dim_3510_l0c_gm.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_FOUR_DIM_3510_L0C_GM_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_FOUR_DIM_3510_L0C_GM_H

#include "impl/experimental/tensor_api/detail/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_nz2dn_four_dim_3510_l0c_gm.h"
#include "impl/experimental/tensor_api/detail/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_nz2nd_four_dim_3510_l0c_gm.h"
#include "impl/experimental/tensor_api/detail/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_nz2nz_four_dim_3510_l0c_gm.h"

namespace AscendC {
namespace Te {

enum class Format3510 : uint8_t { None, NZ, ND, DN };
enum class QuantMode3510 : uint8_t { None, Scalar, Vector, Direct };

template <typename T>
__aicore__ inline constexpr Format3510 GetDataFormat()
{
    if constexpr (IsL0cNZFormat<T>::value) {
        return Format3510::NZ;
    } else if constexpr (IsNDFormat<T>::value) {
        return Format3510::ND;
    } else if constexpr (IsDNFormat<T>::value) {
        return Format3510::DN;
    }
    return Format3510::None;
}

template <const FixpipeTrait& trait>
__aicore__ inline constexpr QuantMode3510 GetQuantMode()
{
    if constexpr (IsVectorQuantMode<trait.quantPre>()) {
        return QuantMode3510::Vector;
    } else if constexpr (IsScalarQuantMode<trait.quantPre>()) {
        return QuantMode3510::Scalar;
    } else if constexpr (IsDirectQuantMode<trait.quantPre>()) {
        return QuantMode3510::Direct;
    }
    return QuantMode3510::None;
}

class FormatRegistorIgnore3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename V, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant, const Coord& coord) {}
};

template <Format3510 dstFormat, Format3510 srcFormat, QuantMode3510 QuantMode3510>
struct FormatRegistor3510 {
    using type = FormatRegistorIgnore3510;
};

template <>
struct FormatRegistor3510<Format3510::NZ, Format3510::NZ, QuantMode3510::Direct> {
    using type = FixpipeNZ2NZSimpleQuant3510;
};

template <>
struct FormatRegistor3510<Format3510::ND, Format3510::NZ, QuantMode3510::Direct> {
    using type = FixpipeNZ2NDSimpleQuant3510;
};

template <>
struct FormatRegistor3510<Format3510::DN, Format3510::NZ, QuantMode3510::Direct> {
    using type = FixpipeNZ2DNSimpleQuant3510;
};

template <>
struct FormatRegistor3510<Format3510::NZ, Format3510::NZ, QuantMode3510::Scalar> {
    using type = FixpipeNZ2NZSimpleQuant3510;
};

template <>
struct FormatRegistor3510<Format3510::ND, Format3510::NZ, QuantMode3510::Scalar> {
    using type = FixpipeNZ2NDSimpleQuant3510;
};

template <>
struct FormatRegistor3510<Format3510::DN, Format3510::NZ, QuantMode3510::Scalar> {
    using type = FixpipeNZ2DNSimpleQuant3510;
};

template <>
struct FormatRegistor3510<Format3510::NZ, Format3510::NZ, QuantMode3510::Vector> {
    using type = FixpipeNZ2NZVectorQuant3510;
};

template <>
struct FormatRegistor3510<Format3510::ND, Format3510::NZ, QuantMode3510::Vector> {
    using type = FixpipeNZ2NDVectorQuant3510;
};

template <>
struct FormatRegistor3510<Format3510::DN, Format3510::NZ, QuantMode3510::Vector> {
    using type = FixpipeNZ2DNVectorQuant3510;
};

template <const FixpipeTrait& trait, typename T, typename U>
__aicore__ inline void CheckFixpipeQuantParams()
{
    using srcType = typename U::elementType;
    using dstType = typename T::elementType;
    using currentType = Std::tuple<srcType, dstType>;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
        if constexpr (trait.quantPre == QuantMode_t::NoQuant) {
            using quantDataType1 = Std::tuple<__cc__ float, __gm__ float>;
            using quantDataType2 = Std::tuple<__cc__ int32_t, __gm__ int32_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType1, quantDataType2>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::F322F16 || trait.quantPre == QuantMode_t::QF322F16_PRE ||
                trait.quantPre == QuantMode_t::VQF322F16_PRE) {
            using quantDataType = Std::tuple<__cc__ float, __gm__ half>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::F322BF16 || trait.quantPre == QuantMode_t::QF322BF16_PRE ||
                trait.quantPre == QuantMode_t::VQF322BF16_PRE) {
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
        } else if constexpr (trait.quantPre == QuantMode_t::QF322FP8_PRE || trait.quantPre == QuantMode_t::VQF322FP8_PRE) {
            using quantDataType = Std::tuple<__cc__ float, __gm__ fp8_e4m3fn_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::QF322HIF8_PRE || trait.quantPre == QuantMode_t::VQF322HIF8_PRE ||
                trait.quantPre == QuantMode_t::QF322HIF8_PRE_HYBRID ||  trait.quantPre == QuantMode_t::VQF322HIF8_PRE_HYBRID) {
            using quantDataType = Std::tuple<__cc__ float, __gm__ hifloat8_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::QS322BF16_PRE || trait.quantPre == QuantMode_t::VQS322BF16_PRE) {
            using quantDataType = Std::tuple<__cc__ int32_t, __gm__ bfloat16_t>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        } else if constexpr (trait.quantPre == QuantMode_t::QF322F32_PRE || trait.quantPre == QuantMode_t::VQF322F32_PRE) {
            using quantDataType = Std::tuple<__cc__ float, __gm__ float>;
            static_assert((Std::is_one_of_v<currentType, quantDataType>),
                "Failed to check quantPre value in Fixpipe");
        }
    }
#endif
}

class FixpipeQuantFourDimL0C2GM3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename V, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant, const Coord& coord)
    {
        CheckFixpipeQuantParams<trait, T, U>();
        using FixpipeQuantL0C2GM =
            typename FormatRegistor3510<GetDataFormat<T>(), GetDataFormat<U>(), GetQuantMode<trait>()>::type;
        FixpipeQuantL0C2GM{}.template Run<trait, T, U, V, Coord>(dst, src, quant, coord);
    }
};
}  // namespace Te
}  // namespace AscendC

#endif  // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_FOUR_DIM_3510_L0C_GM_H
