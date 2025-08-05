/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad_utils.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_UTILS_H
#define ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_UTILS_H

#include <type_traits>
#include "../../utils/common_utils.h"
#include "../../utils/integral_constant.h"
#include "../../utils/tuple_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsF16F16F16()
{
    return AscendC::IsSameTypeV<typename AType::T, half> && AscendC::IsSameTypeV<typename BType::T, half> &&
           AscendC::IsSameTypeV<typename CType::T, half>;
}

template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsF16F16F32()
{
    return AscendC::IsSameTypeV<typename AType::T, half> && AscendC::IsSameTypeV<typename BType::T, half> &&
           AscendC::IsSameTypeV<typename CType::T, float>;
}

template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsBf16Bf16Bf16()
{
    return AscendC::IsSameTypeV<typename AType::T, bfloat16_t> && AscendC::IsSameTypeV<typename BType::T, bfloat16_t> &&
           AscendC::IsSameTypeV<typename CType::T, bfloat16_t>;
}

template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsBf16Bf16F32()
{
    return AscendC::IsSameTypeV<typename AType::T, bfloat16_t> && AscendC::IsSameTypeV<typename BType::T, bfloat16_t> &&
           AscendC::IsSameTypeV<typename CType::T, float>;
}

template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsF32F32F32()
{
    return AscendC::IsSameTypeV<typename AType::T, float> && AscendC::IsSameTypeV<typename BType::T, float> &&
           AscendC::IsSameTypeV<typename CType::T, float>;
}

template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsI8I8I32()
{
    return AscendC::IsSameTypeV<typename AType::T, int8_t> && AscendC::IsSameTypeV<typename BType::T, int8_t> &&
           AscendC::IsSameTypeV<typename CType::T, int32_t>;
}

template <class MatmulType>
__aicore__ inline constexpr bool IsF8()
{
#if defined(__DAV_C310__)
    return AscendC::IsSameTypeV<typename MatmulType::T, fp8_e5m2_t> ||
           AscendC::IsSameTypeV<typename MatmulType::T, fp8_e4m3fn_t>;
#else
    return false;
#endif
}

template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsFp8Fp8F32()
{
#if defined(__DAV_C310__)
    return IsF8<AType>() && IsF8<BType>() && AscendC::IsSameTypeV<typename CType::T, float>;
#else
    return false;
#endif
}

template <class AType, class BType, class CType>
__aicore__ inline constexpr bool IsHIF8HIF8F32()
{
#if defined(__DAV_C310__)
    return AscendC::IsSameTypeV<typename AType::T, hifloat8_t> && AscendC::IsSameTypeV<typename BType::T, hifloat8_t> &&
           AscendC::IsSameTypeV<typename CType::T, float>;
#else
    return false;
#endif
}

template <class MatmulType>
__aicore__ inline constexpr bool IsND()
{
    return (MatmulType::format == CubeFormat::ND || MatmulType::format == CubeFormat::ND_ALIGN);
}

template <class MatmulType>
__aicore__ inline constexpr bool IsNz()
{
    return MatmulType::format == CubeFormat::NZ;
}

template <class L1TileShape>
__aicore__ inline constexpr auto GetL1Kb()
{
    static_assert(AscendC::Std::tuple_size_v<L1TileShape> >= 3, "L1TileShape must have at least 3 elements"); // 3: mnk
    if constexpr (AscendC::Std::tuple_size_v<L1TileShape> > 3) { // 3: MNKaKb Kb index
        return GetIntegralConstant<3, L1TileShape>();            // 3: MNKaKb Kb index
    } else {
        return GetIntegralConstant<MNK_K, L1TileShape>();
    }
}

template <class L1TileShape, class L0TileShape>
__aicore__ inline constexpr bool IsTileShapeValid()
{
    constexpr auto l1M = GetIntegralConstant<MNK_M, L1TileShape>();
    constexpr auto l1N = GetIntegralConstant<MNK_N, L1TileShape>();
    constexpr auto l1Ka = GetIntegralConstant<MNK_K, L1TileShape>();
    constexpr auto l1Kb = GetL1Kb<L1TileShape>();

    constexpr auto l0M = GetIntegralConstant<MNK_M, L0TileShape>();
    constexpr auto l0N = GetIntegralConstant<MNK_N, L0TileShape>();
    constexpr auto l0K = GetIntegralConstant<MNK_K, L0TileShape>();

    // Check align
    if constexpr (!(l1M % MATMUL_MNK_ALIGN == 0 && l1N % MATMUL_MNK_ALIGN == 0 && l1Ka % MATMUL_MNK_ALIGN == 0 &&
                    l1Kb % MATMUL_MNK_ALIGN == 0) ||
                  !(l0M % MATMUL_MNK_ALIGN == 0 && l0N % MATMUL_MNK_ALIGN == 0 && l0K % MATMUL_MNK_ALIGN == 0)) {
        return false;
    }
    // Check L1 L0 shape
    return l1M == l0M && l1N == l0N && (l1Ka >= l0K && (l0K == 0 || l1Ka % l0K == 0)) &&
           (l1Kb >= l0K && (l0K == 0 || l1Kb % l0K == 0));
}

template <class AType, class BType, class L1TileShape>
__aicore__ inline constexpr bool IsL1BufferValid(const int bufferNum = DOUBLE_BUFFER_COUNT)
{
    constexpr auto l1M = GetIntegralConstant<MNK_M, L1TileShape>();
    constexpr auto l1N = GetIntegralConstant<MNK_N, L1TileShape>();
    constexpr auto l1Ka = GetIntegralConstant<MNK_K, L1TileShape>();
    constexpr auto l1Kb = GetL1Kb<L1TileShape>();

    return (l1M * l1Ka * sizeof(typename AType::T) + l1N * l1Kb * sizeof(typename BType::T)) * bufferNum <= L1_SIZE;
}

template <class AType, class BType, class L0TileShape>
__aicore__ inline constexpr bool IsL0BufferValid(const int bufferNum = 1) // L0 DB is optional
{
    constexpr auto l0M = GetIntegralConstant<MNK_M, L0TileShape>();
    constexpr auto l0N = GetIntegralConstant<MNK_N, L0TileShape>();
    constexpr auto l0K = GetIntegralConstant<MNK_K, L0TileShape>();

    return l0M * l0K * sizeof(typename AType::T) * bufferNum <= L0A_SIZE &&
           l0N * l0K * sizeof(typename BType::T) * bufferNum <= L0B_SIZE &&
           l0M * l0N * sizeof(typename AscendC::GetMmDstType<typename AType::T>::Type) <= L0C_SIZE;
}

template <class SingleShape, class L0TileShape>
__aicore__ inline constexpr MatmulShapeParams GetMatmulShapeParams()
{
    return {GetIntegralConstant<MNK_M, SingleShape>(), GetIntegralConstant<MNK_N, SingleShape>(),
            GetIntegralConstant<MNK_K, SingleShape>(), GetIntegralConstant<MNK_M, L0TileShape>(),
            GetIntegralConstant<MNK_N, L0TileShape>(), GetIntegralConstant<MNK_K, L0TileShape>()};
}

__aicore__ inline constexpr MatmulFuncParams GetFuncParams(bool intrinsicsCheck)
{
    MatmulFuncParams params{};
    params.intrinsicsCheck = intrinsicsCheck;
    return params;
}

__aicore__ inline constexpr MatmulBiasParams GetBiasParams(bool enableSetBias)
{
    MatmulBiasParams params{};
    params.enableSetBias = enableSetBias;
    return params;
}
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
