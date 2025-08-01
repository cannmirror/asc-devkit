/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file cube_in_buffer_utils.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_UTILS_H
#define AICORE_ADV_API_DETAIL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_UTILS_H

#include "../../utils/matmul_type_def.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <class INPUT_TYPE>
__aicore__ inline constexpr TPosition GetQuePos()
{
    if constexpr (HasScalePosition<INPUT_TYPE>::value && IsScaleTag<INPUT_TYPE>()) {
        return INPUT_TYPE::scalePosition;
    }
    return INPUT_TYPE::pos;
}

template <class INPUT_TYPE>
__aicore__ inline constexpr TPosition GetQuePhyPos()
{
    if constexpr (IsScaleTag<INPUT_TYPE>()) {
        return INPUT_TYPE::TAG == InputTypeTag::scaleA ? TPosition::A1 : TPosition::B1;
    }
    return INPUT_TYPE::TAG == InputTypeTag::A ? TPosition::A1 : TPosition::B1;
}

template <class INPUT_TYPE>
struct CubeInQueType {
#if __CCE_AICORE__ == 220
    using QUE = TQueBind<TPosition::GM, INPUT_TYPE::TAG == InputTypeTag::A ? TPosition::A1 : TPosition::B1, QUEUE_DEPTH,
        GetNdNzMask(CubeFormat::NZ, INPUT_TYPE::format)>;
#else
    using QUE = TQueBind<GetQuePos<INPUT_TYPE>(), GetQuePhyPos<INPUT_TYPE>(), QUEUE_DEPTH,
        GetNdNzMask(CubeFormat::NZ, INPUT_TYPE::format)>;
#endif
};

constexpr int32_t DOUBLE_QUE = 2;
constexpr int32_t SINGLE_QUE = 1;
constexpr int32_t BANK_CONFLICT_SIZE = 512;

enum class CubeInBufferType : uint8_t {
    NONE = 0,
    NORMAL = 1,
    SINGLE_BUFFER,
    DOUBLE_BUFFER,
    SINGLE_GLOBAL_BUFFER,
    DOUBLE_GLOBAL_BUFFER,
    DOUBLE_BUFFER_SPARSE,
    NORMAL_MX,
    DOUBLE_BUFFER_MX,
};

template <typename INPUT_TYPE, const auto& MM_CFG>
__aicore__ inline constexpr bool IsSetSingleGlobalQue()
{
    return INPUT_TYPE::ibShare && !ToMatmulConfig(MM_CFG).enableDoubleCache;
}

template <typename INPUT_TYPE, const auto& MM_CFG>
__aicore__ inline constexpr bool IsSetDoubleGlobalQue()
{
    return INPUT_TYPE::ibShare && ToMatmulConfig(MM_CFG).enableDoubleCache;
}

template <typename INPUT_TYPE, const auto& MM_CFG>
__aicore__ inline constexpr bool IsSetNoDB()
{
    return IsBasic(MM_CFG) || (INPUT_TYPE::TAG == InputTypeTag::B && ToMatmulConfig(MM_CFG).intraBlockPartSum)
           || (INPUT_TYPE::layout != LayoutMode::NONE
               && ToMatmulConfig(MM_CFG).batchMode != BatchMode::SINGLE_LARGE_THAN_L1);
}

template <typename INPUT_TYPE, const auto& MM_CFG>
__aicore__ inline constexpr CubeInBufferType GetCubeInBufferType()
{
    if constexpr (InputPhyPosIsL1<INPUT_TYPE>()) {
        return CubeInBufferType::NONE;
    } else if constexpr (MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1Singleshape() && InputPhyPosIsUB<INPUT_TYPE>()) {
        return CubeInBufferType::SINGLE_BUFFER;
    } else if constexpr (DoMatmulIBShareNorm(MM_CFG)) {
        if constexpr (IsSetDoubleGlobalQue<INPUT_TYPE, MM_CFG>()) {
            return CubeInBufferType::DOUBLE_GLOBAL_BUFFER;
        } else if (IsSetSingleGlobalQue<INPUT_TYPE, MM_CFG>()) {
            return CubeInBufferType::SINGLE_GLOBAL_BUFFER;
        } else {
            return CubeInBufferType::NORMAL;
        }
    } else if constexpr (DoMatmulNorm(MM_CFG)) {
        if constexpr (IsSetNoDB<INPUT_TYPE, MM_CFG>()) {
            return CubeInBufferType::SINGLE_BUFFER;
        } else if (IsScaleTag<INPUT_TYPE>()) {
            return CubeInBufferType::NORMAL_MX;
        } else {
            return CubeInBufferType::NORMAL;
        }
    } else if constexpr (DoMatmulMDL(MM_CFG) || DoMatmulSpecialMDL(MM_CFG)) {
        if constexpr (HasSparseIndex<INPUT_TYPE>()) {
            return CubeInBufferType::DOUBLE_BUFFER_SPARSE;
        } else if (IsScaleTag<INPUT_TYPE>()) {
            return CubeInBufferType::DOUBLE_BUFFER_MX;
        } else {
            return CubeInBufferType::DOUBLE_BUFFER;
        }
    } else if constexpr (DoMatmulBasicBlock(MM_CFG) || DoMatmulSpecialBasicBlock(MM_CFG)) {
        return CubeInBufferType::NORMAL;
    } else {
        return CubeInBufferType::NONE;
    }
}

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // _CUBE_IN_BUFFER_UTILS_H_