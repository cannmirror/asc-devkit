/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantize_tiling.h
 * \brief
 */
#ifndef AICORE_ADV_API_QUANTIZATION_QUANTIZE_TILING_H
#define AICORE_ADV_API_QUANTIZATION_QUANTIZE_TILING_H
#include <cstdint>

#include "graph/tensor.h"
namespace AscendC {
/*
 * @ingroup GetQuantizeMaxMinTmpSize
 * @brief This interface is used to obtain the maximum and minimum temporary space reserved or applied.
 *  The developer selects a proper space size based on this range as the tiling parameter.
 * @param [in] srcShape : input src Tensor shape
 * @param [in] typeSize : src tensor dtype size
 * @param [out] maxValue: max temporary local space size
 * @param [out] minValue: min temporary local space size
 */
void GetQuantizeMaxMinTmpSize(
    const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue);
} // namespace AscendC
#endif // AICORE_ADV_API_QUANTIZATION_QUANTIZE_TILING_H