/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file antiquantize_tiling_impl.cpp
 * \brief
 */
#include "graph/tensor.h"
#include "graph/types.h"
#include "quantization/antiquantize_tiling.h"

namespace AscendC {
namespace {
constexpr uint32_t ANTI_QUANT_MIN_TMP_SIZE = 1024;
constexpr uint32_t ASCEND_ANTIQUANT_TWO = 2;
constexpr uint32_t ASCEND_ANTIQUANT_SINGE_N_SIZE = 64;

uint32_t GetAntiQuantizeTmpSizeOfFp4(const ge::Shape& scaleShape, bool isTranspose)
{
    (void)isTranspose;
    (void)scaleShape;
    return 0;
}

uint32_t GetAntiQuantizeScaleSize(const ge::Shape& scaleShape)
{
    auto shapeDims = scaleShape.GetDims();
    uint32_t scaleSize = 1;
    for (uint32_t i = 0; i < shapeDims.size(); i++) { scaleSize *= shapeDims[i]; }
    return scaleSize;
}

uint32_t GetAntiQuantizeMaxTmpSize(const ge::Shape& srcShape, const ge::Shape& scaleShape, bool isTranspose,
    ge::DataType inputDataType, ge::DataType outputDataType)
{
    if (inputDataType == ge::DT_FLOAT4_E2M1 || inputDataType == ge::DT_FLOAT4_E1M2) {
        return GetAntiQuantizeTmpSizeOfFp4(scaleShape, isTranspose);
    }
    if (outputDataType == ge::DT_FLOAT16) {
        return 0;
    }

    uint32_t scaleSize = GetAntiQuantizeScaleSize(scaleShape);
    auto shapeDims = srcShape.GetDims();
    uint32_t srcSize = 1;
    for (uint32_t i = 0; i < shapeDims.size(); i++) { srcSize *= shapeDims[i]; }
    bool isPerChannel = (scaleSize == 1) ? false : true;

    if (isPerChannel) {
        uint32_t k = srcShape.GetDims()[0];
        return scaleSize * ASCEND_ANTIQUANT_TWO * sizeof(float) + ASCEND_ANTIQUANT_SINGE_N_SIZE * k * sizeof(float);
    } else {
        return srcSize * sizeof(float);
    }
}

uint32_t GetAntiQuantizeMinTmpSize(const ge::Shape& srcShape, const ge::Shape& scaleShape, bool isTranspose,
    ge::DataType inputDataType, ge::DataType outputDataType)
{
    if (inputDataType == ge::DT_FLOAT4_E2M1 || inputDataType == ge::DT_FLOAT4_E1M2) {
        return GetAntiQuantizeTmpSizeOfFp4(scaleShape, isTranspose);
    }
    if (outputDataType == ge::DT_FLOAT16) {
        return 0;
    }

    uint32_t scaleSize = GetAntiQuantizeScaleSize(scaleShape);
    bool isPerChannel = (scaleSize == 1) ? false : true;
    if (!isPerChannel) {
        return ANTI_QUANT_MIN_TMP_SIZE;
    }

    auto shapeDims = srcShape.GetDims();
    uint32_t k = srcShape.GetDims()[0];
    return scaleSize * ASCEND_ANTIQUANT_TWO * sizeof(float) + ASCEND_ANTIQUANT_SINGE_N_SIZE * k * sizeof(float);
}
} // namespace

void GetAntiQuantizeMaxMinTmpSize(const ge::Shape& srcShape, const ge::Shape& scaleShape, bool isTranspose,
    ge::DataType inputDataType, ge::DataType outputDataType, uint32_t& maxValue, uint32_t& minValue)
{
    maxValue = GetAntiQuantizeMaxTmpSize(srcShape, scaleShape, isTranspose, inputDataType, outputDataType);
    minValue = GetAntiQuantizeMinTmpSize(srcShape, scaleShape, isTranspose, inputDataType, outputDataType);
}
} // namespace AscendC
