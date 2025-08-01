/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "normalization/layernorm_grad_tiling.h"
#include "graph/tensor.h"
#include "normalization/layernorm_grad_tilingdata.h"
#include "detail/host_log.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
REGISTER_TILING_DATA_CLASS(LayerNormGradTilingOpApi, LayerNormGradTiling);
}
namespace AscendC {
namespace {
constexpr uint32_t LAYERNORM_GRAD_HALF_SIZE = 2;
constexpr uint32_t LAYERNORM_GRAD_FLOAT_SIZE = 4;
constexpr uint32_t LAYERNORM_GRAD_SRC_DIM_NUM = 4;
union LastDimValue {
    float floatValue;
    uint32_t uint32Value;
};
void CheckSrcShape(std::vector<int64_t> shapeDims)
{
    constexpr uint32_t LAYERNORM_GRAD_SHAPE_SIZE = 4;
    ASCENDC_HOST_ASSERT(
        shapeDims.size() >= LAYERNORM_GRAD_SHAPE_SIZE, return, "srcShape dims must not be less than 4.");
    ASCENDC_HOST_ASSERT(shapeDims[0] > 0, return, "srcShape[0] must be greater than 0.");
    ASCENDC_HOST_ASSERT(shapeDims[1] > 0, return, "srcShape[1] must be greater than 0.");
    ASCENDC_HOST_ASSERT(shapeDims[2] > 0, return, "srcShape[2] must be greater than 0.");
    ASCENDC_HOST_ASSERT(shapeDims[3] > 0, return, "srcShape[3] must be greater than 0.");
}

void CheckLayerNormGradHostCommon(
    const char* apiName, const char* hostFuncName, const ge::Shape& srcShape, const uint32_t typeSize)
{
    ASCENDC_HOST_ASSERT(typeSize == LAYERNORM_GRAD_HALF_SIZE || typeSize == LAYERNORM_GRAD_FLOAT_SIZE, return,
        "[%s][%s] Type size %u is unsupported!", apiName, hostFuncName, typeSize);
    ASCENDC_HOST_ASSERT(srcShape.GetShapeSize() > 0, return, "[%s][%s] Input Shape size must be greater than 0.",
        apiName, hostFuncName);
    ASCENDC_HOST_ASSERT(srcShape.GetDimNum() == LAYERNORM_GRAD_SRC_DIM_NUM, return,
        "[%s][%s] The dims of srcShape is %zu, should be 4 (e.g. [B, S, storageHLength, originHLength])!", apiName,
        hostFuncName, srcShape.GetDimNum());
    return;
}
} // namespace

void GetLayerNormGradMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
{
    CheckLayerNormGradHostCommon("LayerNormGrad", "GetLayerNormGradMaxMinTmpSize", srcShape, typeSize);
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    CheckSrcShape(shapeDims);
    uint32_t bLength = shapeDims[0];
    uint32_t sLength = shapeDims[1];
    uint32_t hLength = shapeDims[2];
    uint32_t inputSize = bLength * sLength * hLength;
    uint32_t maxBaseSize = (inputSize > (hLength * hLength)) ? inputSize : (hLength * hLength);
    uint32_t minBaseSize = (inputSize < (hLength * hLength)) ? inputSize : (hLength * hLength);
    if (typeSize == LAYERNORM_GRAD_B16_BYTE_SIZE) {
        maxValue = LAYERNORM_GRAD_HALF_BUF_NUM * maxBaseSize * typeSize;
        minValue = minBaseSize * LAYERNORM_GRAD_HALF_BUF_NUM * typeSize;
        return;
    }
    if (isReuseSource) {
        maxValue = LAYERNORM_GRAD_REUSE_FLOAT_BUF_NUM * maxBaseSize * typeSize;
        minValue = minBaseSize * LAYERNORM_GRAD_REUSE_FLOAT_BUF_NUM * typeSize;
    } else {
        maxValue = LAYERNORM_GRAD_FLOAT_BUF_NUM * maxBaseSize * typeSize;
        minValue = minBaseSize * LAYERNORM_GRAD_FLOAT_BUF_NUM * typeSize;
    }
}

void GetLayerNormGradNDTilingInfo(const ge::Shape srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
    const bool isReuseSource, optiling::LayerNormGradTiling& tiling)
{
    CheckLayerNormGradHostCommon("LayerNormGrad", "GetLayerNormGradNDTilingInfo", srcShape, typeSize);
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    CheckSrcShape(shapeDims);
    uint32_t bLength = shapeDims[0];
    uint32_t sLength = shapeDims[1];
    uint32_t hLength = shapeDims[2];
    uint32_t originalHLength = shapeDims[3];
    uint32_t inputXSize = bLength * sLength * hLength;
    uint32_t needBufferBlock;
    if (typeSize == LAYERNORM_GRAD_B16_BYTE_SIZE) {
        needBufferBlock = LAYERNORM_GRAD_HALF_BUF_NUM;
    } else if (isReuseSource) {
        needBufferBlock = LAYERNORM_GRAD_REUSE_FLOAT_BUF_NUM;
    } else {
        needBufferBlock = LAYERNORM_GRAD_FLOAT_BUF_NUM;
    }
    uint32_t oneCalSize = stackBufferSize * sizeof(uint8_t) / sizeof(float) / needBufferBlock;
    oneCalSize = oneCalSize / hLength * hLength;
    ASCENDC_HOST_ASSERT(oneCalSize > 0, return, "stackBufferSize is not enough.");
    uint32_t nohCalSize = oneCalSize / hLength;
    if (typeSize == LAYERNORM_GRAD_B32_BYTE_SIZE) {
        nohCalSize = (nohCalSize + LAYERNORM_GRAD_B32_DATA_NUM_PER_BLOCK - 1) / LAYERNORM_GRAD_B32_DATA_NUM_PER_BLOCK
                     * LAYERNORM_GRAD_B32_DATA_NUM_PER_BLOCK;
    } else {
        nohCalSize = (nohCalSize + LAYERNORM_GRAD_B16_DATA_NUM_PER_BLOCK - 1) / LAYERNORM_GRAD_B16_DATA_NUM_PER_BLOCK
                     * LAYERNORM_GRAD_B16_DATA_NUM_PER_BLOCK;
    }
    oneCalSize = nohCalSize * hLength;
    uint32_t loopNum = inputXSize / oneCalSize;
    uint32_t tailSize = inputXSize % oneCalSize;
    uint32_t nohTailSize = tailSize / hLength;
    uint32_t tmpTensorBSHPos = 0;
    uint32_t tmpTensorBSHSize = oneCalSize;
    uint32_t pdVarTensorPos = tmpTensorBSHPos + tmpTensorBSHSize;
    uint32_t pdVarTensorSize = oneCalSize;
    uint32_t pdMeanTensorPos = pdVarTensorPos + pdVarTensorSize;
    uint32_t pdMeanTensorSize = oneCalSize;
    uint32_t x1TensorPos = 0;
    uint32_t x1TensorSize = oneCalSize;
    uint32_t x2TensorPos = 0;
    uint32_t x2TensorSize = oneCalSize;
    uint32_t x3TensorPos = pdMeanTensorPos + pdMeanTensorSize;
    uint32_t x3TensorSize = oneCalSize;
    if (!(isReuseSource && typeSize == LAYERNORM_GRAD_B32_BYTE_SIZE)) {
        x1TensorPos = pdMeanTensorPos + pdMeanTensorSize;
        x1TensorSize = oneCalSize;
        x2TensorPos = x1TensorPos + x1TensorSize;
        x2TensorSize = oneCalSize;
        x3TensorPos = x2TensorPos + x2TensorSize;
        x3TensorSize = oneCalSize;
    }
    uint32_t tmpTensorPos = 0;
    uint32_t tmpTensorSize = oneCalSize;
    uint32_t tmpTensor1Pos = 0;
    uint32_t tmpTensor1Size = oneCalSize;
    uint32_t tmpTensor2Pos = 0;
    uint32_t tmpTensor2Size = oneCalSize;
    if (typeSize == LAYERNORM_GRAD_B16_BYTE_SIZE) {
        tmpTensorPos = x3TensorPos + x3TensorSize;
        tmpTensorSize = oneCalSize;
        tmpTensor1Pos = tmpTensorPos + tmpTensorSize;
        tmpTensor1Size = oneCalSize;
        tmpTensor2Pos = tmpTensor1Pos + tmpTensor1Size;
        tmpTensor2Size = oneCalSize;
    }
    LastDimValue lastDimValueBack;
    lastDimValueBack.floatValue = 1.0;
    lastDimValueBack.floatValue = lastDimValueBack.floatValue / static_cast<float>(originalHLength);
    LastDimValue lastDimValueBackMulTwo;
    lastDimValueBackMulTwo.floatValue = 2.0;
    lastDimValueBackMulTwo.floatValue = lastDimValueBackMulTwo.floatValue / static_cast<float>(originalHLength);
    tiling.set_stackBufferSize(stackBufferSize);
    tiling.set_bLength(bLength);
    tiling.set_sLength(sLength);
    tiling.set_hLength(hLength);
    tiling.set_originalHLength(originalHLength);
    tiling.set_oneCalSize(oneCalSize);
    tiling.set_nohCalSize(nohCalSize);
    tiling.set_loopNum(loopNum);
    tiling.set_tailSize(tailSize);
    tiling.set_nohTailSize(nohTailSize);
    tiling.set_tmpTensorBSHPos(tmpTensorBSHPos);
    tiling.set_tmpTensorBSHSize(tmpTensorBSHSize);
    tiling.set_pdVarTensorPos(pdVarTensorPos);
    tiling.set_pdVarTensorSize(pdVarTensorSize);
    tiling.set_pdMeanTensorPos(pdMeanTensorPos);
    tiling.set_pdMeanTensorSize(pdMeanTensorSize);
    tiling.set_x1TensorPos(x1TensorPos);
    tiling.set_x1TensorSize(x1TensorSize);
    tiling.set_x2TensorPos(x2TensorPos);
    tiling.set_x2TensorSize(x2TensorSize);
    tiling.set_x3TensorPos(x3TensorPos);
    tiling.set_x3TensorSize(x3TensorSize);
    tiling.set_tmpTensorPos(tmpTensorPos);
    tiling.set_tmpTensorSize(tmpTensorSize);
    tiling.set_tmpTensor1Pos(tmpTensor1Pos);
    tiling.set_tmpTensor1Size(tmpTensor1Size);
    tiling.set_tmpTensor2Pos(tmpTensor2Pos);
    tiling.set_tmpTensor2Size(tmpTensor2Size);
    tiling.set_lastDimValueBack(lastDimValueBack.uint32Value);
    tiling.set_lastDimValueBackMulTwo(lastDimValueBackMulTwo.uint32Value);
}
} // namespace AscendC
