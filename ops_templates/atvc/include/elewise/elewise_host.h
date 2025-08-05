/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file elewise_host.h
 * \brief
 */

#ifndef ATVC_ELEWISE_HOST_H
#define ATVC_ELEWISE_HOST_H
#include <vector>
#include <cstdint>
#include "common/compile_info.h"
#include "common/atvc_opdef.h"
#include "elewise/common/elewise_common.h"
#include "register/op_def_registry.h"

namespace ATVC {
namespace Host {
namespace {
constexpr uint32_t basicCntMin = 32; // 基本块最小值
constexpr uint32_t TILE_CONTROL = 2;

struct EleWiseTilingHyperParam {
    int32_t basicCntBase = 16 * 1024;   // 基块单次搬运的初始元素个数，推荐在[1K, 32K]的范围内变动
    int32_t nBufferNum = 2;             // 每个Queue中的Tensor数量
};

int32_t GetEleWiseBasicCnt(const EleWiseTilingHyperParam &hiperParam,
                           int32_t totalCnt, uint32_t blockNum, uint32_t ubufLimitCnt)
{
    uint32_t basicCnt = hiperParam.basicCntBase; // 基本块初始值
    if (blockNum == 0) {
        return basicCnt;
    }
    uint32_t avgElePerBlock = totalCnt / blockNum;
    if (avgElePerBlock > basicCnt) {
        basicCnt = avgElePerBlock  / 128 * 128; // 128 向下对齐
        if (basicCnt > basicCntMin) {
            basicCnt = basicCnt / hiperParam.nBufferNum; // 乒乓搬运avgCoreCnt的1/bufferNum数据
        }
    } else {
        while ((basicCnt > avgElePerBlock) && (basicCnt > basicCntMin)) {
            basicCnt = basicCnt / hiperParam.nBufferNum;
        }
    }
    if (basicCnt > ubufLimitCnt) {
        basicCnt = ubufLimitCnt / basicCntMin * basicCntMin;
    }
    return basicCnt;
}

void PrintParam(const ATVC::EleWiseParam &param)
{
    printf("[EleWise] Tiling Result: tiledCnt = %d\n", param.tilingData.tiledCnt);
    printf("[EleWise] Tiling Result: tailBlockCnt = %d\n", param.tilingData.tailBlockCnt);
    printf("[EleWise] Tiling Result: blockNum: = %d\n", param.tilingData.blockNum);
    printf("[EleWise] Tiling Result: numPerBlock = %d\n", param.tilingData.numPerBlock);
    printf("[EleWise] Tiling Result: tailElemCnt = %d\n", param.tilingData.tailElemCnt);
    printf("[EleWise] Tiling Result: nBufferNum = %d\n", param.nBufferNum);
    return;
}
}
/**
 * @brief 计算EleWise的EleWiseParam运行态参数
 * @param totalCnt 单个输入的总元素个数
 * @param param 输出参数。
 * @return bool 返回true表示计算成功，false表示失败。
 */
template <class OpTraits>
bool CalcEleWiseTiling(int32_t totalCnt, ATVC::EleWiseParam &param)
{
    EleWiseTilingHyperParam hiperParam;
    using Inputs = typename OpTraits::In::types;
    using Outputs = typename OpTraits::Out::types;
    using Temps = typename OpTraits::Temp::types;
    // xxTensroSumBytes表示TensorList里面所有数据类型长度的累加值， xxTensroSumBytes = sum(sizeof(Tensor_i::type))
    static constexpr size_t inTensorSumBytes = ATVC::TypeListReduce<Inputs, SizeValue<0>, SumSizes>::Type::value;
    static constexpr size_t outTensorSumBytes = ATVC::TypeListReduce<Outputs, SizeValue<0>, SumSizes>::Type::value;
    static constexpr size_t tempTensorSumBytes = ATVC::TypeListReduce<Temps, SizeValue<0>, SumSizes>::Type::value;
    if (inTensorSumBytes == 0 || outTensorSumBytes == 0) {
        printf("[ERROR] Tiling Error: OpTraits input cannot be null!\n");
        return false;
    }
    auto compileInfo = ATVC::GetOpCompileInfo();
    uint32_t ubSize = compileInfo.ubSize;
    uint32_t blockNum = (totalCnt / hiperParam.basicCntBase == 0) ? 1 : totalCnt / hiperParam.basicCntBase;
    if (blockNum > compileInfo.vectorCoreNum) {
        blockNum = compileInfo.vectorCoreNum;
    }
    uint32_t ubufLimitCnt = ubSize / ((inTensorSumBytes + outTensorSumBytes) * hiperParam.nBufferNum +
                            tempTensorSumBytes) / TILE_CONTROL;
    if (tempTensorSumBytes == 0) {
        // 未声明tempbuffer时，预留 1 / (bufferNum + 1)的空间给AscendC高阶API内部临时空间使用
        ubufLimitCnt = ubSize / ((inTensorSumBytes + outTensorSumBytes) * (hiperParam.nBufferNum + 1));
    }

    int32_t basicCnt = GetEleWiseBasicCnt(hiperParam, totalCnt, blockNum, ubufLimitCnt);
    if (basicCnt == 0 || blockNum == 0) {
        printf("[ERROR] Tiling Error: initial basic count and block number cannot be zero!\n");
        return false;
    }
    param.tilingData.tiledCnt = basicCnt;
    param.totalCnt = totalCnt;
    uint32_t totalCopyCnt = totalCnt / basicCnt;
    param.tilingData.tailBlockCnt = (totalCopyCnt) % blockNum;
    param.tilingData.blockNum = blockNum;
    param.tilingData.numPerBlock = totalCopyCnt / blockNum; // 每个block要搬运的基本块数量
    param.tilingData.tailElemCnt = totalCnt % basicCnt; // 尾块元素个数
    param.nBufferNum = hiperParam.nBufferNum;
    PrintParam(param);
    return true;
};

}
} // namespace ATVC
#endif // ATVC_ELEWISE_HOST_H