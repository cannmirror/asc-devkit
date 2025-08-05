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
#ifndef ATVC_REDUCE_HOST_H
#define ATVC_REDUCE_HOST_H
#include "common/atvc_opdef.h"
#include "common/dtype_utils.h"
#include "common/const_def.h"
#include "reduce/common/reduce_common.h"
#include "reduce/tiling/reduce_tiling.h"
#include "register/op_def_registry.h"
#include "reduce/tiling/tiling_common.h"

namespace ATVC {
namespace Host {

void PrintParam(ReducePolicy* policy, ReduceParam* param)
{
    printf("[Reduce] Tiling result: factorACntPerCore = %zu\n", param->tilingData.factorACntPerCore);
    printf("[Reduce] Tiling result: factorATotalCnt = %zu\n", param->tilingData.factorATotalCnt);
    printf("[Reduce] Tiling result: ubFactorA = %zu\n", param->tilingData.ubFactorA);
    printf("[Reduce] Tiling result: factorRCntPerCore = %zu\n", param->tilingData.factorRCntPerCore);
    printf("[Reduce] Tiling result: factorRTotalCnt = %zu\n", param->tilingData.factorRTotalCnt);
    printf("[Reduce] Tiling result: ubFactorR = %zu\n", param->tilingData.ubFactorR);
    printf("[Reduce] Tiling result: groupR = %zu\n", param->tilingData.groupR);
    printf("[Reduce] Tiling result: outSize = %zu\n", param->tilingData.outSize);
    printf("[Reduce] Tiling result: basicBlock = %zu\n", param->tilingData.basicBlock);
    printf("[Reduce] Tiling result: coreNum = %d\n", param->tilingData.coreNum);
    printf("[Reduce] Tiling result: nBufferNum = %d\n", param->nBufferNum);
    printf("[Reduce] Tiling result: workspaceSize = %u\n", param->workspaceSize);
    printf("[Reduce] Tiling result: policy = (%d, %d, %d)\n",
        policy->patternID, policy->loopARCount, policy->loopInnerARCount);
    return;
}

/**
 * @brief 计算Reduce的TilingData和策略参数
 * @param inputShape 输入张量的形状。
 * @param reduceDim 需要进行Reduce操作的具体维度。
 * @param policy 输出参数。
 * @param param 输出参数。
 * @return bool 返回true表示计算成功，false表示失败。
 */
template<class OpTraits>
bool CalcReduceTiling(std::vector<int64_t> inputShape,
                      std::vector<int64_t> reduceDim, ReducePolicy *policy,
                      ReduceParam *param) 
{
    if (policy == nullptr || param == nullptr) {
        printf("[ERROR] Invalid input: policy or param is null pointer!\n");
        return false;
    }
    struct ReduceTilingHyperParam {
        int32_t basicBlock = 16 * 1024;  // 最大为UB的总大小的1/3
        int32_t nBufferNum = 2;          // 每个Queue中的Tensor数量
    };
    using inputDTypeList = typename OpTraits::In::types;
    static constexpr size_t inTensorSumBytes =
        ATVC::TypeListReduce<inputDTypeList, SizeValue<0>, SumSizes>::Type::value;
    if (inTensorSumBytes == 0) {
        printf("[ERROR] Tiling Error: OpTraits Input cannot be null!\n");
        return false;
    }
    using DataType = typename ATVC::TypeListGet<inputDTypeList, 0>::Type;
    auto inputDtype = GetOriInputType<DataType>();
    // check output
    using outputDTypeList = typename OpTraits::Out::types;
    using DataTypeOut = typename ATVC::TypeListGet<outputDTypeList, 0>::Type;
    if (GetOriInputType<DataTypeOut>() != inputDtype) {
        printf("[ERROR] Reduce template does not support different input/output data types!\n");
        return false;
    }
    if (inputDtype == ge::DataType::DT_UNDEFINED) {
        printf("[ERROR] Reduce template does not support this data type!\n");
        return false;
    }
    OpTiling::ReduceTilingInputParam opInput = {reduceDim, inputShape, inputDtype, GetPromoteDataType(inputDtype)};
    OpTiling::ReduceOpTiling tiling(opInput, policy, param);
    if (tiling.Run() != 0) {
        printf("[ERROR] Tiling Error\n");
        return false;
    }
    PrintParam(policy, param);
    return true;
};
} // Host
} // ATVC
#endif // ATVC_REDUCE_HOST_H