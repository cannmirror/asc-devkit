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
 * \file broadcast_host.h
 * \brief
 */
#ifndef ATVC_BROADCAST_HOST_H
#define ATVC_BROADCAST_HOST_H
#include <vector>
#include "common/atvc_opdef.h"
#include "common/const_def.h"
#include "common/dtype_utils.h"
#include "broadcast/common/broadcast_common.h"
#include "broadcast/tiling/broadcast_tiling.h"
#include "register/op_def_registry.h"

namespace ATVC {
namespace Host {
void PrintParam(BroadcastPolicy* policy, BroadcastParam* param)
{
    printf("[Broadcast] Tiling result: A0 = %lu\n", param->tilingData.A0);
    printf("[Broadcast] Tiling result: A11 = %lu\n", param->tilingData.A11);
    printf("[Broadcast] Tiling result: A12 = %lu\n", param->tilingData.A12);
    printf("[Broadcast] Tiling result: A2 = %lu\n", param->tilingData.A2);
    printf("[Broadcast] Tiling result: B0 = %lu\n", param->tilingData.B0);
    printf("[Broadcast] Tiling result: B1 = %lu\n", param->tilingData.B1);
    printf("[Broadcast] Tiling result: B2 = %lu\n", param->tilingData.B2);
    printf("[Broadcast] Tiling result: coreNum = %d\n", param->tilingData.coreNum);
    printf("[Broadcast] Tiling result: basicBlock = %lu\n", param->tilingData.basicBlock);
    printf("[Broadcast] Tiling result: factorACntPerCore = %lu\n", param->tilingData.factorACntPerCore);
    printf("[Broadcast] Tiling result: factorATotalCnt = %lu\n", param->tilingData.factorATotalCnt);
    printf("[Broadcast] Tiling result: factorBCntPerCore = %lu\n", param->tilingData.factorBCntPerCore);
    printf("[Broadcast] Tiling result: factorBTotalCnt = %lu\n", param->tilingData.factorBTotalCnt);
    for (int32_t i = 0; i < ATVC::CONST2; i++) {
        printf("[Broadcast] Tiling result: shape[%d] = %lu\n", i, param->tilingData.shape[i]);
        printf("[Broadcast] Tiling result: dstShape[%d] = %lu\n", i, param->tilingData.dstShape[i]);
    }
    printf("[Broadcast] Tiling result: policy.patternID = %d\n", policy->patternID);
    printf("[Broadcast] Tiling result: workspaceSize = %u\n", param->workspaceSize);
    return;
}

template<class OpTraits>
bool CalcBroadcastTiling(std::vector<int64_t> shapeIn,
                         std::vector<int64_t> shapeOut,
                         BroadcastPolicy* policy,
                         BroadcastParam* param)
{
    if(policy == nullptr || param == nullptr) {
        printf("[ERROR] Invalid input: policy or param is null pointer!\n");
        return false;
    }
    struct BroadcastTilingHyperParam {
        int32_t basicBlock = 16 * 1024;  // 最大为UB的总大小的1/3
        int nBufferNum = 2;
    };
    using inputDTypeList = typename OpTraits::In::types;
    using DataType = typename ATVC::TypeListGet<inputDTypeList, 0>::Type;
    auto inputDtype = GetOriInputType<DataType>();
    BroadcastTilingInputParam opInput = {shapeIn, shapeOut, inputDtype};
    OpTiling::BroadcastOpTiling tiling(opInput, policy, param);
    if (!tiling.Run()) {
        printf("[ERROR] Tiling Error\n");
        return false;
    }
    PrintParam(policy, param);
    return true;
};
} // Host
} // ATVC
#endif // ATVC_BROADCAST_HOST_H