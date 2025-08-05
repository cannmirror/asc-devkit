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

#ifndef ATVC_BROADCAST_COMPUTE_H
#define ATVC_BROADCAST_COMPUTE_H
#include "kernel_operator.h"
#include "common/kernel_utils.h"
#include "broadcast/common/broadcast_common.h"

namespace ATVC {
template<class OpTraits>
class BroadcastCompute {
public:
    using inputDTypeList = typename OpTraits::In::types;
    using DataType = typename ATVC::TypeListGet<inputDTypeList, 0>::Type;

    template <int32_t patternID>
    __aicore__ inline void Compute(AscendC::LocalTensor<DataType> &src,
                                   uint32_t inputOffset,
                                   AscendC::LocalTensor<DataType> &dst,
                                   uint32_t dimA,
                                   uint32_t dimB)
    {
        if (patternID == ATVC::AB_PATTERN::ABA) {
            ComputeBA(src, inputOffset, dst, dimA, dimB);
        } else {
            ComputeAB(src, inputOffset, dst, dimA, dimB);
        }
    }

private:
    __aicore__ inline void ComputeBAByDataCopy(AscendC::LocalTensor<DataType> &src,
                                               uint32_t inputOffset,
                                               AscendC::LocalTensor<DataType> &dst,
                                               uint32_t dimA,
                                               uint32_t dimB)
    {
        AscendC::DataCopy(dst, src[inputOffset], dimA);
        uint32_t i = 1;
        uint32_t cnt = 1;
        while(i < dimB) {
            cnt = i > (dimB - i) ? (dimB - i) : i;
            AscendC::DataCopy(dst[dimA * i], dst, dimA * cnt);
            i += cnt;
        }
    }

    __aicore__ inline void ComputeBA(AscendC::LocalTensor<DataType> &src,
                                     uint32_t inputOffset,
                                     AscendC::LocalTensor<DataType> &dst,
                                     uint32_t dimA,
                                     uint32_t dimB)
    {
        /*
        X1 X2 X3 X4
        ->
        X1 X2 X3 X4
        X1 X2 X3 X4
        */
        ComputeBAByDataCopy(src, inputOffset, dst, dimA, dimB);
    }

    __aicore__ inline void ComputeABByBrcbCopy(AscendC::LocalTensor<DataType> &src,
                                               uint32_t inputOffset,
                                               AscendC::LocalTensor<DataType> &dst,
                                               uint32_t dimA,
                                               uint32_t dimB)
    {
        uint32_t brcbProcCnt = 8; // 一次brcb 处理8个元素
        uint32_t dSize = sizeof(DataType);
        AscendC::BrcbRepeatParams repeatParam(dimB * dSize / ATVC::UB_ALIGN_32,
                                              brcbProcCnt * dimB * dSize / ATVC::UB_ALIGN_32);
        AscendC::Brcb(dst, src[inputOffset], dimA / brcbProcCnt, repeatParam);
        uint32_t i = brcbProcCnt;
        uint16_t step;
        while (i < dimB) {
            step = i * 2 > dimB ? (dimB - i) : i; // 2: 每次循环 将已拷贝长度为i的元素拷贝到下一个dst，要保证不超出dimB
            step = step * dSize / ATVC::UB_ALIGN_32;
            uint16_t stride = (uint16_t)(dimB * dSize / ATVC::UB_ALIGN_32 - step);
            AscendC::DataCopyParams repeatParam = {
                (uint16_t)dimA, // blockCount [1, 4095]
                step,           // 单位为32B
                stride,         // 取值范围不能超uint16_t
                stride};        // 取值范围不能超uint16_t
            AscendC::DataCopy(dst[i], dst, repeatParam);
            i = i + step * ATVC::UB_ALIGN_32 / dSize;
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeAB(AscendC::LocalTensor<DataType> &src,
                                     uint32_t inputOffset,
                                     AscendC::LocalTensor<DataType> &dst,
                                     uint32_t dimA,
                                     uint32_t dimB)
    {
        /*
        X1 
        X2 
        X3 
        X4
        ->
        X1 X1
        X2 X2
        X3 X3
        X4 X4
        */
        ComputeABByBrcbCopy(src, inputOffset, dst, dimA, dimB);
    }
};
}
#endif // ATVC_BROADCAST_COMPUTE_H