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

#ifndef ATVC_REDUCE_SUM_H
#define ATVC_REDUCE_SUM_H

#include "common/kernel_utils.h"
#include "reduce/common/patterns.h"
#include "reduce/reduce_utils/reduce_block_aux_util.h"

namespace {
struct ReduceARParam {
    uint32_t repStride = 0;
    uint16_t dimA = 0;
    uint16_t dimMax = 0;
    uint16_t mainR = 0;
    uint16_t tailR = 0;
    uint64_t maskAddRNum = 0;
    uint16_t loopRNum = 0;
    uint16_t dtypeSize = 0;
    uint16_t dimR = 0;
};
}


namespace ATVC {
// OpTraits: 算子描述的ATVC::OpTraits结构体
template <typename OpTraits>
// 计算模板，不感知数据从GM到UB上的搬运
class ReduceSumCompute {
public:
    // 从OpTraits中萃取算子输入描述信息
    using inputDTypeList = typename OpTraits::In::types;
    using DataType = typename ATVC::TypeListGet<inputDTypeList, 0>::Type;
    using PrompteDtype = typename KernelUtils::GetPromoteType<DataType>::T;
    __aicore__ inline ReduceSumCompute() {}

    template <bool needMask, class Pattern>
    __aicore__ inline void
    Compute(KernelUtils::Shape<2> &shape,
            const AscendC::LocalTensor<PrompteDtype> &dst,
            const AscendC::LocalTensor<PrompteDtype> &src)
    {
        // AR场景，硬件限制，R轴需要做UB上32B对齐，对齐方式有2种：
        // 1. 高性能对齐(补充元素值不确定), 后续累加计算只能计算实际有效的元素个数
        // 2. 补0对齐(补值是由用户实现的GetPaddingValue()接口决定的）
        if (std::is_same<Pattern, ReducePattern::AR>::value) {
            if constexpr (needMask) { // 1. 高性能对齐模式
                // MainR(int64_t dimR, bool isAR)： 框架提供的计算R轴二分长度(元素个数)， dimR为原始的元素个数
                int16_t mainR = KernelUtils::Reduce::MainR(shape.oriBurstLen, true);
                ReduceAR(dst, src, shape.value[0], shape.value[1],  mainR, shape.oriBurstLen);
            } else {
                // MainR：框架提供的计算R轴二分长度(元素个数)，dimR为补齐后的元素个数
                int16_t mainR = KernelUtils::Reduce::MainR(shape.value[1], true);
                ReduceAR(dst, src, shape.value[0], shape.value[1],  mainR, shape.value[1]);
            }
        }
        if (std::is_same<Pattern, ReducePattern::RA>::value) {
            int16_t mainR = KernelUtils::Reduce::MainR(shape.value[0], false);
            ReduceRA(dst, src, shape.value[1], shape.value[0], mainR);
        }
    }

    __aicore__ inline void
    ReduceRA(const AscendC::LocalTensor<PrompteDtype> &dst,
             const AscendC::LocalTensor<PrompteDtype> &src, uint16_t dimA,
             uint16_t dimR, uint16_t mainR)
    {
        uint32_t totalNum = dimR * dimA;
        uint32_t mainNum = dimA * mainR;
        uint32_t dtypeSize = sizeof(PrompteDtype);
        uint32_t tailNum = totalNum - mainNum;
        // add mask最大值为256 bytes 且要满足32bytes对齐
        uint32_t maskAddNum = UB_ALIGN_256 / dtypeSize / UB_ALIGN_32 * UB_ALIGN_32;
        // 处理tail
        uint16_t repeatTimes = tailNum / maskAddNum;
        uint16_t repeatNum = repeatTimes * maskAddNum;
        uint16_t repTailNum = tailNum - repeatNum;
        uint32_t repStride = dtypeSize * maskAddNum / UB_ALIGN_32; // 不同迭代间同一datablock步长
        // dstBlkStride, src0BlkStride,src1BlkStride, dstRepStride, src0RepStride, src1RepStride
        AscendC::BinaryRepeatParams repeatParams(1, 1, 1, repStride, repStride, repStride);
        if (repeatTimes > 0) {
            AscendC::Add(src, src[mainNum], src, maskAddNum, repeatTimes, repeatParams);
        }
        if (repTailNum > 0) {
            repStride = dtypeSize * repTailNum / UB_ALIGN_32; // 不同迭代间同一datablock步长
            repeatParams.dstRepStride = repStride;
            repeatParams.src0RepStride = repStride;
            repeatParams.src1RepStride = repStride;
            AscendC::Add(src[repeatNum], src[repeatNum + mainNum], src[repeatNum], repTailNum, 1, repeatParams);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // 二分主体
        uint16_t loopRNum = mainR;
        while (loopRNum > 1) {
            loopRNum = loopRNum >> 1;
            mainNum = loopRNum * dimA; // LoopR的前半部分数据量
            repeatTimes = mainNum / maskAddNum;
            repeatNum = repeatTimes * maskAddNum;
            repTailNum = mainNum - repeatNum;
            if (repeatTimes > 0) {
                repStride = dtypeSize * maskAddNum / UB_ALIGN_32; // 不同迭代间同一datablock步长
                repeatParams.dstRepStride = repStride;
                repeatParams.src0RepStride = repStride;
                repeatParams.src1RepStride = repStride;
                AscendC::Add(src, src[mainNum], src, maskAddNum, repeatTimes, repeatParams);
            }
            if (repTailNum > 0) {
                repStride = dtypeSize * repTailNum / UB_ALIGN_32; // 不同迭代间同一datablock步长
                repeatParams.dstRepStride = repStride;
                repeatParams.src0RepStride = repStride;
                repeatParams.src1RepStride = repStride;
                AscendC::Add(src[repeatNum], src[repeatNum + mainNum], src[repeatNum], repTailNum, 1, repeatParams);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::DataCopy(dst, src, dimA);
    }

    __aicore__ inline void
    ReduceAR(const AscendC::LocalTensor<PrompteDtype> &dstTensor,
             const AscendC::LocalTensor<PrompteDtype> &srcTensor, uint16_t dimA,
             uint16_t dimR, uint16_t mainR, uint64_t oriBurstLen)
    {
        uint16_t tailR = oriBurstLen - mainR;
        uint16_t dtypeSize = sizeof(PrompteDtype);
        uint32_t repStride = dtypeSize * dimR / UB_ALIGN_32;
        uint16_t dimMax = dimA * dimR;
        uint64_t maskAddRNum = UB_ALIGN_256 / dtypeSize;

        ReduceARParam param{
            .repStride   = repStride,
            .dimA        = dimA,
            .dimMax      = dimMax,
            .mainR       = mainR,
            .tailR       = tailR,
            .maskAddRNum = maskAddRNum,
            .dtypeSize   = dtypeSize,
            .dimR        = dimR
        };

        if (mainR > 0 && tailR > 0) {
            PerformInitialAdd(srcTensor, param);
        }

        // 二分计算
        param.loopRNum = mainR;
        while (param.loopRNum > maskAddRNum) {
            param.loopRNum = param.loopRNum / 2; // 除2二分
            PerformBinaryReduction(srcTensor, param);
        }
        if (param.loopRNum == 0) { // small shape, 直接reduce
            param.loopRNum = tailR;
        }
        PerformFinalReduction(dstTensor, srcTensor, param);
    }

    template <class Pattern, class V>
    __aicore__ inline void UpdateCache(int64_t index, V& shape, const AscendC::LocalTensor<PrompteDtype>& tempBuf,
                                       const AscendC::LocalTensor<PrompteDtype>& computeRes)
    {
        int64_t cacheID = KernelUtils::Reduce::GetCacheID(index);
        int64_t dimA = Pattern::TailA ? shape.value[1] : shape.value[0];
        int32_t element_one_repeat = Platform::GetVRegSize() / sizeof(PrompteDtype);
        int64_t stride = OpsUtils::CeilDiv(dimA, static_cast<int64_t>(element_one_repeat)) * element_one_repeat;
        // count A轴的大小 * VL
        uint16_t outerLoopTimes = OpsUtils::CeilDiv(
            static_cast<int64_t>(dimA * sizeof(PrompteDtype)), static_cast<int64_t>(Platform::GetVRegSize()));
        uint16_t innerLoopTimes = cacheID;
        uint32_t outerLoopStride = element_one_repeat;
        uint32_t innerLoopStride = stride;  // cacahe的每一个idex的块的大小， A轴的大小
        AscendC::LocalTensor<PrompteDtype> dstTensor = tempBuf;
        AscendC::LocalTensor<PrompteDtype> srcTensor = computeRes;
        uint32_t cah = cacheID * stride;

        for (uint16_t i = 0; i < outerLoopTimes; ++i) {  // outerLoopTimes是dimA的大小
            uint32_t srcIdx = i * outerLoopStride;
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                AscendC::Add(srcTensor[srcIdx], srcTensor[srcIdx],
                             dstTensor[srcIdx + j * innerLoopStride],
                             outerLoopStride);
                AscendC::PipeBarrier<PIPE_V>();
            }
            DataCopy(dstTensor[cah + srcIdx], srcTensor[srcIdx], outerLoopStride);
        }
    }

    __aicore__ inline void
    ReduceBetweenUB(const AscendC::LocalTensor<PrompteDtype> &ubTensorLeft,
                    const AscendC::LocalTensor<PrompteDtype> &ubTensorRight,
                    const int32_t &calCount)
    {
        Add(ubTensorRight, ubTensorRight, ubTensorLeft, calCount);
    }

    template <typename U>
    __aicore__ inline U GetPaddingValue() // 设置框架内每一次搬入UB的数据对齐补充的值
    {
        U paddingValue = 0; // 由于ReduceSum是累加R轴数据，补齐的元素值设为0，才能保证累加的结果不受影响
        return paddingValue;
    }

private:
    __aicore__ inline void PerformInitialAdd(const AscendC::LocalTensor<PrompteDtype> &srcTensor, const ReduceARParam& param)
    {
        uint16_t addRTotalNum = param.tailR / param.maskAddRNum * param.maskAddRNum;
        uint16_t addRTail = param.tailR - addRTotalNum;
        // dstBlkStride, src0BlkStride,src1BlkStride, dstRepStride, src0RepStride, src1RepStride
        AscendC::BinaryRepeatParams repeatParams(1, 1, 1, param.repStride, param.repStride, param.repStride);

        if (param.repStride > UB_ALIGN_255) {
            for (uint16_t i = 0; i < param.dimMax; i += param.dimR) {
                AscendC::Add(srcTensor[i], srcTensor[i], srcTensor[i + param.mainR], param.tailR);
            }
        } else {
            for (uint16_t i = 0; i < addRTotalNum; i += param.maskAddRNum) {
                AscendC::Add(srcTensor[i], srcTensor[i + param.mainR], srcTensor[i], param.maskAddRNum, param.dimA, repeatParams);
            }
            if (addRTail > 0) {
                AscendC::Add(srcTensor[addRTotalNum],
                    srcTensor[addRTotalNum + param.mainR],
                    srcTensor[addRTotalNum],
                    addRTail,
                    param.dimA,
                    repeatParams);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PerformBinaryReduction(const AscendC::LocalTensor<PrompteDtype> &srcTensor,
        const ReduceARParam& param)
    {
        if (param.repStride > UB_ALIGN_255) {
            for (uint16_t i = 0; i < param.dimMax; i += param.loopRNum) {
                AscendC::Add(srcTensor[i], srcTensor[i], srcTensor[i + param.loopRNum], param.loopRNum);
            }
        } else {
            uint16_t addRTotalNum = param.loopRNum / param.maskAddRNum * param.maskAddRNum;
            uint16_t addRTail = param.loopRNum - addRTotalNum;
            // dstBlkStride, src0BlkStride,src1BlkStride, dstRepStride, src0RepStride, src1RepStride
            AscendC::BinaryRepeatParams repeatParams(1, 1, 1, param.repStride, param.repStride, param.repStride);
            for (uint16_t i = 0; i < addRTotalNum; i += param.maskAddRNum) {
                AscendC::Add(srcTensor[i], srcTensor[i + param.loopRNum], srcTensor[i], param.maskAddRNum, param.dimA, repeatParams);
            }
            if (addRTail > 0) {
                AscendC::Add(srcTensor[addRTotalNum],
                    srcTensor[addRTotalNum],
                    srcTensor[addRTotalNum + param.loopRNum],
                    addRTail,
                    param.dimA,
                    repeatParams);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PerformFinalReduction(const AscendC::LocalTensor<PrompteDtype> &dstTensor,
        const AscendC::LocalTensor<PrompteDtype> &srcTensor, const ReduceARParam& param)
    {
        if constexpr (AscendC::IsSameType<PrompteDtype, float>::value ||
                      AscendC::IsSameType<PrompteDtype, half>::value) {
            uint16_t reduceLoopTimes = UB_ALIGN_255 * param.dtypeSize / UB_ALIGN_32 * UB_ALIGN_32 / param.dtypeSize;
            // WholeReduceSum repeattime最大值为255 255附近为了dimA需要分多次
            for (uint16_t dimAIdx = 0; dimAIdx < param.dimA; dimAIdx += reduceLoopTimes) {
                uint16_t curDimA = (dimAIdx + reduceLoopTimes < param.dimA) ? reduceLoopTimes : param.dimA - dimAIdx;
                AscendC::WholeReduceSum(
                    dstTensor[dimAIdx], srcTensor[dimAIdx * param.dimR], param.loopRNum, curDimA, 1, 1, param.repStride);
            }
            AscendC::PipeBarrier<PIPE_V>();
        } else if constexpr (AscendC::IsSameType<PrompteDtype, int32_t>::value ||
                             AscendC::IsSameType<PrompteDtype, uint32_t>::value) {
            // 尽量二分add到最后32bytes
            // int32 -> float 都是4字,一把cast 用CAST_NONE
            AscendC::LocalTensor<float> interpreSrc = srcTensor.template ReinterpretCast<float>();
            AscendC::LocalTensor<float> interpreDst = dstTensor.template ReinterpretCast<float>();
            AscendC::Cast(interpreSrc, srcTensor, AscendC::RoundMode::CAST_NONE, param.dimA * param.dimR);
            AscendC::PipeBarrier<PIPE_V>();

            uint16_t reduceLoopTimes = 255 * param.dtypeSize / UB_ALIGN_32 * UB_ALIGN_32 / param.dtypeSize;
            // WholeReduceSum repeattime最大值为255 255附近为了dimA需要分多次
            for (uint16_t dimAIdx = 0; dimAIdx < param.dimA; dimAIdx += reduceLoopTimes) {
                uint16_t curDimA = (dimAIdx + reduceLoopTimes < param.dimA) ? reduceLoopTimes : param.dimA - dimAIdx;
                AscendC::WholeReduceSum(
                    interpreDst[dimAIdx], interpreSrc[dimAIdx * param.dimR], param.loopRNum, curDimA, 1, 1, param.repStride);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(dstTensor, interpreDst, AscendC::RoundMode::CAST_RINT, dstTensor.GetSize());
        }
    }
};
} // namespace ATVC

#endif // ATVC_REDUCE_SUM_H
