/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_scheduler.h
 * \brief
 */

#ifndef CONV_BLOCK_BLOCK_SCHEDULER_H
#define CONV_BLOCK_BLOCK_SCHEDULER_H

#include "../utils/conv_common_utils.h"

namespace Act {
namespace Conv {
namespace Block {
struct DimDataToFill {
    __aicore__ __forceinline__ DimDataToFill(uint64_t& singleCoreDim_, uint64_t& dimIdxStart_, bool& isDimTail_) :
        singleCoreDim(singleCoreDim_), dimIdxStart(dimIdxStart_), isDimTail(isDimTail_) {}
    uint64_t& singleCoreDim;
    uint64_t& dimIdxStart;
    bool& isDimTail;
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, class mode_, class BlockSchedulerPolicy = void>
struct BlockSchedulerSelector {};

#define BLOCK_CONV_SCHEDULER_CLASS_PARAMS template                                          \
        <class ProblemShape_, class L1TileShape_, class L0TileShape_, class mode_>
#define BLOCK_CONV_SCHEDULER_FUNC_PARAMS ProblemShape_, L1TileShape_, L0TileShape_, mode_

struct IterateMFirst
{
};
struct IterateNFirst
{
};

BLOCK_CONV_SCHEDULER_CLASS_PARAMS
class BlockSchedulerIterate {
    using ProblemShape = ProblemShape_;
public:
    uint64_t singleCoreN;
    uint64_t singleCoreBatch;
    uint64_t singleCoreM;
    uint64_t batchIdxStart;
    uint64_t nIdxStart;
    uint64_t mIdxStart;
    uint64_t fmStartAddr;
    uint64_t weightStartAddr;
    uint64_t outputStartAddr;
    uint64_t biasStartAddr;
    bool isBatchDimTail;
    bool isNDimTail;
    bool isMDimTail;
    int64_t  blockIdx = AscendC::GetBlockIdx();
    ProblemShape convShape;
    ConvDim convDim;
    uint64_t hwIn;
    uint64_t hwOut;
    uint64_t fmapOneBatchSize;
    uint64_t outputOneBatchSize;
    const uint32_t n0 = 16;
    const uint32_t m0 = 16;
    static constexpr int64_t mAL1 = GetIntegralConstant<0, L1TileShape_>();
    static constexpr int64_t nBL1 = GetIntegralConstant<1, L1TileShape_>();

    __aicore__ inline void Init(ProblemShape shape, ConvDim dim)
    {
        convShape = shape;
        convDim = dim;
        InitSingleCoreData(convDim.mDim, 1);
    };
    __aicore__ __forceinline__ bool InitSingleCoreData(uint32_t blockPerNDim, uint32_t blockPerMDim);
    __aicore__ __forceinline__ bool CalcDimData(const uint32_t& blockPerDim, const uint32_t& dim,
                                                const uint64_t& wholeDim, const uint64_t &realWholeDim,
                                                DimDataToFill& curStruct);
    __aicore__ __forceinline__ void CalcStartAddrCommon(const uint32_t din, const uint32_t dout);
    __aicore__ __forceinline__ void CalcStartAddrMMode(const uint32_t din = 1,
                                                       const uint32_t dout = 1, const uint32_t kd = 1);
    __aicore__ __forceinline__ void InterateMax(ConvInterateMax& interateMax, SingleCoreShape& singleCoreShape);
    __aicore__ __forceinline__ uint64_t GetFmapStartAddr() {return fmStartAddr;}
    __aicore__ __forceinline__ uint64_t GetWeightStartAddr() {return weightStartAddr;}
    __aicore__ __forceinline__ uint64_t GetBiasStartAddr() {return biasStartAddr;}
    __aicore__ __forceinline__ uint64_t GetOutputStartAddr() {return outputStartAddr;}
    __aicore__ __forceinline__ uint64_t GetMIdxStart() {return mIdxStart;}
    __aicore__ __forceinline__ void GetSingleCoreShape(SingleCoreShape& singleCoreShape)
    {
        singleCoreShape.singleCoreN = singleCoreN;
        singleCoreShape.singleCoreM = singleCoreM;
        singleCoreShape.singleCoreBatch = singleCoreBatch;
        singleCoreShape.singleCoreCi = convShape.cin_;
    }
};

BLOCK_CONV_SCHEDULER_CLASS_PARAMS
__aicore__ __forceinline__ void BlockSchedulerIterate<BLOCK_CONV_SCHEDULER_FUNC_PARAMS>::CalcStartAddrCommon(const uint32_t din, const uint32_t dout)
{
    hwIn = convShape.hin_ * convShape.win_;
    hwOut = convShape.ho_ * convShape.wo_;
    fmapOneBatchSize = convShape.cin_ * din * hwIn;
    outputOneBatchSize = convShape.cout_ * dout * hwOut;
    if (convShape.hasbias_) {
        biasStartAddr = nIdxStart;
    }
}

BLOCK_CONV_SCHEDULER_CLASS_PARAMS
__aicore__ __forceinline__ void BlockSchedulerIterate<BLOCK_CONV_SCHEDULER_FUNC_PARAMS>::InterateMax(ConvInterateMax& interateMax, SingleCoreShape& singleCoreShape)
{
    interateMax.ddr2l1LoopBatch = ConvCeilDiv(convShape.batch_, convDim.batchDim);
    interateMax.ddr2l1LoopN = CeilDiv(singleCoreShape.singleCoreN, nBL1);
    interateMax.ddr2l1LoopM = CeilDiv(singleCoreShape.singleCoreM, mAL1);
}

BLOCK_CONV_SCHEDULER_CLASS_PARAMS
__aicore__ __forceinline__ void BlockSchedulerIterate<BLOCK_CONV_SCHEDULER_FUNC_PARAMS>::CalcStartAddrMMode(const uint32_t din, const uint32_t dout, const uint32_t kd)
{
    CalcStartAddrCommon(din, dout);
    fmStartAddr = batchIdxStart * fmapOneBatchSize;
    weightStartAddr = nIdxStart * convShape.cin_ * kd * convShape.kh_ * convShape.kw_;
    outputStartAddr = batchIdxStart * outputOneBatchSize + nIdxStart * dout * hwOut + mIdxStart;
}

BLOCK_CONV_SCHEDULER_CLASS_PARAMS
__aicore__ __forceinline__ bool BlockSchedulerIterate<BLOCK_CONV_SCHEDULER_FUNC_PARAMS>::CalcDimData(const uint32_t& blockPerDim, const uint32_t& dim, const uint64_t& wholeDim, const uint64_t &realWholeDim, DimDataToFill& curStruct)
{
    const uint32_t dimIdx = (blockIdx / blockPerDim) % dim;
    const uint64_t maxDimPerCore = CeilDiv(wholeDim, dim);
    const uint64_t realDim = CeilDiv(realWholeDim, maxDimPerCore);

    if (unlikely(dimIdx >= realDim)) {
        return false;
    }

    curStruct.isDimTail = (dimIdx == (realDim - 1));
    curStruct.singleCoreDim = !curStruct.isDimTail ? maxDimPerCore : realWholeDim - (realDim - 1) * maxDimPerCore;
    curStruct.dimIdxStart = dimIdx * maxDimPerCore;
    return true;
}


BLOCK_CONV_SCHEDULER_CLASS_PARAMS
__aicore__ inline bool BlockSchedulerIterate<BLOCK_CONV_SCHEDULER_FUNC_PARAMS>::InitSingleCoreData(uint32_t blockPerNDim, uint32_t blockPerMDim)
{
    DimDataToFill nToFill(singleCoreN, nIdxStart, isNDimTail);
    bool isRealDim = CalcDimData(blockPerNDim, convDim.nDim, Align(convShape.cout_, n0), convShape.cout_, nToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }
    DimDataToFill batchToFill(singleCoreBatch, batchIdxStart, isBatchDimTail);
    isRealDim = CalcDimData(convDim.mDim * convDim.nDim, convDim.batchDim, convShape.batch_, convShape.batch_, batchToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }
    DimDataToFill mToFill(singleCoreM, mIdxStart, isMDimTail);
    uint64_t totalM = convShape.ho_ * convShape.wo_;
    isRealDim = CalcDimData(blockPerMDim, convDim.mDim, Align(totalM, n0), totalM, mToFill);
    if (unlikely(!isRealDim)) {
        return false;
    }
    return true;
}

BLOCK_CONV_SCHEDULER_CLASS_PARAMS
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, mode_, IterateMFirst>
{
    using SchedulerObj = IterateMFirst;
    using SchedulerOp = BlockSchedulerIterate<BLOCK_CONV_SCHEDULER_FUNC_PARAMS>;
};

BLOCK_CONV_SCHEDULER_CLASS_PARAMS
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, mode_, IterateNFirst>
{
    using SchedulerObj = IterateNFirst;
    using SchedulerOp = BlockSchedulerIterate<BLOCK_CONV_SCHEDULER_FUNC_PARAMS>;
};


} // namespace Block
} // namespace Conv
} // namespace Act
#endif
