/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file kernel_conv_forward.h
 * \brief
 */

#ifndef CONV_KERNEL_KERNEL_CONV_FORWARD_H
#define CONV_KERNEL_KERNEL_CONV_FORWARD_H

#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"

#include "../utils/conv_common_utils.h"
#include "../utils/conv_host_utils.h"
#include "../utils/conv_status_utils.h"
#include "../utils/conv_layout_utils.h"
#include "../utils/conv_status_utils.h"
#include "../utils/tensor_utils.h"

#include "../block/block_conv_builder.h"
#include "../block/block_scheduler.h"
#include "../block/block_conv_forward.h"
#include "../prologue/block_prologue.h"
#include "../epilogue/block_epilogue.h"

namespace Act {
namespace Conv {
namespace Kernel {

template <class ProblemShape_, class BlockConvBuilder_, class BlockPrologue_, class BlockEpilogue_,
          class BlockSchedulerObj_, class OutputOrder_, typename Enable_ = void>
class KernelConv {
    static_assert(AscendC::Std::always_false_v<BlockPrologue_> && AscendC::Std::always_false_v<BlockEpilogue_>,
                  "KernelConv is not implemented for this BlockEpilogue");
};

template <class ProblemShape_, class BlockConvBuilder_, class BlockPrologue_, class BlockEpilogue_,
          class BlockSchedulerObj_, class OutputOrder_>
class KernelConv<ProblemShape_, BlockConvBuilder_, BlockPrologue_, BlockEpilogue_, BlockSchedulerObj_, OutputOrder_,
                   AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockPrologue_, Block::BlockPrologueEmpty> &&
                   AscendC::Std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>>> {
public:
    __aicore__ inline KernelConv() {}
    __aicore__ inline ~KernelConv() {}

    using BlockConvBuilder = BlockConvBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockSchedulerObj = BlockSchedulerObj_;
    using BlockPrologue = BlockPrologue_;
    using BlockEpilogue = BlockEpilogue_;
    using OutputOrder = OutputOrder_;

    // ConvOp
    using BlockConvOp = typename BlockConvBuilder::BlockConvOp;
    using BlockConvArguments = typename BlockConvBuilder::BlockConvArguments;

    using BlockPrologueArguments = typename BlockPrologue::Arguments;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;

    using AType = typename BlockConvBuilder::AType;
    using BType = typename BlockConvBuilder::BType;
    using CType = typename BlockConvBuilder::CType;
    using BiasType = typename BlockConvBuilder::BiasType;

    using LayoutA = typename BlockConvBuilder::LayoutA;
    using LayoutB = typename BlockConvBuilder::LayoutB;
    using LayoutC = typename BlockConvBuilder::LayoutC;
    using LayoutBias = typename BlockConvBuilder::LayoutBias;

    using AGlobalTensorType = typename BlockConvBuilder::AGlobalTensorType;
    using BGlobalTensorType = typename BlockConvBuilder::BGlobalTensorType;
    using CGlobalTensorType = typename BlockConvBuilder::CGlobalTensorType;
    using BiasGlobalTensorType = typename BlockConvBuilder::BiasGlobalTensorType;

    using L1TileShape = typename BlockConvBuilder::L1TileShape;
    using L0TileShape = typename BlockConvBuilder::L0TileShape;

    using BlockScheduler = typename Block::BlockSchedulerSelector<ProblemShape, L1TileShape,
                                        L0TileShape, OutputOrder, BlockSchedulerObj>;
    using BlockSchedulerOp = typename BlockScheduler::SchedulerOp;
    BlockSchedulerOp schedule;
    BlockConvOp blockConvOp;

    // attribute
    AGlobalTensorType aGlobal_;
    BGlobalTensorType bGlobal_;
    CGlobalTensorType cGlobal_;
    BiasGlobalTensorType biasGlobal_;
    // shape
    ProblemShape problemShape{};
    ConvInterate iterate{true, true, 0, 0, 0};
    ConvInterateMax interateMax{0, 0, 0};

    struct Arguments {
        ProblemShape problemShape;
        ConvDim dimArgs;
        BlockConvArguments convArgs;
        BlockPrologueArguments prelogueArgs;
        BlockEpilogueArguments epilogueArgs;
        Arguments() = default;
    };

    __aicore__ inline void InitBuffer(Arguments const& args, uint64_t fmStartAddr, uint64_t weightStartAddr,
                                      uint64_t outputStartAddr, uint64_t biasStartAddr)
    {
        BlockConvArguments blockConvArgs = args.convArgs;
        if (problemShape.enable_hf32_) {
            SetHF32Mode(problemShape.enable_hf32_);
            SetHF32TransMode(false);
        }
        static_assert(AscendC::Std::is_same_v<LayoutA, typename layout::NCHW>,
                     "unsupported formart, conv ACT support NCHW only");
        static_assert(AscendC::Std::is_same_v<OutputOrder, typename order::OutputMMode>,
                     "unsupported split mode, conv ACT support MMode only");

        InitGlobalTensor<AGlobalTensorType, AType>(aGlobal_, blockConvArgs.aGmAddr, fmStartAddr);
        InitGlobalTensor<BGlobalTensorType, BType>(bGlobal_, blockConvArgs.bGmAddr, weightStartAddr);
        InitGlobalTensor<CGlobalTensorType, CType>(cGlobal_, blockConvArgs.cGmAddr, outputStartAddr);
        if (problemShape.hasbias_) {
            InitGlobalTensor<AGlobalTensorType, AType>(biasGlobal_, blockConvArgs.biasGmAddr, biasStartAddr);
        }
    }

    __host_aicore__ static size_t GetWorkspaceSize(ProblemShape shape, int64_t blockNum)
    {
        // 当前迭代版本不涉及融合算子，返回 0
        size_t workSpaceSize = 0;
        (void)shape;
        return workSpaceSize;
    }

    static int64_t GetBlockNum(ProblemShape shape)
    {
        return AscendC::GetBlockIdx();
    }

    __aicore__ inline void NFirstIterateAll()
    {
        for (iterate.batchIter = 0; iterate.batchIter < interateMax.ddr2l1LoopBatch; iterate.batchIter++) {
            for (iterate.mAL1Iter = 0; iterate.mAL1Iter < interateMax.ddr2l1LoopM; iterate.mAL1Iter++) {
                iterate.loadAL1Flag = true;
                iterate.loadBL1Flag = true;
                for (iterate.nBL1Iter = 0; iterate.nBL1Iter < interateMax.ddr2l1LoopN; iterate.nBL1Iter++) {
                    blockConvOp.IterateK(iterate, schedule.GetMIdxStart());
                    blockConvOp.GetTensorC(iterate, cGlobal_);
                    iterate.loadAL1Flag = false;
                }
            }
        }
    }

    __aicore__ inline void MFirstIterateAll()
    {
        for (iterate.nBL1Iter = 0; iterate.nBL1Iter < interateMax.ddr2l1LoopN; iterate.nBL1Iter++) {
            for (iterate.batchIter = 0; iterate.batchIter < interateMax.ddr2l1LoopBatch; iterate.batchIter++) {
                iterate.loadBL1Flag = true;
                iterate.loadAL1Flag = true;
                for (iterate.mAL1Iter = 0; iterate.mAL1Iter < interateMax.ddr2l1LoopM; iterate.mAL1Iter++) {
                    blockConvOp.IterateK(iterate, schedule.GetMIdxStart());
                    blockConvOp.GetTensorC(iterate, cGlobal_);
                    iterate.loadBL1Flag = false;
                }
            }
        }
    }

    __aicore__ inline void operator()(Arguments const& args)
    {
        SingleCoreShape singleCoreShape;
        ConvDim dim = args.dimArgs;
        problemShape = args.problemShape;

        if ASCEND_IS_AIV {return;}
        schedule.Init(problemShape, dim);
        schedule.CalcStartAddrMMode();
        schedule.GetSingleCoreShape(singleCoreShape);
        schedule.InterateMax(interateMax, singleCoreShape);

        uint64_t fmStartAddr = schedule.GetFmapStartAddr();
        uint64_t weightStartAddr = schedule.GetWeightStartAddr();
        uint64_t outputStartAddr = schedule.GetOutputStartAddr();
        uint64_t biasStartAddr = schedule.GetBiasStartAddr();
        InitBuffer(args, fmStartAddr, weightStartAddr, outputStartAddr, biasStartAddr);

        blockConvOp.Init(problemShape, dim, singleCoreShape, biasGlobal_, aGlobal_, bGlobal_);
        blockConvOp.LoadBiasFull();

        static_assert(AscendC::Std::is_same_v<typename BlockScheduler::SchedulerObj, typename Block::IterateNFirst> ||
                      AscendC::Std::is_same_v<typename BlockScheduler::SchedulerObj, typename Block::IterateMFirst>,
                      "unsupported iterate policy, conv ACT support NFirst or MFirst only");

        if constexpr (AscendC::Std::is_same_v<typename BlockScheduler::SchedulerObj, typename Block::IterateNFirst>) {
            NFirstIterateAll();
        } else if constexpr (AscendC::Std::is_same_v<typename BlockScheduler::SchedulerObj, typename Block::IterateMFirst>) {
            MFirstIterateAll();
        }

        blockConvOp.End(); // 清楚
    }
};

} // namespace Kernel
} // namespace Conv
} // namespace Act
#endif