/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_qgmm_pertile.h
 * \brief
 */

#ifndef ACT_KERNEL_QGMM_PERTILE_H
#define ACT_KERNEL_QGMM_PERTILE_H

#include "include/utils/common_utils.h"
#include "include/utils/grouped_matmul_constant.h"
#include "include/utils/layout_utils.h"
#include "include/utils/tensor_utils.h"
#include "include/matmul/matmul_intf.h"
#include "include/matmul/block/block_scheduler_utils.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"

namespace Act {
namespace Gemm {
namespace Kernel {
#define QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS                                                                           \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

using namespace Act::Gemm::GroupedMatmul;

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
class QuantMmGroupedPerTile {
public:
    __aicore__ inline QuantMmGroupedPerTile() {}
    __aicore__ inline ~QuantMmGroupedPerTile() {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockType = typename BlockScheduler::BlockType;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using ScaleType = typename BlockEpilogue::ScaleType;
    using PtScaleType = typename BlockEpilogue::PtScaleType;
    using YType = typename BlockEpilogue::YType;
    using LayoutB = typename BlockMmad::LayoutB;
    using TilingTypeGMMQuantParams = typename BlockScheduler::TilingTypeGMMQuantParams;
    using TilingTypeGMMArrayAddr = typename BlockScheduler::TilingTypeGMMArrayAddr;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams scheduleParams;
        Params() = default;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    __aicore__ inline void SetMNK(uint32_t groupIdx, int32_t& mSize, int32_t& nSize, int32_t& kSize);
    __aicore__ inline void ProcessSingleGroup(uint32_t groupIdx);
    __aicore__ inline bool IsLastGroupAndRound(uint32_t groupIdx, uint64_t roundIdx);
    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx);
    __aicore__ inline void UpdateMMGlobalAddr();
    __aicore__ inline void Iterate();
    __aicore__ inline void End();

private:
    BlockScheduler blockSch_;
    BlockType* block_;

    BlockMmad mmadOp;
    BlockEpilogue epilogueOp;

    const TilingTypeGMMQuantParams* gmmQuantParams_;
    const TCubeTiling* mmTilingData_;
    const TilingTypeGMMArrayAddr* mListGm_;
    const TilingTypeGMMArrayAddr* kListGm_;
    const TilingTypeGMMArrayAddr* nListGm_;

    AscendC::GlobalTensor<int64_t> groupListGlobal_;
    AscendC::LocalTensor<CType> mmResPing_;
    AscendC::LocalTensor<CType> mmResPong_;

    GM_ADDR groupListPtr_;

private:
    AscendC::TPipe* pipe_;
    uint32_t blockIdx_;
    int32_t preOffset_ = 0;
    uint32_t groupNum_;
    int32_t m_;
    int32_t n_;
    int32_t k_;
    int8_t groupType_;
    uint8_t groupListType_;

    // define the queue
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> vecQueMMRes_;
};

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::Run(const Params& params)
{
    Init(params);
    bool isKZeroInit = false;
    for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
        // Update input parameters M, N, K within the group
        SetMNK(groupIdx, m_, n_, k_);
        block_->template UpdateGroupOffset<transA, transB, ScaleType, true>(m_, n_, k_, groupIdx);
        if (m_ <= 0 || n_ <= 0) {
            continue;
        }
        if (k_ <= 0) {
            // With K-axis grouping: output (m,n) required. int8 inputs disable K-axis grouping.
            // No bias (all zeros) when K-axis grouped.
            if ASCEND_IS_AIV {
                if (groupType_ == GMM_SPLIT_K) {
                    epilogueOp.template InitOutputWithZero<YType>(static_cast<uint64_t>(m_) * n_,
                                                                  mmTilingData_->usedCoreNum, isKZeroInit);
                }
            }
            continue;
        }
        if ASCEND_IS_AIC {
            mmadOp.UpdateParamForNextGroup(m_, n_, k_);
        }
        if ASCEND_IS_AIV {
            epilogueOp.UpdateParamForNextGroup(m_, n_, k_);
        }

        block_->template UpdateGroupParams<true>();

        UpdateMMGlobalAddr();
        ProcessSingleGroup(groupIdx);
    }
    End();
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{
    groupListPtr_ = params.mmadParams.groupListGmAddr;
    mmTilingData_ = params.scheduleParams.gmmTilingDataIn;
    gmmQuantParams_ = params.scheduleParams.gmmBaseParamsIn;
    groupNum_ = gmmQuantParams_->groupNum;
    groupType_ = gmmQuantParams_->groupType;
    groupListType_ = gmmQuantParams_->groupListType;
    pipe_ = GetTPipePtr();

    blockIdx_ = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        blockIdx_ = blockIdx_ / AscendC::GetTaskRation();
    }

    block_ = &blockSch_.block;
    block_->template Init<true>(mmTilingData_, blockIdx_);

    if (groupListPtr_ != nullptr) {
        groupListGlobal_.SetGlobalBuffer((__gm__ int64_t*)groupListPtr_);
    }
    mListGm_ = params.scheduleParams.gmmArrayAddrIn;
    kListGm_ = params.scheduleParams.gmmArrayAddrIn + GMM_MKN_LIST_LEN;
    nListGm_ = params.scheduleParams.gmmArrayAddrIn + GMM_MKN_LIST_LEN * 2; // 2: mListGm_ + kListGm_
#if defined(__DAV_C310__)
    uint64_t baseL0cSingleV =
        Act::Gemm::CeilDiv(static_cast<uint64_t>(mmTilingData_->baseM) * mmTilingData_->baseN, 2UL); // 2: AIC:AIV=1:2
#else
    uint64_t baseL0cSingleV = Act::Gemm::CeilDiv(mmTilingData_->baseM * mmTilingData_->baseN, AscendC::GetTaskRation());
#endif
    pipe_->InitBuffer(vecQueMMRes_, GMM_BUFFER_NUM, baseL0cSingleV * sizeof(CType));
    mmResPing_ = vecQueMMRes_.template AllocTensor<CType>();
    mmResPong_ = vecQueMMRes_.template AllocTensor<CType>();
    mmadOp.Init(&params.mmadParams, block_, mmTilingData_, &mmResPing_, &mmResPong_);
    epilogueOp.Init(&params.epilogueParams, block_, mmTilingData_, gmmQuantParams_, &mmResPing_, &mmResPong_,
                    baseL0cSingleV);
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::ProcessSingleGroup(uint32_t groupIdx)
{
    for (uint64_t roundIdx = 0; roundIdx < block_->params_.round; ++roundIdx) {
        bool isLastGroupRound = IsLastGroupAndRound(groupIdx, roundIdx);
        block_->template UpdateBasicIndex<true>(roundIdx, isLastGroupRound);
        // 1. Set single core param
        block_->template UpdateBlockParams<transA, transB>(roundIdx, isLastGroupRound);
        if (block_->params_.singleCoreM <= 0 || block_->params_.singleCoreN <= 0) {
            return;
        }
        if ASCEND_IS_AIC {
            block_->template CalcGMOffset<transA, transB, ScaleType, true>();
        }
        block_->template UpdatePerBlockMmParam<transA, transB>();
        Iterate();
    }
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::Iterate()
{
    if ASCEND_IS_AIC {
        mmadOp.ProcessAicSingleK();
    }
    if ASCEND_IS_AIV {
        epilogueOp.ProcessAivSingleKPertile();
    }
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline bool QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::IsLastGroupAndRound(uint32_t groupIdx,
                                                                                                      uint64_t roundIdx)
{
    return groupIdx == groupNum_ - 1 && roundIdx == block_->params_.round - 1 && blockIdx_ <= block_->GetEndBlockIdx();
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::SetMNK(uint32_t groupIdx,
                                                                                         int32_t& mSize, int32_t& nSize,
                                                                                         int32_t& kSize)
{
    int32_t splitValue = GetSplitValueFromGroupList(groupIdx);
    switch (groupType_) {
        case (GMM_SPLIT_M): {
            mSize = splitValue;
            uint32_t valueIdx = gmmQuantParams_->singleW == 1 ? 0 : groupIdx;
            kSize = kListGm_[valueIdx];
            nSize = nListGm_[valueIdx];
            break;
        }
        case (GMM_SPLIT_K): {
            mSize = gmmQuantParams_->singleX == 1 ? mListGm_[0] : mListGm_[groupIdx];
            kSize = splitValue;
            nSize = gmmQuantParams_->singleW == 1 ? nListGm_[0] : nListGm_[groupIdx];
            break;
        }
        default: {
            mSize = mListGm_[groupIdx];
            kSize = kListGm_[groupIdx];
            nSize = nListGm_[groupIdx];
        }
    }
    return;
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::UpdateMMGlobalAddr()
{
    if ASCEND_IS_AIC {
        mmadOp.UpdateGlobalAddr();
    }
    if ASCEND_IS_AIV {
        epilogueOp.UpdateGlobalAddr();
    }
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::End()
{
    vecQueMMRes_.FreeTensor(mmResPing_);
    vecQueMMRes_.FreeTensor(mmResPong_);
    if ASCEND_IS_AIC {
        mmadOp.End();
    }
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline int32_t
QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::GetSplitValueFromGroupList(uint32_t groupIdx)
{
    int32_t splitValue = 0;
    if (likely(groupType_ != -1)) { // -1: no  need to split
        if (groupListType_ == 0) {
            int32_t offset = static_cast<int32_t>(groupListGlobal_.GetValue(groupIdx));
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = static_cast<int32_t>(groupListGlobal_.GetValue(groupIdx));
        }
    }
    return splitValue;
}

} // namespace Kernel
} // namespace Gemm
} // namespace Act

#endif