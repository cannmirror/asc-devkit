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
 * \file block_mmad_qgmm_pertile.h
 * \brief
 */
#ifndef ACT_BLOCK_MMAD_QGMM_PERTILE_H
#define ACT_BLOCK_MMAD_QGMM_PERTILE_H

#include "lib/matmul/matmul.h"
#include "lib/matmul/tiling.h"
#include "lib/matmul/constant_tiling.h"

#include "include/utils/grouped_matmul_constant.h"
#include "include/utils/layout_utils.h"
#include "include/utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tile/tile_copy.h"
#include "block_mmad_qgmm_pertile_param.h"

namespace Act {
namespace Gemm {
namespace Block {

#define QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS                                                                             \
    template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_,                \
              class BiasType_, class LayoutBias_, class L1TileShape_, class L0TileShape_, class BlockScheduler_,       \
              class TileCopyParam_>

#define QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS                                                                              \
    GMMPerTile<>, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_, L1TileShape_,          \
        L0TileShape_, BlockScheduler_, TileCopyParam_

using namespace Act::Gemm::GroupedMatmul;

template <class BlockMatmulPolicy_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_,
          class LayoutC_, class BiasType_, class LayoutBias_, class L1TileShape_, class L0TileShape_,
          class BlockScheduler_, class TileCopyParam_ = void>
class BlockMmadGmm {
    static_assert(AscendC::Std::always_false_v<BlockMatmulPolicy_>, "Should not be here!");
};

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
class BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using TileCopyParam = TileCopyParam_;
    using BlockScheduler = BlockScheduler_;

    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;

    using BlockType = typename BlockScheduler::BlockType;
    // host side kernel arguments
    struct Arguments {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
    };

    // params
    using Params = Arguments;

public:
    __aicore__ BlockMmadGmm() = default;
    __aicore__ ~BlockMmadGmm() = default;
    __aicore__ inline void Init(const Params* params, BlockType* block, const TCubeTiling* tilingData,
                                AscendC::LocalTensor<CType>* ping, AscendC::LocalTensor<CType>* pong);
    __aicore__ inline void AicBaseMadProcess(AscendC::LocalTensor<AType>& aL1, AscendC::LocalTensor<BType>& bL1,
                                             uint64_t kInner, uint64_t kAL1Offset, bool isTailAL1, uint64_t kBL1Offset,
                                             bool isTailBL1);
    __aicore__ inline void CopyInA1Nd2Nz(const uint64_t kOffset, const bool isTailAL1);
    __aicore__ inline void CopyInB1Nd2Nz(const uint64_t kOffset, const bool isTailBL1);
    __aicore__ inline void CopyInA2(AscendC::LocalTensor<AType>& aL1, const uint64_t mAL1Offset,
                                    const uint64_t kAL1Offset, const uint64_t kOffset, const bool isTailAL1);
    __aicore__ inline void CopyInB2(AscendC::LocalTensor<BType>& bL1, const uint64_t nBL1Offset,
                                    const uint64_t kBL1Offset, const uint64_t kOffset, const bool isTailBL1);
    __aicore__ inline void MmadBase(const uint64_t kOffset);
    __aicore__ inline void ProcessAicSingleK();
    __aicore__ inline void UpdateParamForNextGroup(int32_t m, int32_t n, int32_t k);
    __aicore__ inline void UpdateGlobalAddr();
    __aicore__ inline void End();
    __aicore__ inline void WaitForVector(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreWaitFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_FIX>(GMM_AIC_SYNC_AIV_FLAG + crossPingPongID);
        AscendC::CrossCoreWaitFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_FIX>(GMM_AIC_SYNC_AIV_FLAG + crossPingPongID +
                                                                    GMM_FLAG_ID_MAX);
    }
    __aicore__ inline void NotifyVector(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreSetFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_FIX>(GMM_AIV_SYNC_AIC_FLAG + crossPingPongID);
        AscendC::CrossCoreSetFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_FIX>(GMM_AIV_SYNC_AIC_FLAG + crossPingPongID +
                                                                   GMM_FLAG_ID_MAX);
    }

public:
    BlockType* block_;
    const TCubeTiling* mmTilingData_;
    MatMulCommonParam<transA, transB, BlockType> matmulParam_;

    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::LocalTensor<CType>* mmResPing_;
    AscendC::LocalTensor<CType>* mmResPong_;

    GM_ADDR xTensorPtr_;
    GM_ADDR weightTensorPtr_;
    // define the queue
    AscendC::TQue<AscendC::QuePosition::A1, 1> inQueueTensorAL1_;
    AscendC::TQue<AscendC::QuePosition::B1, 1> inQueueTensorBL1_;
    AscendC::TQue<AscendC::QuePosition::A2, 1> inQueueTensorAL0_;
    AscendC::TQue<AscendC::QuePosition::B2, 1> inQueueTensorBL0_;
    AscendC::TQue<AscendC::QuePosition::CO1, 1> inQueueTensorCL0_;

private:
    AscendC::TPipe* pipe_;
    uint64_t baseCount_ = 0;
    uint64_t maxStepK_ = 0;
    uint64_t minStepK_ = 0;
    int32_t k_ = 0;
    uint16_t aL1BlockNum_ = 0;
    uint16_t crossPingPongID_ = 0;
    bool needAicWait_ = false;
    bool orderAL1BL1_ = false;
};

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::Init(const Params* params, BlockType* block,
                                                                             const TCubeTiling* tilingData,
                                                                             AscendC::LocalTensor<CType>* ping,
                                                                             AscendC::LocalTensor<CType>* pong)
{
    if ASCEND_IS_AIC {
        block_ = block;
        mmTilingData_ = tilingData;
        mmResPing_ = ping;
        mmResPong_ = pong;

        xTensorPtr_ = params->aGmAddr;
        weightTensorPtr_ = params->bGmAddr;
        pipe_ = GetTPipePtr();

        // L1 buffer ping-pong enabled by default
        auto aL1ElementNum = mmTilingData_->baseM * mmTilingData_->baseK * mmTilingData_->stepKa;
        aL1BlockNum_ = static_cast<uint16_t>(aL1ElementNum * sizeof(AType) / static_cast<uint64_t>(GMM_DATA_BLOCK));
        pipe_->InitBuffer(inQueueTensorAL1_, GMM_BUFFER_NUM, aL1ElementNum * sizeof(AType));
        pipe_->InitBuffer(inQueueTensorBL1_, GMM_BUFFER_NUM,
                          mmTilingData_->baseN * mmTilingData_->baseK * mmTilingData_->stepKb * sizeof(BType));
        // L0 buffer ping-pong enabled by default
        pipe_->InitBuffer(inQueueTensorAL0_, GMM_BUFFER_NUM,
                          mmTilingData_->baseM * mmTilingData_->baseK * sizeof(AType));
        pipe_->InitBuffer(inQueueTensorBL0_, GMM_BUFFER_NUM,
                          mmTilingData_->baseN * mmTilingData_->baseK * sizeof(BType));
        pipe_->InitBuffer(inQueueTensorCL0_, GMM_BUFFER_NUM,
                          mmTilingData_->baseM * mmTilingData_->baseN * sizeof(CType));
        matmulParam_.Init(block_, mmTilingData_);
    }
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::UpdateParamForNextGroup(int32_t m, int32_t n,
                                                                                                int32_t k)
{
    k_ = k;
    matmulParam_.UpdateForNextGroup(m, n, k);
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr()
{
    if ASCEND_IS_AIC {
        aGlobal_.SetGlobalBuffer(GetTensorAddr<AType>(0, xTensorPtr_) + block_->params_.aGroupAddrOffset);
        bGlobal_.SetGlobalBuffer(GetTensorAddr<BType>(0, weightTensorPtr_) + block_->params_.bGroupAddrOffset);
        orderAL1BL1_ = mmTilingData_->stepKa >= mmTilingData_->stepKb;
        maxStepK_ = (orderAL1BL1_ ? mmTilingData_->stepKa : mmTilingData_->stepKb) * mmTilingData_->baseK;
        minStepK_ = (orderAL1BL1_ ? mmTilingData_->stepKb : mmTilingData_->stepKa) * mmTilingData_->baseK;
    }
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::ProcessAicSingleK()
{
    bool isTailAL1 = false;
    bool isTailBL1 = false;
    if (orderAL1BL1_) {
        for (uint64_t kOuter = 0; kOuter < k_; kOuter += maxStepK_) {
            isTailAL1 = (kOuter + maxStepK_) >= k_;
            CopyInA1Nd2Nz(kOuter, isTailAL1);
            auto aL1 = inQueueTensorAL1_.template DeQue<AType>();
            for (uint64_t kInner = kOuter; kInner < Min(kOuter + maxStepK_, static_cast<uint64_t>(k_));
                 kInner += minStepK_) {
                isTailBL1 = (kInner + minStepK_) >= k_;
                CopyInB1Nd2Nz(kInner, isTailBL1);
                auto bL1 = inQueueTensorBL1_.template DeQue<BType>();
                uint64_t kAL1Offset = kInner - kOuter;
                AicBaseMadProcess(aL1, bL1, kInner, kAL1Offset, isTailAL1, 0UL, isTailBL1);
                inQueueTensorBL1_.FreeTensor(bL1);
            }
            inQueueTensorAL1_.FreeTensor(aL1);
        }
    } else {
        for (uint64_t kOuter = 0; kOuter < k_; kOuter += maxStepK_) {
            isTailBL1 = (kOuter + maxStepK_) >= k_;
            CopyInB1Nd2Nz(kOuter, isTailBL1);
            auto bL1 = inQueueTensorBL1_.template DeQue<BType>();
            for (uint64_t kInner = kOuter; kInner < Min(kOuter + maxStepK_, static_cast<uint64_t>(k_));
                 kInner += minStepK_) {
                isTailAL1 = (kInner + minStepK_) >= k_;
                CopyInA1Nd2Nz(kInner, isTailAL1);
                uint64_t kBL1Offset = kInner - kOuter;
                auto aL1 = inQueueTensorAL1_.template DeQue<AType>();
                AicBaseMadProcess(aL1, bL1, kInner, 0UL, isTailAL1, kBL1Offset, isTailBL1);
                inQueueTensorAL1_.FreeTensor(aL1);
            }
            inQueueTensorBL1_.FreeTensor(bL1);
        }
    }
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::AicBaseMadProcess(
    AscendC::LocalTensor<AType>& aL1, AscendC::LocalTensor<BType>& bL1, uint64_t kInner, uint64_t kAL1Offset,
    bool isTailAL1, uint64_t kBL1Offset, bool isTailBL1)
{
    for (uint64_t kb = kInner; kb < Min(kInner + minStepK_, static_cast<uint64_t>(k_)); kb += mmTilingData_->baseK) {
        CopyInA2(aL1, 0, kAL1Offset, kb, isTailAL1);
        CopyInB2(bL1, 0, kBL1Offset, kb, isTailBL1);
        MmadBase(kb);
        auto cL0 = inQueueTensorCL0_.template DeQue<CType>();
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams(
            block_->mmParams_.fixpipeN, block_->mmParams_.fixpipeM, block_->mmParams_.fixSrcStride,
            block_->mmParams_.fixpipeD);
        fixpipeParams.dualDstCtl = block_->mmParams_.fixpipeSplitN ? 2 : 1; // 2 means splitting N with ratio 1:2
        if (needAicWait_) {
            WaitForVector(crossPingPongID_);
        }
        AscendC::Fixpipe<CType, CType, AscendC::Impl::CFG_ROW_MAJOR_UB>(
            crossPingPongID_ == 0 ? *mmResPing_ : *mmResPong_, cL0, fixpipeParams);
        NotifyVector(crossPingPongID_);
        needAicWait_ = needAicWait_ || crossPingPongID_ == 1;
        crossPingPongID_ = (crossPingPongID_ + 1) & 1;
        inQueueTensorCL0_.FreeTensor(cL0);

        kAL1Offset = kAL1Offset + mmTilingData_->baseK;
        kBL1Offset = kBL1Offset + mmTilingData_->baseK;
        baseCount_++;
    }
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::CopyInA2(AscendC::LocalTensor<AType>& aL1,
                                                                                 const uint64_t mAL1Offset,
                                                                                 const uint64_t kAL1Offset,
                                                                                 const uint64_t kOffset,
                                                                                 const bool isTailAL1)
{
    uint64_t offsetAL1 = matmulParam_.CalcAL1Offset(mAL1Offset, kAL1Offset, isTailAL1);
    AscendC::LocalTensor<AType> aL0 = inQueueTensorAL0_.template AllocTensor<AType>();
    AscendC::LoadData2DParamsV2 loadData2dParams;
    matmulParam_.LoadData2dParamsA(loadData2dParams, kOffset, isTailAL1);
    AscendC::LoadData(aL0, aL1[offsetAL1], loadData2dParams);
    inQueueTensorAL0_.EnQue(aL0);
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::CopyInB2(AscendC::LocalTensor<BType>& bL1,
                                                                                 const uint64_t nBL1Offset,
                                                                                 const uint64_t kBL1Offset,
                                                                                 const uint64_t kOffset,
                                                                                 const bool isTailBL1)
{
    uint64_t offsetBL1 = matmulParam_.CalcBL1Offset(nBL1Offset, kBL1Offset, isTailBL1);
    AscendC::LocalTensor<BType> bL0 = inQueueTensorBL0_.template AllocTensor<BType>();
    AscendC::LoadData2DParamsV2 loadData2dParams;
    matmulParam_.LoadData2dParamsB(loadData2dParams, kOffset, isTailBL1);
    AscendC::LoadData(bL0, bL1[offsetBL1], loadData2dParams);
    inQueueTensorBL0_.EnQue(bL0);
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::MmadBase(const uint64_t kOffset)
{
    auto aL0 = inQueueTensorAL0_.template DeQue<AType>();
    auto bL0 = inQueueTensorBL0_.template DeQue<BType>();
    AscendC::LocalTensor<CType> cL0 = inQueueTensorCL0_.template AllocTensor<CType>();
    uint32_t mmadK = Min(static_cast<uint64_t>(mmTilingData_->baseK), k_ - kOffset);
    AscendC::MmadParams mmadParams;
    if constexpr (transA) {
        mmadParams.m = Align(block_->params_.singleCoreM, static_cast<uint64_t>(GMM_DATA_BLOCK));
    } else {
        mmadParams.m = block_->params_.singleCoreM;
    }
    if constexpr (transB) {
        mmadParams.n = block_->params_.singleCoreN;
    } else {
        mmadParams.n = Align(block_->params_.singleCoreN, static_cast<uint64_t>(GMM_DATA_BLOCK));
    }
    mmadParams.k = mmadK;
    mmadParams.disableGemv = true;
    AscendC::Mmad(cL0, aL0, bL0, mmadParams);
    inQueueTensorCL0_.EnQue(cL0);
    inQueueTensorAL0_.FreeTensor(aL0);
    inQueueTensorBL0_.FreeTensor(bL0);
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::CopyInA1Nd2Nz(const uint64_t kOffset,
                                                                                      const bool isTailAL1)
{
    AscendC::LocalTensor<AType> aL1 = inQueueTensorAL1_.template AllocTensor<AType>();
    uint64_t offset = matmulParam_.CalcAGMOffsetInnerLoop(0, kOffset);
    AscendC::Nd2NzParams nd2nzParam;
    matmulParam_.CalNd2NzParamA(nd2nzParam, isTailAL1);
    AscendC::DataCopy(aL1, aGlobal_[offset], nd2nzParam);
    inQueueTensorAL1_.EnQue(aL1);
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::CopyInB1Nd2Nz(const uint64_t kOffset,
                                                                                      const bool isTailBL1)
{
    AscendC::LocalTensor<BType> bL1 = inQueueTensorBL1_.template AllocTensor<BType>();
    uint64_t offset = matmulParam_.CalcBGMOffsetInnerLoop(0, kOffset);
    AscendC::Nd2NzParams nd2nzParam;
    matmulParam_.CalNd2NzParamB(nd2nzParam, isTailBL1);
    AscendC::DataCopy(bL1, bGlobal_[offset], nd2nzParam);
    inQueueTensorBL1_.EnQue(bL1);
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::End()
{
    if ASCEND_IS_AIC {
        if (baseCount_ > 1) {
            WaitForVector(0); // ping
            WaitForVector(1); // pong
        } else if (baseCount_ == 1) {
            WaitForVector(0);
        }
    }
}
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
