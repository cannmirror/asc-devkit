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
 * \file block_epilogue_pertile.h
 * \brief
 */

#ifndef ACT_BLOCK_EPILOGUE_PERTILE_H
#define ACT_BLOCK_EPILOGUE_PERTILE_H
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "include/utils/common_utils.h"
#include "include/utils/grouped_matmul_constant.h"
#include "include/utils/layout_utils.h"
#include "include/utils/tensor_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
#define QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS                                                                         \
    template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeBias_,              \
              typename DataTypeScale_, typename LayoutScale_, typename DataTypePtScale_, typename LayoutPtScale_,      \
              typename BlockScheduler_>
#define QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS                                                                          \
    L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeBias_, DataTypeScale_, LayoutScale_, DataTypePtScale_,            \
        LayoutPtScale_, BlockScheduler_

using namespace Act::Gemm::GroupedMatmul;

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
class BlockEpiloguePerTile {
public:
    __aicore__ inline BlockEpiloguePerTile() {}

    struct Arguments {
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
        GM_ADDR pertokenScaleGmAddr{nullptr};
        GM_ADDR inGmAddr{nullptr};
        GM_ADDR outGmAddr{nullptr};
        Arguments() = default;
    };

    struct Params {
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
        GM_ADDR pertokenScaleGmAddr{nullptr};
        GM_ADDR inGmAddr{nullptr};
        GM_ADDR outGmAddr{nullptr};
        Params() = default;
    };

    using YType = DataTypeOut_;
    using CType = DataTypeIn_;
    using BiasType = DataTypeBias_;
    using ScaleType = DataTypeScale_;
    using PtScaleType = DataTypePtScale_;
    using LayoutScale = LayoutScale_;
    using LayoutPtScale = LayoutPtScale_;
    using BlockScheduler = BlockScheduler_;
    using BlockType = typename BlockScheduler::BlockType;
    using TilingTypeGMMQuantParams = typename BlockScheduler::TilingTypeGMMQuantParams;

    static constexpr bool transA = TagToTrans<LayoutScale>::value;
    static constexpr bool transB = TagToTrans<LayoutPtScale>::value;

public:
    template <typename T>
    __aicore__ inline void InitOutputWithZero(uint64_t ySize, int32_t usedCoreNum, bool& isKZeroInit);
    __aicore__ inline void Init(const Params* params, BlockType* block, const TCubeTiling* tilingData,
                                const TilingTypeGMMQuantParams* gmmQuantParams, AscendC::LocalTensor<CType>* ping,
                                AscendC::LocalTensor<CType>* pong, uint64_t baseL0cSingleV);
    template <class T, bool isFirst>
    __aicore__ inline __ubuf__ T* CopyInX1Scale(uint64_t srcOffset, uint64_t m, uint64_t k,
                                                AscendC::LocalTensor<T>& buf);
    template <class T>
    __aicore__ inline T CopyInX2Scale(__gm__ T* src, uint64_t offset);
    __aicore__ inline uint64_t CalcX1OffsetPerGroup();
    __aicore__ inline uint64_t CalcX2OffsetPerGroup();
    template <bool isFirstKLoop>
    __aicore__ inline void AivPerTensor(__ubuf__ CType* dst, __ubuf__ CType* l0cOut, __ubuf__ PtScaleType* x1Scale,
                                        uint16_t mSize, uint16_t nSize, uint16_t kSize, uint32_t nSrcUbAligned,
                                        ScaleType x2Scale, uint64_t ptIdx);
    __aicore__ inline void AivPostProcess(AscendC::LocalTensor<CType>& mmAddUb);
    __aicore__ inline void ProcessAivSingleKPertile();
    __aicore__ inline void UpdateGlobalAddr();
    __aicore__ inline void UpdateParamForNextGroup(int32_t m, int32_t n, int32_t k);
    __aicore__ inline void WaitForCube(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreWaitFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_V>(GMM_AIV_SYNC_AIC_FLAG + crossPingPongID);
    }
    __aicore__ inline void NotifyCube(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreSetFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_V>(GMM_AIC_SYNC_AIV_FLAG + crossPingPongID);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> vecQueBias_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> vecQuePertokenScale_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> vecQueAdd_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> vecQueOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> initBuff_;

    AscendC::GlobalTensor<YType> cGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::GlobalTensor<PtScaleType> scaleAGlobal_;
    __gm__ ScaleType* scaleBGlobal_;
    AscendC::LocalTensor<CType>* mmResPing_;
    AscendC::LocalTensor<CType>* mmResPong_;
    AscendC::LocalTensor<YType> initLocal_;

    GM_ADDR biasTensorPtr_;
    GM_ADDR scaleTensorPtr_;
    GM_ADDR perTokenScalePtr_;
    GM_ADDR yTensorPtr_;

private:
    BlockType* block_;
    const TilingTypeGMMQuantParams* gmmQuantParams_;
    const TCubeTiling* mmTilingData_;

    uint64_t scaleM_ = 0;
    uint64_t scaleN_ = 0;
    uint64_t scaleK_ = 0;
    uint32_t subBlockIdx_;
    int32_t n_;
    int32_t k_;
    uint16_t crossPingPongID_ = 0;
    int8_t groupType_;
    bool isBiasEpilogue_;
    bool isPertile_ = false;
};

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::Init(
    const Params* params, BlockType* block, const TCubeTiling* tilingData,
    const TilingTypeGMMQuantParams* gmmQuantParams, AscendC::LocalTensor<CType>* ping,
    AscendC::LocalTensor<CType>* pong, uint64_t baseL0cSingleV)
{
    if ASCEND_IS_AIV {
        biasTensorPtr_ = params->biasGmAddr;
        scaleTensorPtr_ = (GM_ADDR)(GetTensorAddr<ScaleType>(0, params->scaleGmAddr));
        perTokenScalePtr_ = params->pertokenScaleGmAddr;
        yTensorPtr_ = params->outGmAddr;
        block_ = block;
        mmTilingData_ = tilingData;
        gmmQuantParams_ = gmmQuantParams;
        mmResPing_ = ping;
        mmResPong_ = pong;
        groupType_ = gmmQuantParams_->groupType;
        pipe_ = GetTPipePtr();

        subBlockIdx_ = AscendC::GetSubBlockIdx();
        pipe_->InitBuffer(vecQueAdd_, 1, baseL0cSingleV * sizeof(CType));
        if constexpr (!AscendC::IsSameType<CType, YType>::value) {
            pipe_->InitBuffer(vecQueOut_, GMM_BUFFER_NUM, Act::Gemm::CeilDiv(baseL0cSingleV * sizeof(YType), 4));
        }

        isPertile_ = block_->params_.groupSizeM == 1;
        if (isPertile_) {
            pipe_->InitBuffer(
                vecQuePertokenScale_, GMM_BUFFER_NUM,
                Align(Act::Gemm::CeilDiv(Max(mmTilingData_->baseM, mmTilingData_->baseN), AscendC::GetTaskRation()) *
                          GMM_MAX_STEP_SCALEA_K * sizeof(PtScaleType),
                      GMM_UB_ALIGN_SIZE));
        }
        isBiasEpilogue_ = gmmQuantParams_->hasBias == 1;
        if (isBiasEpilogue_) {
            pipe_->InitBuffer(vecQueBias_, GMM_BUFFER_NUM, mmTilingData_->baseN * sizeof(BiasType));
        }

        // k = 0, init out
        if (subBlockIdx_ == 0 && groupType_ == GMM_SPLIT_K) {
            pipe_->InitBuffer(initBuff_, GMM_MAX_REPEAT_TIMES * GMM_UB_ALIGN_SIZE);
            initLocal_ = initBuff_.Get<YType>();
        }
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::UpdateParamForNextGroup(int32_t m, int32_t n, int32_t k)
{
    if ASCEND_IS_AIV {
        n_ = n;
        k_ = k;

        scaleM_ = Act::Gemm::CeilDiv(m, block_->params_.groupSizeM);
        scaleN_ = Act::Gemm::CeilDiv(n, block_->params_.groupSizeN);
        scaleK_ = Act::Gemm::CeilDiv(k, block_->params_.groupSizeK);
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr()
{
    if ASCEND_IS_AIV {
        if (static_cast<bool>(gmmQuantParams_->hasBias)) {
            biasGlobal_.SetGlobalBuffer(GetTensorAddr<BiasType>(0, biasTensorPtr_) +
                                        block_->params_.biasGroupAddrOffset);
        }
        scaleAGlobal_.SetGlobalBuffer((__gm__ PtScaleType*)perTokenScalePtr_ + block_->params_.xScaleGroupAddrOffset);
        scaleBGlobal_ = (__gm__ ScaleType*)scaleTensorPtr_ + block_->params_.wScaleGroupAddrOffset;
        cGlobal_.SetGlobalBuffer(GetTensorAddr<YType>(0, yTensorPtr_) + block_->params_.cGroupAddrOffset);
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline uint64_t BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CalcX1OffsetPerGroup()
{
    if constexpr (transA) {
        return block_->ubParams_.offsetScaleM;
    } else {
        return block_->ubParams_.offsetScaleM * scaleK_;
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline uint64_t BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CalcX2OffsetPerGroup()
{
    if constexpr (transB) {
        return block_->ubParams_.offsetScaleN * scaleK_;
    } else {
        return block_->ubParams_.offsetScaleN;
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <class T, bool isFirst>
__aicore__ inline __ubuf__ T*
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CopyInX1Scale(uint64_t srcOffset, uint64_t m, uint64_t k,
                                                                           AscendC::LocalTensor<T>& buf)
{
    AscendC::DataCopyParams ptScale2UbParams{0, 0, 0, 0};
    AscendC::DataCopyPadParams padParams;
    if constexpr (transA) {
        ptScale2UbParams.blockCount = k;
        ptScale2UbParams.blockLen = m * sizeof(T);
        ptScale2UbParams.srcStride = (scaleM_ - m) * sizeof(T);
    } else {
        ptScale2UbParams.blockCount = m;
        ptScale2UbParams.blockLen = k * sizeof(T);
        ptScale2UbParams.srcStride = (scaleK_ - k) * sizeof(T);
    }
    if constexpr (!isFirst) {
        vecQuePertokenScale_.FreeTensor(buf);
        buf = vecQuePertokenScale_.template AllocTensor<T>();
    }
    AscendC::DataCopyPad(buf, scaleAGlobal_[srcOffset], ptScale2UbParams, padParams);
    vecQuePertokenScale_.EnQue(buf);
    buf = vecQuePertokenScale_.template DeQue<T>();
    return reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr());
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <class T>
__aicore__ inline T BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CopyInX2Scale(__gm__ T* src,
                                                                                               uint64_t offset)
{
    if constexpr (transB) {
        return src[offset];
    } else {
        return src[offset * scaleN_];
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::ProcessAivSingleKPertile()
{
    block_->UpdatePerBlockUBParam();
    uint64_t scaleOffsetX1 = CalcX1OffsetPerGroup();
    uint64_t scaleOffsetX2 = CalcX2OffsetPerGroup();
    auto scaleX2Addr = scaleBGlobal_ + scaleOffsetX2;

    AscendC::LocalTensor<CType> mmAddUb = vecQueAdd_.template AllocTensor<CType>();
    auto mmAddUbAddr = reinterpret_cast<__ubuf__ CType*>(mmAddUb.GetPhyAddr());
    const uint16_t ptScaleKElem = Min(GMM_MAX_STEP_SCALEA_K, scaleK_);
    uint64_t kElem;
    __ubuf__ PtScaleType* ptScaleUbAddr;
    AscendC::LocalTensor<PtScaleType> mmPertokenScaleUb = vecQuePertokenScale_.template AllocTensor<PtScaleType>();
    for (uint64_t kb = 0, kOffset = 0; kb < k_; kb += mmTilingData_->baseK, kOffset++) {
        ScaleType scaleX2 = CopyInX2Scale<ScaleType>(scaleX2Addr, kOffset);
        uint64_t pertokenIdx = kOffset % ptScaleKElem;
        if (pertokenIdx == 0) {
            kElem = Min(static_cast<uint64_t>(ptScaleKElem), scaleK_ - kOffset);
            uint64_t scaleX1GmOffset;
            if constexpr (transA) {
                scaleX1GmOffset = scaleOffsetX1 + kOffset * scaleM_;
            } else {
                scaleX1GmOffset = scaleOffsetX1 + kOffset;
            }
            ptScaleUbAddr = kOffset != 0 ? CopyInX1Scale<PtScaleType, false>(scaleX1GmOffset, block_->ubParams_.validM,
                                                                             kElem, mmPertokenScaleUb) :
                                           CopyInX1Scale<PtScaleType, true>(scaleX1GmOffset, block_->ubParams_.validM,
                                                                            kElem, mmPertokenScaleUb);
        }

        WaitForCube(crossPingPongID_);
        auto mmUbInputAddr = crossPingPongID_ == 0 ? reinterpret_cast<__ubuf__ CType*>(mmResPing_->GetPhyAddr()) :
                                                     reinterpret_cast<__ubuf__ CType*>(mmResPong_->GetPhyAddr());
        if (kb == 0) {
            AivPerTensor<true>(mmAddUbAddr, mmUbInputAddr, ptScaleUbAddr, block_->ubParams_.validM,
                               block_->ubParams_.validN, kElem, block_->ubParams_.singleN, scaleX2, pertokenIdx);
        } else {
            AivPerTensor<false>(mmAddUbAddr, mmUbInputAddr, ptScaleUbAddr, block_->ubParams_.validM,
                                block_->ubParams_.validN, kElem, block_->ubParams_.singleN, scaleX2, pertokenIdx);
        }
        NotifyCube(crossPingPongID_);
        crossPingPongID_ = (crossPingPongID_ + 1) & 1;
    }
    vecQuePertokenScale_.FreeTensor(mmPertokenScaleUb);
    vecQueAdd_.EnQue(mmAddUb);
    mmAddUb = vecQueAdd_.template DeQue<CType>();
    AivPostProcess(mmAddUb);
    vecQueAdd_.FreeTensor(mmAddUb);
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <bool isFirstKLoop>
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::AivPerTensor(
    __ubuf__ CType* dst, __ubuf__ CType* l0cOut, __ubuf__ PtScaleType* x1Scale, uint16_t mSize, uint16_t nSize,
    uint16_t kSize, uint32_t nSrcUbAligned, ScaleType x2Scale, uint64_t ptIdx)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(CType);
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    uint16_t alignM = Align(mSize, GMM_UB_ALIGN_SIZE / sizeof(PtScaleType));
    uint16_t alignK = Align(kSize, GMM_UB_ALIGN_SIZE / sizeof(PtScaleType));
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            AscendC::MicroAPI::RegTensor<PtScaleType> ptScaleReg;
            AscendC::MicroAPI::RegTensor<PtScaleType> muledScaleReg;
            if constexpr (transA) {
                AscendC::MicroAPI::DataCopy<PtScaleType, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    ptScaleReg, x1Scale + ptIdx * alignM + mIdx);
            } else {
                AscendC::MicroAPI::DataCopy<PtScaleType, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    ptScaleReg, x1Scale + mIdx * alignK + ptIdx);
            }
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<CType>(elementNum);
                AscendC::MicroAPI::RegTensor<CType> l0cOutReg;
                AscendC::MicroAPI::RegTensor<CType> addReg;
                AscendC::MicroAPI::RegTensor<CType> ResReg, mulScaleOutReg;
                // copy input from ub to register, addr of ub should align to 32B
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
                // l0c_out * scale
                AscendC::MicroAPI::Muls(muledScaleReg, ptScaleReg, x2Scale, maskN);
                AscendC::MicroAPI::Mul(mulScaleOutReg, l0cOutReg, muledScaleReg, maskN);
                uint32_t dstUbOffset = l0cOutOffset;
                if constexpr (isFirstKLoop) {
                    AscendC::MicroAPI::DataCopy<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                        dst + dstUbOffset, mulScaleOutReg, maskN);
                } else {
                    AscendC::MicroAPI::DataCopy(addReg, dst + l0cOutOffset);
                    AscendC::MicroAPI::Add(ResReg, mulScaleOutReg, addReg, maskN);
                    // copy out from register to ub
                    AscendC::MicroAPI::DataCopy<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                    ResReg, maskN);
                }
            }
        }
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::AivPostProcess(AscendC::LocalTensor<CType>& mmAddUb)
{
    if (block_->ubParams_.validM == 0 || block_->ubParams_.validN == 0) {
        return;
    }
    // Output is transferred to GM in 4 batches
    uint16_t splitNumOfOut = block_->ubParams_.validM >= 4 ? 4 : block_->ubParams_.validM;
    uint64_t mSizeForOnce = Act::Gemm::CeilDiv(block_->ubParams_.validM, static_cast<uint64_t>(splitNumOfOut));
    splitNumOfOut = Act::Gemm::CeilDiv(block_->ubParams_.validM, mSizeForOnce);
    for (uint16_t i = 0; i < splitNumOfOut; i++) {
        uint64_t mLeft = block_->ubParams_.validM - i * mSizeForOnce;
        uint64_t mSize = mLeft >= mSizeForOnce ? mSizeForOnce : mLeft;

        AscendC::LocalTensor<YType> mmRes;
        if constexpr (!AscendC::IsSameType<YType, CType>::value) {
            mmRes = vecQueOut_.template AllocTensor<YType>();
            AscendC::Cast(mmRes, mmAddUb[i * mSizeForOnce * block_->ubParams_.singleN], AscendC::RoundMode::CAST_RINT,
                          mSize * block_->ubParams_.singleN);
            vecQueOut_.EnQue(mmRes);
            mmRes = vecQueOut_.template DeQue<YType>();
        } else {
            mmRes = mmAddUb[i * mSizeForOnce * block_->ubParams_.singleN];
        }

        AscendC::DataCopyExtParams copyParams{0, 0, 0, 0, 0};
        copyParams.blockCount = static_cast<uint16_t>(mSize);
        copyParams.blockLen = static_cast<uint32_t>(block_->ubParams_.validN * sizeof(YType));
        copyParams.srcStride = (block_->ubParams_.singleN - block_->ubParams_.validN) * sizeof(YType) / GMM_DATA_BLOCK;
        copyParams.dstStride = (static_cast<uint64_t>(n_) - block_->ubParams_.validN) * sizeof(YType);
        AscendC::DataCopyPad<YType>(cGlobal_[block_->ubParams_.offsetC + i * mSizeForOnce * n_], mmRes, copyParams);
        if constexpr (!AscendC::IsSameType<YType, CType>::value) {
            vecQueOut_.FreeTensor(mmRes);
        }
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <typename T>
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::InitOutputWithZero(uint64_t ySize, int32_t usedCoreNum,
                                                                                bool& isKZeroInit)
{
    if ASCEND_IS_AIC {
        return;
    }
    // Shared bandwidth transfer: Only need to move aiv0's 0 to GM, also support aic:aiv=1:1 cases
    if (AscendC::GetSubBlockIdx() >= 1) {
        return;
    }
    AscendC::GlobalTensor<YType> yInitGlobal;
    yInitGlobal.SetGlobalBuffer(GetTensorAddr<YType>(0, yTensorPtr_) + block_->params_.cGroupAddrOffset);
    uint32_t blockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    // Fetch values following the InitOutput interface pattern
    // Maximum number of elements storable in output dtype
    uint64_t initSize = (GMM_MAX_REPEAT_TIMES * AscendC::ONE_BLK_SIZE) / sizeof(T);
    uint64_t perCoreSize = Act::Gemm::CeilDiv(ySize, usedCoreNum);
    perCoreSize = Act::Gemm::Align(perCoreSize * sizeof(T), static_cast<uint64_t>(GMM_UB_ALIGN_SIZE)) / sizeof(T);
    initSize = Act::Gemm::Min(initSize, perCoreSize);
    uint64_t realCoreNum = Act::Gemm::Min(Act::Gemm::CeilDiv(ySize, initSize), static_cast<uint64_t>(usedCoreNum));
    if (blockIdx >= realCoreNum) { // Return excess cores, minimum 32B alignment per core
        return;
    }
    if (!isKZeroInit) { // Initialize all buffers in ub to zero when k==0 (first iter)
        AscendC::Duplicate<T>(initLocal_, 0, initSize);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        isKZeroInit = true;
    }
    uint64_t yOffset = perCoreSize * blockIdx;
    uint64_t outCurSize = (blockIdx == realCoreNum - 1) ? (ySize - yOffset) : perCoreSize;
    uint64_t movRound = outCurSize / initSize;
    uint64_t movTail = outCurSize - movRound * initSize;

    AscendC::DataCopyExtParams ub2GmParams{1, static_cast<uint32_t>(initSize * sizeof(T)), 0, 0, 0};
    for (uint64_t i = 0; i < movRound; ++i) {
        AscendC::DataCopyPad(yInitGlobal[yOffset], initLocal_, ub2GmParams);
        yOffset += initSize;
    }
    if (movTail != 0) { // mov tail zero data
        ub2GmParams.blockLen = static_cast<uint32_t>(movTail * sizeof(T));
        AscendC::DataCopyPad(yInitGlobal[yOffset], initLocal_, ub2GmParams);
    }
}
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif // ACT_BLOCK_EPILOGUE_PERTILE_H
