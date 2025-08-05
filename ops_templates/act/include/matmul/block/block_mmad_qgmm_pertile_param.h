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
 * \file block_mmad_qgmm_pertile_param.h
 * \brief
 */

#ifndef ACT_BLOCK_QGMM_PERTILE_PARAM_H
#define ACT_BLOCK_QGMM_PERTILE_PARAM_H

#include "include/utils/common_utils.h"
#include "include/utils/grouped_matmul_constant.h"

namespace Act {
namespace Gemm {
namespace Block {

using namespace Act::Gemm::GroupedMatmul;

template <bool aTrans, bool bTrans, class blockType>
class MatMulCommonParam {
public:
    __aicore__ inline MatMulCommonParam(){};
    __aicore__ inline void Init(blockType* block, const TCubeTiling* tilingData);
    __aicore__ inline void UpdateForNextGroup(int32_t m, int32_t n, int32_t k);
    __aicore__ inline uint64_t CalcAGMOffsetInnerLoop(const uint64_t mOffset, const uint64_t kOffset);
    __aicore__ inline uint64_t CalcBGMOffsetInnerLoop(const uint64_t nOffset, const uint64_t kOffset);
    __aicore__ inline void CalNd2NzParamA(AscendC::Nd2NzParams& nd2nzParam, const bool isTailAL1);
    __aicore__ inline void CalNd2NzParamB(AscendC::Nd2NzParams& nd2nzParam, const bool isTailBL1);
    __aicore__ inline uint32_t CalcAL1Offset(const uint64_t mAL1Offset, const uint64_t kAL1Offset, const bool isKTail);
    __aicore__ inline uint32_t CalcBL1Offset(const uint64_t nBL1Offset, const uint64_t kBL1Offset, const bool isKTail);
    __aicore__ inline void LoadData2dParamsA(AscendC::LoadData2DParamsV2& loadData2dParams, const uint64_t kOffset,
                                             const bool isTailAL1);
    __aicore__ inline void LoadData2dParamsB(AscendC::LoadData2DParamsV2& loadData2dParams, const uint64_t kOffset,
                                             const bool isTailBL1);

protected:
    blockType* block_;
    const TCubeTiling* mmTilingData_;
    uint64_t kA1_;
    uint64_t kB1_;
    uint64_t mA1C0_;
    uint64_t nB1C0_;
    uint64_t kB1C0_;
    uint64_t kA1C0_;
    uint64_t kA1Tail_;
    uint64_t kB1Tail_;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
};

template <bool aTrans, bool bTrans, class blockType>
__aicore__ inline void MatMulCommonParam<aTrans, bTrans, blockType>::Init(blockType* block,
                                                                          const TCubeTiling* tilingData)
{
    mmTilingData_ = tilingData;
    block_ = block;

    if constexpr (bTrans) {
        nB1C0_ = GMM_BMM_BLOCK_NUM;
        kB1C0_ = GMM_K0_INT8;
    } else {
        nB1C0_ = GMM_K0_INT8;
        kB1C0_ = GMM_BMM_BLOCK_NUM;
    }
    if constexpr (aTrans) {
        kA1C0_ = GMM_BMM_BLOCK_NUM;
        mA1C0_ = GMM_K0_INT8;
    } else {
        kA1C0_ = GMM_K0_INT8;
        mA1C0_ = GMM_BMM_BLOCK_NUM;
    }
    kA1_ = mmTilingData_->baseK * mmTilingData_->stepKa;
    kB1_ = mmTilingData_->baseK * mmTilingData_->stepKb;
}

template <bool aTrans, bool bTrans, class blockType>
__aicore__ inline void MatMulCommonParam<aTrans, bTrans, blockType>::UpdateForNextGroup(int32_t m, int32_t n, int32_t k)
{
    m_ = m;
    n_ = n;
    k_ = k;

    kB1Tail_ = k_ % kB1_ == 0 ? kB1_ : k_ % kB1_;
    kA1Tail_ = k_ % kA1_ == 0 ? kA1_ : k_ % kA1_;
}

template <bool aTrans, bool bTrans, class blockType>
__aicore__ inline uint64_t MatMulCommonParam<aTrans, bTrans, blockType>::CalcAGMOffsetInnerLoop(const uint64_t mOffset,
                                                                                                const uint64_t kOffset)
{
    uint64_t offsetA = block_->offset_.offsetA;
    if constexpr (aTrans) {
        offsetA += kOffset * m_ + mOffset;
    } else {
        offsetA += kOffset + mOffset * k_;
    }
    return offsetA;
}

template <bool aTrans, bool bTrans, class blockType>
__aicore__ inline uint64_t MatMulCommonParam<aTrans, bTrans, blockType>::CalcBGMOffsetInnerLoop(const uint64_t nOffset,
                                                                                                const uint64_t kOffset)
{
    uint64_t offsetB = block_->offset_.offsetB;
    if constexpr (bTrans) {
        offsetB += kOffset + nOffset * k_;
    } else {
        offsetB += kOffset * n_ + nOffset;
    }
    return offsetB;
}

template <bool aTrans, bool bTrans, class blockType>
__aicore__ inline void MatMulCommonParam<aTrans, bTrans, blockType>::CalNd2NzParamA(AscendC::Nd2NzParams& nd2nzParam,
                                                                                    const bool isTailAL1)
{
    uint64_t currentK = isTailAL1 ? kA1Tail_ : kA1_;
    nd2nzParam.ndNum = 1;
    nd2nzParam.srcNdMatrixStride = 0;
    nd2nzParam.dstNzNStride = 1;
    nd2nzParam.dstNzMatrixStride = 0;
    if constexpr (aTrans) {
        nd2nzParam.nValue = currentK;
        nd2nzParam.dValue = block_->params_.singleCoreM;
        nd2nzParam.srcDValue = m_;
        nd2nzParam.dstNzC0Stride = Align(currentK, static_cast<uint64_t>(GMM_DATA_BLOCK)); // Align to 32-byte boundary
    } else {
        nd2nzParam.nValue = block_->params_.singleCoreM;
        nd2nzParam.dValue = currentK;
        nd2nzParam.srcDValue = k_;
        nd2nzParam.dstNzC0Stride = Align(block_->params_.singleCoreM, static_cast<uint64_t>(GMM_k0_FLOAT16));
    }
}

template <bool aTrans, bool bTrans, class blockType>
__aicore__ inline void MatMulCommonParam<aTrans, bTrans, blockType>::CalNd2NzParamB(AscendC::Nd2NzParams& nd2nzParam,
                                                                                    const bool isTailBL1)
{
    uint64_t currentK = isTailBL1 ? kB1Tail_ : kB1_;
    nd2nzParam.ndNum = 1;
    nd2nzParam.srcNdMatrixStride = 0;
    nd2nzParam.dstNzNStride = 1;
    nd2nzParam.dstNzMatrixStride = 0;
    if constexpr (bTrans) {
        nd2nzParam.nValue = block_->params_.singleCoreN;
        nd2nzParam.dValue = currentK;
        nd2nzParam.srcDValue = k_;
        nd2nzParam.dstNzC0Stride = Align(block_->params_.singleCoreN, static_cast<uint64_t>(GMM_k0_FLOAT16));
    } else {
        nd2nzParam.nValue = currentK;
        nd2nzParam.dValue = block_->params_.singleCoreN;
        nd2nzParam.srcDValue = n_;
        nd2nzParam.dstNzC0Stride = Align(currentK, static_cast<uint64_t>(GMM_DATA_BLOCK)); // Align to 32-byte boundary
    }
}

template <bool aTrans, bool bTrans, class blockType>
__aicore__ inline uint32_t MatMulCommonParam<aTrans, bTrans, blockType>::CalcAL1Offset(const uint64_t mAL1Offset,
                                                                                       const uint64_t kAL1Offset,
                                                                                       const bool isKTail)
{
    uint64_t kAL1 = isKTail ? kA1Tail_ : kA1_;
    kAL1 = Align(kAL1, static_cast<uint64_t>(GMM_K0_INT8));
    uint64_t mAL1 = Align(block_->params_.singleCoreM, mA1C0_);
    if constexpr (aTrans) {
        // (m1, k1, k0, m0)
        return Align(mAL1Offset, mA1C0_) * kAL1 + kAL1Offset * mA1C0_;
    } else {
        // (k1, m1, m0, k0)
        return Align(kAL1Offset, kA1C0_) * mAL1 + mAL1Offset * kA1C0_;
    }
}

template <bool aTrans, bool bTrans, class blockType>
__aicore__ inline uint32_t MatMulCommonParam<aTrans, bTrans, blockType>::CalcBL1Offset(const uint64_t nBL1Offset,
                                                                                       const uint64_t kBL1Offset,
                                                                                       const bool isKTail)
{
    uint64_t kBL1 = isKTail ? kB1Tail_ : kB1_;
    kBL1 = Align(kBL1, static_cast<uint64_t>(GMM_K0_INT8));
    uint64_t nBL1 = Align(block_->params_.singleCoreN, nB1C0_);
    if constexpr (bTrans) {
        // (k1, n1, n0, k0)
        return Align(kBL1Offset, kB1C0_) * nBL1 + nBL1Offset * kB1C0_;
    } else {
        // (n1, k1, k0, n0)
        return Align(nBL1Offset, nB1C0_) * kBL1 + kBL1Offset * nB1C0_;
    }
}

template <bool aTrans, bool bTrans, class blockType>
__aicore__ inline void
MatMulCommonParam<aTrans, bTrans, blockType>::LoadData2dParamsA(AscendC::LoadData2DParamsV2& loadData2dParams,
                                                                const uint64_t kOffset, const bool isTailAL1)
{
    uint64_t currM = Min(block_->params_.singleCoreM, static_cast<uint64_t>(mmTilingData_->baseM));
    uint64_t currK = Min(k_ - kOffset, static_cast<uint64_t>(mmTilingData_->baseK));
    if constexpr (aTrans) {
        // For b8 input in transpose scenarios: use two 16x32 fractals
        loadData2dParams.mStep = Align(Act::Gemm::CeilDiv(currK, static_cast<uint64_t>(GMM_k0_FLOAT16)), 2UL);
        loadData2dParams.kStep = Act::Gemm::CeilDiv(currM, static_cast<uint64_t>(GMM_K0_INT8));
        loadData2dParams.srcStride = Align(Act::Gemm::CeilDiv(isTailAL1 ? kA1Tail_ : kA1_, GMM_k0_FLOAT16), 2UL);
        loadData2dParams.dstStride = Align(Act::Gemm::CeilDiv(currM, static_cast<uint64_t>(GMM_k0_FLOAT16)), 2UL);
        loadData2dParams.ifTranspose = true;
    } else {
        loadData2dParams.mStep = Act::Gemm::CeilDiv(currM, static_cast<uint64_t>(GMM_k0_FLOAT16));
        loadData2dParams.kStep = Act::Gemm::CeilDiv(currK, static_cast<uint64_t>(GMM_K0_INT8));
        loadData2dParams.srcStride =
            Act::Gemm::CeilDiv(currM * mmTilingData_->stepM, static_cast<uint64_t>(GMM_k0_FLOAT16));
        loadData2dParams.dstStride = Act::Gemm::CeilDiv(currM, static_cast<uint64_t>(GMM_k0_FLOAT16));
    }
}

template <bool aTrans, bool bTrans, class blockType>
__aicore__ inline void
MatMulCommonParam<aTrans, bTrans, blockType>::LoadData2dParamsB(AscendC::LoadData2DParamsV2& loadData2dParams,
                                                                const uint64_t kOffset, const bool isTailBL1)
{
    uint64_t currN = Min(block_->params_.singleCoreN, static_cast<uint64_t>(mmTilingData_->baseN));
    uint64_t currK = Min(k_ - kOffset, static_cast<uint64_t>(mmTilingData_->baseK));
    if constexpr (bTrans) {
        loadData2dParams.mStep = Act::Gemm::CeilDiv(currN, static_cast<uint64_t>(GMM_k0_FLOAT16));
        loadData2dParams.kStep = Act::Gemm::CeilDiv(currK, static_cast<uint64_t>(GMM_K0_INT8));
        loadData2dParams.srcStride =
            Act::Gemm::CeilDiv(currN * mmTilingData_->stepN, static_cast<uint64_t>(GMM_k0_FLOAT16));
        loadData2dParams.dstStride = Act::Gemm::CeilDiv(currN, static_cast<uint64_t>(GMM_k0_FLOAT16));
    } else {
        loadData2dParams.ifTranspose = true;
        // For b8 input in transpose scenarios: use two 16x32 fractals
        loadData2dParams.mStep = Align(Act::Gemm::CeilDiv(currK, static_cast<uint64_t>(GMM_k0_FLOAT16)), 2UL);
        loadData2dParams.kStep = Act::Gemm::CeilDiv(currN, static_cast<uint64_t>(GMM_K0_INT8));
        loadData2dParams.srcStride = Align(Act::Gemm::CeilDiv(isTailBL1 ? kB1Tail_ : kB1_, GMM_k0_FLOAT16), 2UL);
        loadData2dParams.dstStride = Align(Act::Gemm::CeilDiv(currN, static_cast<uint64_t>(GMM_k0_FLOAT16)), 2UL);
    }
}
} // namespace Block
} // namespace Gemm
} // namespace Act

#endif