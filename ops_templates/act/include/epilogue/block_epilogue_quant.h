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
 * \file block_epilogue_quant_matmul.h
 * \brief
 */

#ifndef ACT_BLOCK_EPILOGUE_QUANT_H
#define ACT_BLOCK_EPILOGUE_QUANT_H
#include "kernel_operator.h"
#include "include/utils/common_utils.h"
#include "include/utils/device_utils.h"
#include "include/epilogue/fusion/default_fusion_op.h"
#include "include/utils/status_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
struct DequantParams {
    uint32_t m;
    uint32_t n;
    uint32_t calCount;
};
constexpr int64_t ONE_BLK_SIZE = 32;
constexpr int64_t UB_EXTRE_BYTE = 8;
constexpr int64_t UB_BF16_ALIGN_NUM = 16;
constexpr int64_t M_N_TWO_DIMS = 2;

template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_,
          typename FusionOp_ = DefaultFusion<DataTypeOut_, DataTypeIn_>>
class BlockEpilogueQuant {
public:
    using FusionArguments = typename FusionOp_::Arguments;
    using FusionParams = typename FusionOp_::Params;

    __aicore__ inline BlockEpilogueQuant() {}

    struct Arguments {
        GM_ADDR scaleGmAddr{nullptr};
        GM_ADDR pertokenScaleGmAddr{nullptr};
        GM_ADDR inGmAddr{nullptr};
        GM_ADDR outGmAddr{nullptr};
        FusionArguments fusionArgs{};
    };

    struct Params {
        GM_ADDR scaleGmAddr{nullptr};
        GM_ADDR pertokenScaleGmAddr{nullptr};
        GM_ADDR inGmAddr{nullptr};
        GM_ADDR outGmAddr{nullptr};
        FusionParams fusionParams{};
    };

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using FusionOp = FusionOp_;
    using DataTypeScale = float;
    using DataTypePertokenScale = float;

    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();

    // shape
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    // GM ADDR
    AscendC::GlobalTensor<DataTypeScale> scaleGlobal_;
    AscendC::GlobalTensor<DataTypePertokenScale> pertokenScaleGlobal_;
    AscendC::GlobalTensor<DataTypeIn> inGlobal_;
    AscendC::GlobalTensor<DataTypeOut> outGlobal_;
    // PIPE
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_COUNT> vecQueSrc_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_COUNT> vecQueOut_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_COUNT> vecQueScale_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_COUNT> vecQuePertokenScale_;
    TBuf<TPosition::VECCALC> vecQueTmp_;
    TBuf<TPosition::VECCALC> outFp32Tmp_;
    TBuf<TPosition::VECCALC> broadcastFp32Tmp_;
    // attribute
    FusionOp fusionOp_;
    ProblemShape problemShape_;
    BlockShape blockShape_;
    int64_t ubCalcM_{0};
    int64_t ubCalcN_{0};
    int64_t needUbBuffer_{0};
    int64_t l1M_{0};
    int64_t l1N_{0};

    __aicore__ inline void CalcUbTiling(const int64_t l1M, const int64_t l1N)
    {
        ubCalcN_ = l1N;
        // veccalc: Temporary space for Broadcast, min 256b, max align(ubM, 8) * 32b
        // Compute with baseM first (baseM ≤ 2048), no multiplication overflow check needed
        // 7: to comfirm that pertokenScale 32B(8, fp32) aligned up to 7, eg: 1->8
        int64_t needUbSize = l1M * ONE_BLK_SIZE + DOUBLE_BUFFER_COUNT * sizeof(float) * 7;
        // input src + output dst + veccalc dequant api
        int64_t ubSizeOneM =
            (DOUBLE_BUFFER_COUNT * (sizeof(DataTypeIn) + sizeof(DataTypeOut)) + UB_EXTRE_BYTE) * ubCalcN_;
        // scale perchannel
        ubSizeOneM += DOUBLE_BUFFER_COUNT * sizeof(DataTypeScale) * ubCalcN_;
        // pertoken dequant api dst fp32
        ubSizeOneM += sizeof(float) * ubCalcN_;
        // scale pertoken fp32
        ubSizeOneM += DOUBLE_BUFFER_COUNT * sizeof(float);
        // broadcast fp32
        ubSizeOneM += sizeof(float) * ubCalcN_;
        // fusionOp
        ubSizeOneM += fusionOp_.GetUbSizeOneM(ubCalcN_);
        ubCalcM_ = (TOTAL_UB_SIZE - needUbSize) / ubSizeOneM;
        return;
    }
    __aicore__ inline void InitBuffers()
    {
        GetTPipePtr()->InitBuffer(vecQueSrc_, DOUBLE_BUFFER_COUNT, ubCalcM_ * ubCalcN_ * sizeof(DataTypeIn));
        GetTPipePtr()->InitBuffer(vecQueOut_, DOUBLE_BUFFER_COUNT, ubCalcM_ * ubCalcN_ * sizeof(DataTypeOut));
        GetTPipePtr()->InitBuffer(vecQueTmp_, needUbBuffer_);
        GetTPipePtr()->InitBuffer(vecQueScale_, DOUBLE_BUFFER_COUNT, ubCalcN_ * sizeof(DataTypeScale));
        GetTPipePtr()->InitBuffer(outFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
        GetTPipePtr()->InitBuffer(vecQuePertokenScale_, DOUBLE_BUFFER_COUNT, CeilAlign(ubCalcM_, 8U) * sizeof(float));
        GetTPipePtr()->InitBuffer(broadcastFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
    }
    __aicore__ inline void Init(Params const& params, int64_t l1M, int64_t l1N, ProblemShape& problemShape)
    {
        scaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeScale*>(params.scaleGmAddr));
        pertokenScaleGlobal_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataTypePertokenScale*>(params.pertokenScaleGmAddr));
        inGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeIn*>(params.inGmAddr));
        outGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut*>(params.outGmAddr));

        problemShape_ = problemShape;
        CalcUbTiling(l1M, l1N);
        l1M_ = l1M;
        l1N_ = l1N;
        needUbBuffer_ = UB_EXTRE_BYTE * ubCalcM_ * ubCalcN_;
        InitBuffers();
        fusionOp_.Init(params.fusionParams, l1M, l1N, Get<MNK_N>(problemShape_));
    }
    __aicore__ inline void CalcPerChannelDequantParams(DequantParams& dequantParams, int64_t curAivM, int64_t curAivN)
    {
        int64_t computedAivN = CeilAlign(curAivN, UB_FLOAT_ALIGN_NUM); // 8: 32B aligned for int32_t
        int64_t ubResAlignedN = CeilAlign(curAivN, UB_BF16_ALIGN_NUM); // 16: sizeof(DataTypeOut) is 2, 32B / 2
        if (computedAivN == ubResAlignedN) {
            // choose ddequat high performance
            dequantParams.m = 1;
            dequantParams.n = curAivM * computedAivN;
            dequantParams.calCount = computedAivN;
        } else {
            // general
            dequantParams.m = curAivM;
            dequantParams.n = computedAivN;
            dequantParams.calCount = curAivN;
        }
    }

    template <typename T>
    __aicore__ inline void CopyTensorGM2UB(LocalTensor<T> srcLocal, GlobalTensor<T> mmOutGM, const int64_t realM,
                                           const int64_t realN)
    {
        DataCopyParams gm2UbParams{1, 0, 0, 0};
        DataCopyPadParams padParams;
        // datacopypad 32B aligned
        gm2UbParams.blockLen = realN * sizeof(T);
        gm2UbParams.blockCount = realM;
        gm2UbParams.srcStride = 0;
        DataCopyPad(srcLocal, mmOutGM, gm2UbParams, padParams);
    }

    template <typename T>
    __aicore__ inline void CopyTensorUB2GM(GlobalTensor<T> tensorCGM, LocalTensor<T> dstLocal, const int64_t realM,
                                           const int64_t realN, const int64_t strideN)
    {
        DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        ub2GmParams.blockLen = realN * sizeof(T);
        ub2GmParams.blockCount = realM;
        ub2GmParams.dstStride = (strideN - realN) * sizeof(T);
        DataCopyPad(tensorCGM, dstLocal, ub2GmParams);
    }

    __aicore__ inline void run(BlockCoord& blockCoord, int64_t dstStartOffset, int64_t srcStartOffset)
    {
        int64_t blockShapeM = Get<0>(blockShape_);
        int64_t blockShapeN = Get<1>(blockShape_);
        int64_t mUbTileNum = Act::Gemm::CeilDiv(blockShapeM, ubCalcM_);
        for (int64_t mUbTileIdx = 0; mUbTileIdx < mUbTileNum; ++mUbTileIdx) {
            int64_t curUbM = mUbTileIdx == mUbTileNum - 1 ? blockShapeM - ubCalcM_ * (mUbTileNum - 1) : ubCalcM_;
            LocalTensor<DataTypeIn> srcLocal = vecQueSrc_.AllocTensor<DataTypeIn>();
            LocalTensor<DataTypeOut> dstLocal = vecQueOut_.AllocTensor<DataTypeOut>();
            int64_t srcOffset = mUbTileIdx * ubCalcM_ * blockShapeN;
            int64_t dstOffset = dstStartOffset + mUbTileIdx * ubCalcM_ * Get<MNK_N>(problemShape_);
            CopyTensorGM2UB(srcLocal, inGlobal_[srcStartOffset + srcOffset], curUbM, blockShapeN);
            TPipeSetWaitFlag<HardEvent::MTE2_V>();
            CalcUb2Ub(dstLocal, srcLocal, curUbM, blockShapeN, blockCoord, mUbTileIdx);
            vecQueSrc_.FreeTensor(srcLocal);
            TPipeSetWaitFlag<HardEvent::V_MTE3>();
            CopyTensorUB2GM(outGlobal_[dstOffset], dstLocal, curUbM, blockShapeN, Get<MNK_N>(problemShape_));
            vecQueOut_.FreeTensor(dstLocal);
        }
    }

    __aicore__ inline void CalcUb2Ub(LocalTensor<DataTypeOut> dstLocal, LocalTensor<DataTypeIn> srcLocal,
                                     int64_t curAivM, int64_t curAivN, BlockCoord blockCoord, int64_t mUbTileIdx)
    {
        int64_t mTotalIdx = Get<0>(blockCoord) + mUbTileIdx * ubCalcM_;
        int64_t nTotalIdx = Get<1>(blockCoord);
        DequantParams dequantParams;
        CalcPerChannelDequantParams(dequantParams, curAivM, curAivN);
        LocalTensor<float> dstLocalFp32 = outFp32Tmp_.Get<float>();
        LocalTensor<uint8_t> tmpLocal = vecQueTmp_.Get<uint8_t>();
        // 1. per channel dequant
        AscendC::DataCopyParams scale2UbParams{1, static_cast<uint16_t>(curAivN * sizeof(DataTypeScale)), 0, 0};
        LocalTensor<DataTypeScale> scaleLocal = vecQueScale_.AllocTensor<DataTypeScale>();
        DataCopyPadParams padParams;
        DataCopyPad(scaleLocal, scaleGlobal_[nTotalIdx], scale2UbParams, padParams);
        TPipeSetWaitFlag<HardEvent::MTE2_V>();
        AscendDequant(dstLocalFp32, srcLocal, scaleLocal, tmpLocal,
                      {dequantParams.m, dequantParams.n, dequantParams.calCount});
        vecQueScale_.FreeTensor(scaleLocal);
        // 2. per token dequant
        LocalTensor<float> pertokenScaleLocal = vecQuePertokenScale_.AllocTensor<float>();
        scale2UbParams.blockLen = curAivM * sizeof(DataTypePertokenScale);
        DataCopyPad(pertokenScaleLocal, pertokenScaleGlobal_[mTotalIdx], scale2UbParams, padParams);
        TPipeSetWaitFlag<HardEvent::MTE2_V>();
        // 2.1. broadcast
        LocalTensor<float> broadcastFp32 = broadcastFp32Tmp_.Get<float>();
        int64_t computedAivN = CeilAlign(curAivN, UB_FLOAT_ALIGN_NUM); // 8: 32B aligned for int32_t
        int64_t ubResAlignedN = CeilAlign(curAivN, UB_BF16_ALIGN_NUM); // 16: sizeof(DataTypeOut) is 2, 32B / 2
        // broadcast from [m, 1] to [m, n]
        const uint32_t broadcastDst[M_N_TWO_DIMS]{static_cast<uint32_t>(curAivM), static_cast<uint32_t>(computedAivN)};
        const uint32_t broadcastSrc[M_N_TWO_DIMS]{static_cast<uint32_t>(curAivM), 1};
        BroadCast<float, M_N_TWO_DIMS, 1>(broadcastFp32, pertokenScaleLocal, broadcastDst, broadcastSrc);
        vecQuePertokenScale_.FreeTensor(pertokenScaleLocal);
        AscendC::PipeBarrier<PIPE_V>();
        // 2.2 mul to do per token dequant
        LocalTensor<float> tmpDstLocal = vecQueTmp_.Get<float>();
        if (computedAivN == ubResAlignedN) {
            Mul(tmpDstLocal, broadcastFp32, dstLocalFp32, computedAivN * curAivM);
        } else {
            for (auto mIdx = 0; mIdx < curAivM; mIdx++) {
                Mul(tmpDstLocal[ubResAlignedN * mIdx], broadcastFp32[computedAivN * mIdx],
                    dstLocalFp32[computedAivN * mIdx], computedAivN);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        // 3. cast from fp32 to outputDataType
        Cast(dstLocal, tmpDstLocal, RoundMode::CAST_RINT, curAivM * ubResAlignedN);
        fusionOp_(dstLocal, dstLocal, curAivM, curAivN, mTotalIdx, nTotalIdx);
        return;
    }

    __aicore__ inline void operator()(BlockShape& blockShape, BlockCoord& blockCoord, int64_t dstStartOffset = 0,
                                      int64_t srcStartOffset = 0)
    {
        blockShape_ = blockShape;
        run(blockCoord, dstStartOffset, srcStartOffset);
        return;
    }

    // static init
    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR workspaceGm)
    {
        Params params = {args.scaleGmAddr, args.pertokenScaleGmAddr, workspaceGm, args.outGmAddr, {}};
        return params;
    }

    __host_aicore__ static size_t GetWorkSpaceSize(int64_t blockNum, int64_t l1M, int64_t l1N)
    {
        // only quant kernel need workspace
        size_t worksapceSize = blockNum * DOUBLE_BUFFER_COUNT * l1M * l1N * sizeof(int32_t);
        return worksapceSize;
    }

    __host_aicore__ static Status CheckArgs(Arguments const& args)
    {
        if (l0M % MATMUL_MNK_ALIGN_INT8 != 0 || l0N % MATMUL_MNK_ALIGN_INT8 != 0) {
            return Status::l1L0ErrorNotAlignInt8;
        }
        return Status::success;
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif // ACT_BLOCK_EPILOGUE_QUANT_H