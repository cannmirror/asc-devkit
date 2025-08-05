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
 * \file block_mmad_multi_block_on_kaxis_with_layout.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_MULTI_BLOCK_ON_KAXIS_WITH_LAYOUT_H
#define ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_MULTI_BLOCK_ON_KAXIS_WITH_LAYOUT_H

#include "./block_mmad.h"
#include "./block_mmad_utils.h"
#include "../../utils/layout_utils.h"
#include "../../utils/tensor_utils.h"
#include "../../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tile/load_data/tbuf_pool_l0_default.h"
#include "../tile/tile_copy.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_,
          class TileCopy_>
class BlockMmad<MatmulMultiBlockOnKAxisWithLayout<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_,
                TileCopy_> {
public:
    using DispatchPolicy = MatmulMultiBlockOnKAxisWithLayout<>;
    using L1Shape = L1TileShape_;
    using L0Shape = L0TileShape_;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using TileCopy =
        AscendC::Std::conditional_t<AscendC::Std::is_same_v<TileCopy_, void>,
                                    Tile::TileCopy<Arch::Ascend910B, Tile::CopyEnUnitFlagWithLayout>, TileCopy_>;

    using L1ALayout = FormatToLayoutT<typename AType::T, CubeFormat::NZ>;
    using L1BLayout = FormatToLayoutT<typename BType::T, CubeFormat::NZ>;
    using L0ALayout = FormatToLayoutT<typename AType::T, CubeFormat::ZZ>;
    using L0BLayout = FormatToLayoutT<typename BType::T, CubeFormat::ZN>;
    using L0CLayout = AscendC::Layout<AscendC::Shape<AscendC::Shape<_16, int>, // L0C fractal size: fixed 16x16
                                                     AscendC::Shape<_16, int>>,
                                      AscendC::Stride<AscendC::Stride<_16, _256>, AscendC::Stride<_1, int>>>;

    using L1ATensorTrait = AscendC::TensorTrait<typename AType::T, AscendC::TPosition::A1, L1ALayout>;
    using L1BTensorTrait = AscendC::TensorTrait<typename BType::T, AscendC::TPosition::B1, L1BLayout>;
    using L0ATensorTrait = AscendC::TensorTrait<typename AType::T, AscendC::TPosition::A2, L0ALayout>;
    using L0BTensorTrait = AscendC::TensorTrait<typename BType::T, AscendC::TPosition::B2, L0BLayout>;
    using L0CTensorTrait = AscendC::TensorTrait<typename AscendC::GetMmDstType<typename AType::T>::Type,
                                                AscendC::TPosition::CO1, L0CLayout>;

public:
    static_assert(AscendC::Std::is_same_v<TileCopy, Tile::TileCopy<Arch::Ascend910B, Tile::CopyEnUnitFlagWithLayout>>,
                  "Only support CopyEnUnitFlagWithLayout");
    static_assert(IsF16F16F16<AType, BType, CType>() || IsBf16Bf16Bf16<AType, BType, CType>(), "Unsupported dtype");
    static_assert(IsND<AType>() && IsND<BType>() && IsND<CType>(), "Only support ND format");
    static_assert(IsTileShapeValid<L1Shape, L0Shape>(), "L1Shape or L0Shape is invalid");
    static_assert(IsL1BufferValid<AType, BType, L1Shape>(), "L1 buffer overflow");
    static_assert(IsL0BufferValid<AType, BType, L0Shape>(), "L0 buffer overflow");

    __aicore__ BlockMmad() = default;
    __aicore__ ~BlockMmad()
    {
        End();
    }

    __aicore__ inline void Init()
    {
        constexpr static int32_t aL0MatrixByteSize =
            L0_M * L0_K * AscendC::GetBitSize<typename AType::T>() / AscendC::ONE_BYTE_BIT_SIZE;
        constexpr static int32_t bL0MatrixByteSize =
            L0_N * L0_K * AscendC::GetBitSize<typename BType::T>() / AscendC::ONE_BYTE_BIT_SIZE;
        constexpr static int32_t cMatrixByteSize =
            L0_M * L0_N * sizeof(typename AscendC::GetMmDstType<typename AType::T>::Type);

        constexpr static int dbL0AFlag = (aL0MatrixByteSize * DOUBLE_BUFFER_COUNT > L0A_SIZE) ? 1 : DOUBLE_BUFFER_COUNT;
        constexpr static int dbL0BFlag = (bL0MatrixByteSize * DOUBLE_BUFFER_COUNT > L0B_SIZE) ? 1 : DOUBLE_BUFFER_COUNT;

        tbufPoolL0_.Init((dbL0AFlag - 1) & (dbL0BFlag - 1));
        AscendC::SetMMLayoutTransform(0);
        constexpr static int32_t aL1MatrixByteSize =
            L0_M * L1_K * AscendC::GetBitSize<typename AType::T>() / AscendC::ONE_BYTE_BIT_SIZE;
        constexpr static int32_t bL1MatrixByteSize =
            L0_N * L1_K * AscendC::GetBitSize<typename BType::T>() / AscendC::ONE_BYTE_BIT_SIZE;
        constexpr static int dbL1Flag =
            (aL1MatrixByteSize * DOUBLE_BUFFER_COUNT + bL1MatrixByteSize * DOUBLE_BUFFER_COUNT > L1_SIZE) ?
                1 :
                DOUBLE_BUFFER_COUNT;

        auto tpipe = GetTPipePtr();
        uint32_t shareLens[3] = {static_cast<uint32_t>(GetL1UsedSize(dbL1Flag)), cMatrixByteSize, 0};
        InitShareBufStart(tpipe, 0, shareLens, 3, 0); // 3: shareLens num

        tpipe->InitBuffer(qidL1A_, dbL1Flag, aL1MatrixByteSize);
        tpipe->InitBuffer(qidL1B_, dbL1Flag, bL1MatrixByteSize);
        tpipe->InitBuffer(qidL0C_, cMatrixByteSize);

        InitShareBufEnd(tpipe);
    }

    template <class DstTensor, class SrcATensor, class SrcBTensor, class Shape>
    __aicore__ inline void IterateAll(DstTensor& c, const SrcATensor& a, const SrcBTensor& b, const Shape& actualShape)
    {
        static_assert(AscendC::is_global_tensor_v<DstTensor> && AscendC::is_global_tensor_v<SrcATensor> &&
                          AscendC::is_global_tensor_v<SrcBTensor>,
                      "Only support GM in and GM out");
        using DstTrait = typename AscendC::tensor_trait<DstTensor>::trait_type;
        using SrcATrait = typename AscendC::tensor_trait<SrcATensor>::trait_type;
        using SrcBTrait = typename AscendC::tensor_trait<SrcBTensor>::trait_type;
        typename TileCopy::template CopyGmToA1<AType, L1ATensorTrait, SrcATrait> copyGmToA1;
        typename TileCopy::template CopyGmToB1<BType, L1BTensorTrait, SrcBTrait> copyGmToB1;
        typename TileCopy::template CopyCo1ToOut<CType, DstTrait, L0CTensorTrait> copyCo1ToGm;

        int32_t mIter = AscendC::Ceil(Get<MNK_M>(actualShape), L0_M);
        int32_t nIter = AscendC::Ceil(Get<MNK_N>(actualShape), L0_N);
        int32_t kOuterIter = AscendC::Ceil(Get<MNK_K>(actualShape), L1_K);
        int32_t kInnerIter = AscendC::Ceil(L1_K, L0_K);
        int32_t tailBaseM = (Get<MNK_M>(actualShape) % L0_M) == 0 ? L0_M : (Get<MNK_M>(actualShape) % L0_M);
        int32_t tailBaseN = (Get<MNK_N>(actualShape) % L0_N) == 0 ? L0_N : (Get<MNK_N>(actualShape) % L0_N);
        int32_t tailBaseK = (Get<MNK_K>(actualShape) % L0_K) == 0 ? L0_K : (Get<MNK_K>(actualShape) % L0_K);
        int32_t tailL1K = (Get<MNK_K>(actualShape) % L1_K) == 0 ? L1_K : (Get<MNK_K>(actualShape) % L1_K);

        for (auto mIndex = 0; mIndex < mIter; ++mIndex) {
            for (auto nIndex = 0; nIndex < nIter; ++nIndex) {
                int baseM = (mIndex + 1 == mIter) ? tailBaseM : L0_M;
                int baseN = (nIndex + 1 == nIter) ? tailBaseN : L0_N;
                auto l0CLayout = AscendC::MakeLayout(
                    AscendC::MakeShape(AscendC::MakeShape(_16{}, AscendC::Ceil(baseM, 16)),
                                       AscendC::MakeShape(_16{}, AscendC::Ceil(baseN, 16))),
                    AscendC::MakeStride(AscendC::MakeStride(_16{}, _256{}),
                                        AscendC::MakeStride(_1{}, AscendC::CeilAlign(baseM, 16) * 16)));
                auto l0C = qidL0C_.template Get<L0CTensorTrait>();
                l0C.SetTensorTrait(L0CTensorTrait(l0CLayout));

                for (auto kOuterIndex = 0; kOuterIndex < kOuterIter; ++kOuterIndex) {
                    int l1K = (kOuterIndex + 1 == kOuterIter) ? tailL1K : L1_K;
                    // -----------------Step1: GM -> L1 -----------------
                    auto aTileHeight = AType::isTrans ? l1K : baseM;
                    auto aTileWidth = AType::isTrans ? baseM : l1K;
                    auto aL1Row = AType::isTrans ? kOuterIndex : mIndex;
                    auto aL1Col = AType::isTrans ? mIndex : kOuterIndex;

                    auto l1ALayout = MakeLayoutByFormat<typename AType::T, CubeFormat::NZ>(aTileHeight, aTileWidth);
                    auto l1A = qidL1A_.template AllocTensor<L1ATensorTrait>();
                    l1A.SetTensorTrait(L1ATensorTrait(l1ALayout));

                    copyGmToA1(l1A, const_cast<AscendC::GlobalTensor<SrcATrait>&>(a),
                               AscendC::MakeCoord(aL1Row * (AType::isTrans ? L1_K : L0_M),
                                                  aL1Col * (AType::isTrans ? L0_M : L1_K)));

                    qidL1A_.EnQue(l1A);

                    auto bTileHeight = BType::isTrans ? baseN : l1K;
                    auto bTileWidth = BType::isTrans ? l1K : baseN;
                    auto bL1Row = BType::isTrans ? nIndex : kOuterIndex;
                    auto bL1Col = BType::isTrans ? kOuterIndex : nIndex;

                    auto l1BLayout = MakeLayoutByFormat<typename BType::T, CubeFormat::NZ>(bTileHeight, bTileWidth);
                    auto l1B = qidL1B_.template AllocTensor<L1BTensorTrait>();
                    l1B.SetTensorTrait(L1BTensorTrait(l1BLayout));

                    copyGmToB1(l1B, const_cast<AscendC::GlobalTensor<SrcBTrait>&>(b),
                               AscendC::MakeCoord(bL1Row * (BType::isTrans ? L0_N : L1_K),
                                                  bL1Col * (BType::isTrans ? L1_K : L0_N)));

                    qidL1B_.EnQue(l1B);
                    qidL1A_.DeQue();
                    qidL1B_.DeQue();

                    for (auto kInnerIndex = 0; kInnerIndex < kInnerIter; ++kInnerIndex) {
                        int baseK =
                            (kOuterIndex + 1 == kOuterIter) && (kInnerIndex + 1 == kInnerIter) ? tailBaseK : L0_K;

                        // -----------------Step2: L1 -> L0-----------------
                        auto& bufferPool = tbufPoolL0_.Allocate();
                        auto aBaseHeight = AType::isTrans ? baseK : baseM;
                        auto aBaseWidth = AType::isTrans ? baseM : baseK;
                        auto aL0Row = AType::isTrans ? kInnerIndex : mIndex;
                        auto aL0Col = AType::isTrans ? mIndex : kInnerIndex;
                        auto l0ALayout = MakeLayoutByFormat<typename AType::T, CubeFormat::ZZ>(aBaseHeight, aBaseWidth);
                        auto l0A = bufferPool.template GetBuffer<AscendC::TPosition::A2, L0ATensorTrait>();
                        l0A.SetTensorTrait(L0ATensorTrait(l0ALayout));
                        copyA1ToA2_(l0A, l1A,
                                    AscendC::MakeCoord(aL0Row * (AType::isTrans ? L0_K : 0),
                                                       aL0Col * (AType::isTrans ? 0 : L0_K)));

                        auto bBaseHeight = BType::isTrans ? baseN : baseK;
                        auto bBaseWidth = BType::isTrans ? baseK : baseN;
                        auto bL0Row = BType::isTrans ? nIndex : kInnerIndex;
                        auto bL0Col = BType::isTrans ? kInnerIndex : nIndex;
                        auto l0BLayout = MakeLayoutByFormat<typename BType::T, CubeFormat::ZN>(bBaseHeight, bBaseWidth);
                        auto l0B = bufferPool.template GetBuffer<AscendC::TPosition::B2, L0BTensorTrait>();
                        l0B.SetTensorTrait(L0BTensorTrait(l0BLayout));
                        copyB1ToB2_(l0B, l1B,
                                    AscendC::MakeCoord(bL0Row * (BType::isTrans ? 0 : L0_K),
                                                       bL0Col * (BType::isTrans ? L0_K : 0)));

                        bufferPool.EnQue();
                        bufferPool.DeQue();

                        //  -----------------Step3: compute-----------------
                        auto unitFlag = (kOuterIndex + 1 == kOuterIter) && (kInnerIndex + 1 == kInnerIter) ?
                                            UNIT_FLAG_SET :
                                            UNIT_FLAG_CHECK;
                        // GEMV is automatically enabled when setting M=1 in normal mode
                        AscendC::MmadParams mmadParams;
                        mmadParams.m = (baseM == 1 ? ALIGN_NUM : baseM);
                        mmadParams.k = baseK;
                        mmadParams.n = baseN;
                        mmadParams.unitFlag = unitFlag;
                        mmadParams.cmatrixInitVal = (kOuterIndex == 0) && (kInnerIndex == 0) ? true : false;
                        Mmad(l0C, l0A, l0B, mmadParams);
                        if ((baseM / ALIGN_NUM) * (baseN / ALIGN_NUM) < LIMIT_MNSIZE) {
                            AscendC::PipeBarrier<PIPE_M>();
                        }
                        bufferPool.Free();
                    }
                    qidL1A_.FreeTensor(l1A);
                    qidL1B_.FreeTensor(l1B);
                }
                // -----------------Step4: L0C -> GM----------------
                copyCo1ToGm(c, l0C, AscendC::MakeCoord(mIndex * L0_M, nIndex * L0_N));
            }
        }
    }

private:
    __aicore__ inline constexpr int32_t GetL1UsedSize(int32_t dbL1Flag)
    {
        int32_t sharedl1Size = 0;
        if constexpr (!PhyPosIsL1(AType::pos)) {
            // L1 DB ON
            sharedl1Size +=
                dbL1Flag * L0_M * L1_K * AscendC::GetBitSize<typename AType::T>() / AscendC::ONE_BYTE_BIT_SIZE;
        }
        if constexpr (!PhyPosIsL1(BType_::pos)) {
            // L1 DB ON
            sharedl1Size +=
                dbL1Flag * L0_N * L1_K * AscendC::GetBitSize<typename BType::T>() / AscendC::ONE_BYTE_BIT_SIZE;
        }
        return sharedl1Size;
    }

    __aicore__ inline void End()
    {
        qidL1A_.FreeAllEvent();
        qidL1B_.FreeAllEvent();
        event_t eventIDFixToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_M));
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventIDFixToM);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventIDFixToM);
    }

private:
    constexpr static uint16_t LIMIT_MNSIZE = 10;
    constexpr static uint16_t ALIGN_NUM = 16;
    constexpr static uint8_t UNIT_FLAG_SET = 3;
    constexpr static uint8_t UNIT_FLAG_CHECK = 2;
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<typename AType::T>();
    constexpr static auto L1_M = GetIntegralConstant<MNK_M, L1Shape>();
    constexpr static auto L1_N = GetIntegralConstant<MNK_N, L1Shape>();
    constexpr static auto L1_K = GetIntegralConstant<MNK_K, L1Shape>();
    constexpr static auto L0_M = GetIntegralConstant<MNK_M, L0Shape>();
    constexpr static auto L0_N = GetIntegralConstant<MNK_N, L0Shape>();
    constexpr static auto L0_K = GetIntegralConstant<MNK_K, L0Shape>();

    typename TileCopy::template CopyA1ToA2<AType, L0ATensorTrait, L1ATensorTrait> copyA1ToA2_;
    typename TileCopy::template CopyB1ToB2<AscendC::MatmulInputBType<BType, typename AType::T>, L0BTensorTrait,
                                           L1BTensorTrait>
        copyB1ToB2_;

    AscendC::TQueBind<AscendC::TPosition::GM, AscendC::TPosition::A1, 1,
                      AscendC::GetNdNzMask(CubeFormat::NZ, AType::format)>
        qidL1A_;
    AscendC::TQueBind<AscendC::TPosition::GM, AscendC::TPosition::B1, 1,
                      AscendC::GetNdNzMask(CubeFormat::NZ, BType::format)>
        qidL1B_;
    AscendC::TBuf<AscendC::TPosition::CO1> qidL0C_;
    Tile::TBufPoolL0 tbufPoolL0_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
