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
 * \file block_mmad_naive_pipeline_with_layout.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_MMAD_NAIVE_PIPELINE_WITH_LAYOUT_H
#define MATMUL_BLOCK_BLOCK_MMAD_NAIVE_PIPELINE_WITH_LAYOUT_H

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
template <class L1TileShape, class L0TileShape, class AT, class BT, class CT, class BiasT, class TileCopy>
class BlockMmad<MatmulNaivePipelineWithLayout<>, L1TileShape, L0TileShape, AT, BT, CT, BiasT, TileCopy,
    AscendC::Std::enable_if_t<IsMatmulLayoutTypeV<AT>>>
    : public BlockMmad<MatmulNaivePipelineWithLayout<>, L1TileShape, L0TileShape,
        ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy> {
    using Base = BlockMmad<MatmulNaivePipelineWithLayout<>, L1TileShape, L0TileShape,
                           ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy>;
    using Base::Base;
};

/**
* @class BlockMmad
* @brief Implementation class for matrix multiplication operations, supporting matrix multiplication with specific layouts
* @param [in] MatmulNaivePipelineWithLayout<>: policy type
* @param [in] L1TileShape_: the shape of the L1 tile
* @param [in] L0TileShape_: the shape of the L0 tile
* @param [in] AType_: the data type of matrix A
* @param [in] BType_: the data type of matrix B
* @param [in] CType_: the data type of matrix C
* @param [in] BiasType_: the data type of the bias
* @param [in] TileCopy_: tile copy strategy
*/
template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_,
          class TileCopy_>
class BlockMmad<MatmulNaivePipelineWithLayout<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_,
                TileCopy_, AscendC::Std::enable_if_t<!IsMatmulLayoutTypeV<AType_>>> {
public:
    using DispatchPolicy = MatmulNaivePipelineWithLayout<>;
    using L1Shape = L1TileShape_;
    using L0Shape = L0TileShape_;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;

    constexpr static bool useL0CUnitFlag = (DispatchPolicy::l0CStages == 1) && DispatchPolicy::enalbeL0CUnitFlag;
    using CopyType = AscendC::Std::conditional_t<
        useL0CUnitFlag,
        Tile::CopyEnUnitFlagWithLayout,
        Tile::CopyWithLayout
    >;
    using TileCopy = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<TileCopy_, void>,
        Tile::TileCopy<Arch::Ascend910B, CopyType>,
        TileCopy_
    >;

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
    static_assert(IsF16OrBf16AB<AType, BType, CType>() || IsF32F32F32<AType, BType, CType>(), "Unsupported dtype");
    static_assert(IsNDOrAlign<AType>() && IsNDOrAlign<BType>() && IsNDOrAlign<CType>(), "Only support ND format");
    static_assert(IsTileShapeValid<AType, BType, L1Shape, L0Shape>(), "L1Shape or L0Shape is invalid");

    __aicore__ ~BlockMmad()
    {
        End();
    }

    /**
    * @brief Initialization function
    * 
    * This function initializes the buffers and parameters required for matrix multiplication operations
    */
    __aicore__ inline void Init()
    {
        constexpr static int32_t aMatrixByteSize =
            L0_M * L0_K * AscendC::GetBitSize<typename AType::T>() / AscendC::ONE_BYTE_BIT_SIZE;
        constexpr static int32_t bMatrixByteSize =
            L0_N * L0_K * AscendC::GetBitSize<typename BType::T>() / AscendC::ONE_BYTE_BIT_SIZE;
        constexpr static int32_t cMatrixByteSize =
            L0_M * L0_N * sizeof(typename AscendC::GetMmDstType<typename AType::T>::Type);

        constexpr static int32_t dbL0AFlag = (aMatrixByteSize * DOUBLE_BUFFER_COUNT > L0A_SIZE) ? 1 : DOUBLE_BUFFER_COUNT;
        constexpr static int32_t dbL0BFlag = (bMatrixByteSize * DOUBLE_BUFFER_COUNT > L0B_SIZE) ? 1 : DOUBLE_BUFFER_COUNT;

        tbufPoolL0_.Init((dbL0AFlag - 1) & (dbL0BFlag - 1));
        auto tpipe = GetTPipePtr();
        uint32_t shareLens[3] = {static_cast<uint32_t>(GetL1UsedSize()), cMatrixByteSize, 0};
        InitShareBufStart(tpipe, 0, shareLens, 3, 0); // 3: shareLens num

        tpipe->InitBuffer(qidL1A_, 1, aMatrixByteSize);
        tpipe->InitBuffer(qidL1B_, 1, bMatrixByteSize);
        tpipe->InitBuffer(qidL0C_, DispatchPolicy::l0CStages, cMatrixByteSize);

        InitShareBufEnd(tpipe);
    }

    /**
    * @brief Perform the matrix multiplication operations
    * @param [out] DstTensor: destination tensor type
    * @param [in] SrcATensor: matrix A source tensor type
    * @param [in] SrcBTensor: matrix B source tensor type
    * @param [in] Shape: shape type
    * @param [out] c: destination tensor
    * @param [in] a: matrix A source tensor
    * @param [in] b: matrix B source tensor
    * @param [in] actualShape: actual shape
    */
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
        int32_t kIter = AscendC::Ceil(Get<MNK_K>(actualShape), L0_K);
        int32_t tailBaseM = (Get<MNK_M>(actualShape) % L0_M) == 0 ? L0_M : (Get<MNK_M>(actualShape) % L0_M);
        int32_t tailBaseN = (Get<MNK_N>(actualShape) % L0_N) == 0 ? L0_N : (Get<MNK_N>(actualShape) % L0_N);
        int32_t tailBaseK = (Get<MNK_K>(actualShape) % L0_K) == 0 ? L0_K : (Get<MNK_K>(actualShape) % L0_K);

        for (auto mIndex = 0; mIndex < mIter; ++mIndex) {
            for (auto nIndex = 0; nIndex < nIter; ++nIndex) {
                int32_t baseM = (mIndex + 1 == mIter) ? tailBaseM : L0_M;
                int32_t baseN = (nIndex + 1 == nIter) ? tailBaseN : L0_N;

                auto l0CLayout = AscendC::MakeLayout(
                    AscendC::MakeShape(AscendC::MakeShape(_16{}, AscendC::Ceil(baseM, C0_NUM_PER_FRACTAL)),
                                       AscendC::MakeShape(_16{}, AscendC::Ceil(baseN, C0_SIZE_L0C))),
                    AscendC::MakeStride(AscendC::MakeStride(_16{}, _256{}),
                                        AscendC::MakeStride(_1{}, AscendC::CeilAlign(baseM, C0_NUM_PER_FRACTAL) * C0_SIZE_L0C)));
                auto l0C = qidL0C_.template AllocTensor<L0CTensorTrait>();
                l0C.SetTensorTrait(L0CTensorTrait(l0CLayout));

                for (auto kIndex = 0; kIndex < kIter; ++kIndex) {
                    int32_t baseK = (kIndex + 1 == kIter) ? tailBaseK : L0_K;
                    // -----------------Step1: GM -> L1 -----------------
                    auto aTileHeight = AType::isTrans ? baseK : baseM;
                    auto aTileWidth = AType::isTrans ? baseM : baseK;
                    auto aRow = AType::isTrans ? kIndex : mIndex;
                    auto aCol = AType::isTrans ? mIndex : kIndex;

                    auto l1ALayout = MakeLayoutByFormat<typename AType::T, CubeFormat::NZ>(aTileHeight, aTileWidth);
                    auto l1A = qidL1A_.template AllocTensor<L1ATensorTrait>();
                    l1A.SetTensorTrait(L1ATensorTrait(l1ALayout));

                    copyGmToA1(l1A, const_cast<AscendC::GlobalTensor<SrcATrait>&>(a),
                               AscendC::MakeCoord(aRow * (AType::isTrans ? L0_K : L0_M),
                                                  aCol * (AType::isTrans ? L0_M : L0_K)));

                    qidL1A_.EnQue(l1A);
                    qidL1A_.DeQue();

                    auto bTileHeight = BType::isTrans ? baseN : baseK;
                    auto bTileWidth = BType::isTrans ? baseK : baseN;
                    auto bRow = BType::isTrans ? nIndex : kIndex;
                    auto bCol = BType::isTrans ? kIndex : nIndex;

                    auto l1BLayout = MakeLayoutByFormat<typename BType::T, CubeFormat::NZ>(bTileHeight, bTileWidth);
                    auto l1B = qidL1B_.template AllocTensor<L1BTensorTrait>();
                    l1B.SetTensorTrait(L1BTensorTrait(l1BLayout));

                    copyGmToB1(l1B, const_cast<AscendC::GlobalTensor<SrcBTrait>&>(b),
                               AscendC::MakeCoord(bRow * (BType::isTrans ? L0_N : L0_K),
                                                  bCol * (BType::isTrans ? L0_K : L0_N)));

                    qidL1B_.EnQue(l1B);
                    qidL1B_.DeQue();

                    // -----------------Step2: L1 -> L0-----------------
                    auto& bufferPool = tbufPoolL0_.Allocate();

                    auto l0ALayout = MakeLayoutByFormat<typename AType::T, CubeFormat::ZZ>(aTileHeight, aTileWidth);
                    auto l0A = bufferPool.template GetBuffer<AscendC::TPosition::A2, L0ATensorTrait>();
                    l0A.SetTensorTrait(L0ATensorTrait(l0ALayout));
                    copyA1ToA2_(l0A, l1A, AscendC::MakeCoord(0, 0));

                    auto l0BLayout = MakeLayoutByFormat<typename BType::T, CubeFormat::ZN>(bTileHeight, bTileWidth);
                    auto l0B = bufferPool.template GetBuffer<AscendC::TPosition::B2, L0BTensorTrait>();
                    l0B.SetTensorTrait(L0BTensorTrait(l0BLayout));
                    copyB1ToB2_(l0B, l1B, AscendC::MakeCoord(0, 0));

                    bufferPool.EnQue();
                    bufferPool.DeQue();

                    //  -----------------Step3: compute-----------------
                    AscendC::MmadParams mmadParams;
                    // GEMV is automatically enabled when setting M=1 in normal mode
                    mmadParams.m = (baseM == 1 ? ALIGN_NUM : baseM);
                    mmadParams.k = baseK;
                    mmadParams.n = baseN;
                    mmadParams.cmatrixInitVal = (kIndex == 0) ? true : false;
                    Mmad(l0C, l0A, l0B, mmadParams);

                    // add pipe_M required by aicore
                    if ((mmadParams.m / ALIGN_NUM) * (mmadParams.n / ALIGN_NUM) < LIMIT_MNSIZE) {
                        AscendC::PipeBarrier<PIPE_M>();
                    }

                    bufferPool.Free();
                    qidL1A_.FreeTensor(l1A);
                    qidL1B_.FreeTensor(l1B);

                    // -----------------Step4: L0C -> GM-----------------
                    qidL0C_.EnQue(l0C);
                    qidL0C_.template DeQue<L0CTensorTrait>();
                }
                copyCo1ToGm(c, l0C, AscendC::MakeCoord(mIndex * L0_M, nIndex * L0_N));
                qidL0C_.FreeTensor(l0C);
            }
        }
    }

private:
    /**
    * @brief Get the size of L1 cache that has been used
    * @return Return the size of L1 cache that has been used
    */
    __aicore__ inline constexpr int32_t GetL1UsedSize()
    {
        int32_t sharedl1Size = 0;
        if constexpr (!PhyPosIsL1(AType::pos)) {
            // L1 DB ON
            sharedl1Size += DOUBLE_BUFFER_COUNT * L1_M * L1_K * AscendC::GetBitSize<typename AType::T>() /
                            AscendC::ONE_BYTE_BIT_SIZE;
        }
        if constexpr (!PhyPosIsL1(BType_::pos)) {
            // L1 DB ON
            sharedl1Size += DOUBLE_BUFFER_COUNT * L1_N * L1_K * AscendC::GetBitSize<typename BType::T>() /
                            AscendC::ONE_BYTE_BIT_SIZE;
        }
        return sharedl1Size;
    }

    /**
     * @brief End function, release all events
     */
    __aicore__ inline void End()
    {
        qidL1A_.FreeAllEvent();
        qidL1B_.FreeAllEvent();
        qidL0C_.FreeAllEvent();
    }

private:
    constexpr static auto L1_M = GetIntegralConstant<MNK_M, L1Shape>();
    constexpr static auto L1_N = GetIntegralConstant<MNK_N, L1Shape>();
    constexpr static auto L1_K = GetIntegralConstant<MNK_K, L1Shape>();
    constexpr static auto L0_M = GetIntegralConstant<MNK_M, L0Shape>();
    constexpr static auto L0_N = GetIntegralConstant<MNK_N, L0Shape>();
    constexpr static auto L0_K = GetIntegralConstant<MNK_K, L0Shape>();
    constexpr static uint16_t ALIGN_NUM = 16;
    constexpr static uint16_t LIMIT_MNSIZE = 10;
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<typename AType::T>();

    typename TileCopy::template CopyA1ToA2<AType, L0ATensorTrait, L1ATensorTrait> copyA1ToA2_;
    typename TileCopy::template CopyB1ToB2<AscendC::MatmulInputBType<BType, typename AType::T>, L0BTensorTrait,
                                           L1BTensorTrait> copyB1ToB2_;

    AscendC::TQueBind<AscendC::TPosition::GM, AscendC::TPosition::A1, 1,
                      AscendC::GetNdNzMask(CubeFormat::NZ, AType::format)> qidL1A_;
    AscendC::TQueBind<AscendC::TPosition::GM, AscendC::TPosition::B1, 1,
                      AscendC::GetNdNzMask(CubeFormat::NZ, BType::format)> qidL1B_;
    AscendC::TQue<AscendC::TPosition::CO1, 1> qidL0C_;
    Tile::TBufPoolL0 tbufPoolL0_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
