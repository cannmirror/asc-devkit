/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sparse_block_mmad_multi_block_on_kaxis_with_layout.h
 * \brief
 */
#ifndef MATMUL_BLOCK_SPARSE_BLOCK_MMAD_MULTI_BLOCK_ON_KAXIS_WITH_LAYOUT_H
#define MATMUL_BLOCK_SPARSE_BLOCK_MMAD_MULTI_BLOCK_ON_KAXIS_WITH_LAYOUT_H

#include "./block_mmad.h"
#include "./block_mmad_utils.h"
#include "../../utils/layout_utils.h"
#include "../../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tile/load_data/tbuf_pool_l0_default.h"
#include "../tile/tile_copy.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class L1TileShape, class L0TileShape, class AT, class BT, class CT, class BiasT, class TileCopy>
class BlockMmad<SparseMatmulMultiBlockOnKAxisWithLayout<>, L1TileShape, L0TileShape, AT, BT, CT, BiasT, TileCopy,
    AscendC::Std::enable_if_t<IsMatmulLayoutTypeV<AT>>>
    : public BlockMmad<SparseMatmulMultiBlockOnKAxisWithLayout<>, L1TileShape, L0TileShape,
        ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy> {
    using Base = BlockMmad<SparseMatmulMultiBlockOnKAxisWithLayout<>, L1TileShape, L0TileShape,
                           ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy>;
    using Base::Base;
};

/**
* @class BlockMmad
* @brief A class for handling sparse matrix multiplication with multi-block layout on the K-axis
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
class BlockMmad<SparseMatmulMultiBlockOnKAxisWithLayout<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_,
                BiasType_, TileCopy_, AscendC::Std::enable_if_t<!IsMatmulLayoutTypeV<AType_>>> {
public:
    using DispatchPolicy = SparseMatmulMultiBlockOnKAxisWithLayout<>;
    using L1Shape = L1TileShape_;
    using L0Shape = L0TileShape_;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using SparseIndexType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, uint8_t>;

    using TileCopy =
        AscendC::Std::conditional_t<AscendC::Std::is_same_v<TileCopy_, void>,
                                    Tile::TileCopy<Arch::Ascend910B, Tile::CopySparseWithLayout>, TileCopy_>;

    using L1ALayout = FormatToLayoutT<typename AType::T, CubeFormat::NZ>;
    using L1BLayout = FormatToLayoutT<typename BType::T, CubeFormat::NZ>;
    using L1IdxLayout = FormatToLayoutT<int32_t, CubeFormat::NZ>; // sparse index fractal 16*8
    using L0ALayout = FormatToLayoutT<typename AType::T, CubeFormat::ZZ>;
    using L0BLayout = FormatToLayoutT<typename BType::T, CubeFormat::ZN>;
    using L0IdxLayout = FormatToLayoutT<typename SparseIndexType::T, CubeFormat::ZN>;
    using L0CLayout = AscendC::Layout<AscendC::Shape<AscendC::Shape<_16, int>, // L0C fractal size: fixed 16x16
                                                     AscendC::Shape<_16, int>>,
                                      AscendC::Stride<AscendC::Stride<_16, _256>, AscendC::Stride<_1, int>>>;

    using L1ATensorTrait = AscendC::TensorTrait<typename AType::T, AscendC::TPosition::A1, L1ALayout>;
    using L1BTensorTrait = AscendC::TensorTrait<typename BType::T, AscendC::TPosition::B1, L1BLayout>;
    using L1IdxTensorTrait = AscendC::TensorTrait<typename SparseIndexType::T, AscendC::TPosition::B1, L1IdxLayout>;
    using L0ATensorTrait = AscendC::TensorTrait<typename AType::T, AscendC::TPosition::A2, L0ALayout>;
    using L0BTensorTrait = AscendC::TensorTrait<typename BType::T, AscendC::TPosition::B2, L0BLayout>;
    using L0IdxTensorTrait = AscendC::TensorTrait<typename SparseIndexType::T, AscendC::TPosition::B1, L0IdxLayout>;
    using L0CTensorTrait = AscendC::TensorTrait<typename AscendC::GetMmDstType<typename AType::T>::Type,
                                                AscendC::TPosition::CO1, L0CLayout>;

public:
    static_assert(AscendC::Std::is_same_v<TileCopy, Tile::TileCopy<Arch::Ascend910B, Tile::CopySparseWithLayout>>,
                  "Only support CopySparseWithLayout");
    static_assert(IsI8I8I32<AType, BType, CType>(), "Unsupported dtype");
    static_assert(BType::isTrans, "Only support MatrixB transpose");
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
        tpipe->InitBuffer(qidL1B_, dbL1Flag, bL1MatrixByteSize / DENSE_MATRIX_B_OFFSET);
        tpipe->InitBuffer(qidL1BIdx_, dbL1Flag, bL1MatrixByteSize / INDEX_MATRIX_OFFSET);
        tpipe->InitBuffer(qidL0C_, 1, cMatrixByteSize);

        InitShareBufEnd(tpipe);
    }

    /**
    * @brief Perform the matrix multiplication operations
    * @param [in] DstTensor: destination tensor type
    * @param [in] SrcATensor: matrix A source tensor type
    * @param [in] SrcBTensor: matrix B source tensor type
    * @param [in] SparseIdxTensor: the type of the sparse index tensor
    * @param [in] Shape: shape type
    * @param [in] c: destination tensor
    * @param [in] a: matrix A source tensor
    * @param [in] b: matrix B source tensor
    * @param [in] sparseIdx: the sparse index tensor
    * @param [in] actualShape: actual shape
    */
    template <class DstTensor, class SrcATensor, class SrcBTensor, class SparseIdxTensor, class Shape>
    __aicore__ inline void IterateAll(DstTensor& c, const SrcATensor& a, const SrcBTensor& b,
                                      const SparseIdxTensor& sparseIdx, const Shape& actualShape)
    {
        static_assert(AscendC::is_global_tensor_v<DstTensor> && AscendC::is_global_tensor_v<SrcATensor> &&
                          AscendC::is_global_tensor_v<SrcBTensor> && AscendC::is_global_tensor_v<SparseIdxTensor>,
                      "Only support GM in and GM out");
        using DstTrait = typename AscendC::tensor_trait<DstTensor>::trait_type;
        typename TileCopy::template CopyCo1ToOut<CType, DstTrait, L0CTensorTrait> copyCo1ToGm;

        int32_t mIter = AscendC::Ceil(Get<MNK_M>(actualShape), L0_M);
        int32_t nIter = AscendC::Ceil(Get<MNK_N>(actualShape), L0_N);
        int32_t kOuterIter = AscendC::Ceil(Get<MNK_K>(actualShape), L1_K);
        int32_t tailBaseM = (Get<MNK_M>(actualShape) % L0_M) == 0 ? L0_M : (Get<MNK_M>(actualShape) % L0_M);
        int32_t tailBaseN = (Get<MNK_N>(actualShape) % L0_N) == 0 ? L0_N : (Get<MNK_N>(actualShape) % L0_N);
        int32_t tailBaseK = (Get<MNK_K>(actualShape) % L0_K) == 0 ? L0_K : (Get<MNK_K>(actualShape) % L0_K);
        int32_t tailL1K = (Get<MNK_K>(actualShape) % L1_K) == 0 ? L1_K : (Get<MNK_K>(actualShape) % L1_K);

        for (auto mIndex = 0; mIndex < mIter; ++mIndex) {
            for (auto nIndex = 0; nIndex < nIter; ++nIndex) {
                int baseM = (mIndex + 1 == mIter) ? tailBaseM : L0_M;
                int baseN = (nIndex + 1 == nIter) ? tailBaseN : L0_N;
                auto l0CLayout = AscendC::MakeLayout(
                    AscendC::MakeShape(AscendC::MakeShape(_16{}, AscendC::Ceil(baseM, C0_NUM_PER_FRACTAL)),
                                       AscendC::MakeShape(_16{}, AscendC::Ceil(baseN, C0_SIZE_L0C))),
                    AscendC::MakeStride(AscendC::MakeStride(_16{}, _256{}),
                                        AscendC::MakeStride(_1{}, AscendC::CeilAlign(baseM, C0_NUM_PER_FRACTAL) * C0_SIZE_L0C)));
                auto l0C = qidL0C_.template AllocTensor<L0CTensorTrait>();
                l0C.SetTensorTrait(L0CTensorTrait(l0CLayout));

                for (auto kOuterIndex = 0; kOuterIndex < kOuterIter; ++kOuterIndex) {
                    int l1K = (kOuterIndex + 1 == kOuterIter) ? tailL1K : L1_K;
                    // -----------------Step1: GM -> L1 -----------------
                    CopyInA(mIndex, baseM, l1K, kOuterIndex, a);
                    qidL1A_.EnQue(l1A_);
                    CopyInB(nIndex, baseN, l1K, kOuterIndex, b, sparseIdx);

                    qidL1B_.EnQue(l1B_);
                    qidL1BIdx_.EnQue(l1BIdx_);
                    qidL1A_.DeQue();
                    qidL1B_.DeQue();
                    qidL1BIdx_.DeQue();

                    int32_t kInnerIter =
                        (kOuterIndex + 1 == kOuterIter) ? AscendC::Ceil(tailL1K, L0_K) : AscendC::Ceil(L1_K, L0_K);
                    for (auto kInnerIndex = 0; kInnerIndex < kInnerIter; ++kInnerIndex) {
                        int baseK =
                            (kOuterIndex + 1 == kOuterIter) && (kInnerIndex + 1 == kInnerIter) ? tailBaseK : L0_K;

                        // -----------------Step2: L1 -> L0-----------------
                        auto& bufferPool = tbufPoolL0_.Allocate();
                        LoadDataA(mIndex, baseM, baseK, kInnerIndex, bufferPool);
                        LoadDataB(nIndex, baseN, baseK, kInnerIndex, bufferPool);

                        bufferPool.EnQue();
                        bufferPool.DeQue();

                        //  -----------------Step3: compute-----------------
                        AscendC::LocalTensor<typename L0ATensorTrait::LiteType> l0ALocal;
                        AscendC::LocalTensor<typename L0BTensorTrait::LiteType> l0BLocal;
                        l0ALocal.SetAddr(l0A_.address_);
                        l0BLocal.SetAddr(l0B_.address_);

                        AscendC::MmadParams mmadParams;
                        // GEMV is automatically enabled when setting M=1 in normal mode
                        mmadParams.m = (baseM == 1 ? ALIGN_NUM : baseM);
                        mmadParams.k = baseK;
                        mmadParams.n = baseN;
                        mmadParams.unitFlag = 0;
                        mmadParams.cmatrixInitVal = (kOuterIndex == 0) && (kInnerIndex == 0) ? true : false;
                        MmadWithSparse(l0C, l0ALocal, l0BLocal, mmadParams);
                        if ((baseM / ALIGN_NUM) * (baseN / ALIGN_NUM) < LIMIT_MNSIZE) {
                            AscendC::PipeBarrier<PIPE_M>();
                        }
                        bufferPool.Free();
                    }
                    qidL1A_.FreeTensor(l1A_);
                    qidL1B_.FreeTensor(l1B_);
                    qidL1BIdx_.FreeTensor(l1BIdx_);

                    qidL0C_.EnQue(l0C);
                    qidL0C_.template DeQue<L0CTensorTrait>();
                }
                // -----------------Step4: L0C -> GM-----------------
                copyCo1ToGm(c, l0C, AscendC::MakeCoord(mIndex * L0_M, nIndex * L0_N));
                qidL0C_.FreeTensor(l0C);
            }
        }
    }
    
    /**
     * @brief Copy data from SrcATensor to L1 cache
     * @param [in] mIndex: row index of matrix A
     * @param [in] baseM: base row count of matrix A
     * @param [in] l1K: number of rows in L1 cache
     * @param [in] kOuterIndex: outer k loop index
     * @param [in] a: input SrcATensor data
     * @return void
     */
    template <class SrcATensor>
    __aicore__ inline void CopyInA(int32_t mIndex, int baseM, int l1K, int32_t kOuterIndex, const SrcATensor& a)
    {
        auto aTileHeight = AType::isTrans ? l1K : baseM;
        auto aTileWidth = AType::isTrans ? baseM : l1K;
        auto aL1Row = AType::isTrans ? kOuterIndex : mIndex;
        auto aL1Col = AType::isTrans ? mIndex : kOuterIndex;
        using SrcATrait = typename AscendC::tensor_trait<SrcATensor>::trait_type;
        typename TileCopy::template CopyGmToA1<AType, L1ATensorTrait, SrcATrait> copyGmToA1;

        auto l1ALayout = MakeLayoutByFormat<typename AType::T, CubeFormat::NZ>(aTileHeight, aTileWidth);
        l1A_ = qidL1A_.template AllocTensor<L1ATensorTrait>();
        l1A_.SetTensorTrait(L1ATensorTrait(l1ALayout));

        copyGmToA1(l1A_, const_cast<AscendC::GlobalTensor<SrcATrait>&>(a),
                   AscendC::MakeCoord(aL1Row * (AType::isTrans ? L1_K : L0_M),
                                      aL1Col * (AType::isTrans ? L0_M : L1_K)));  
    }

    /**
     * @brief Copy data from SrcBTensor and sparse index tensor to L1 cache
     * @param [in] mIndex: row index of matrix B
     * @param [in] baseM: base row count of matrix B
     * @param [in] l1K: number of rows in L1 cache
     * @param [in] kOuterIndex: outer k loop index
     * @param [in] b: input SrcBTensor data
     * @param [in] sparseIdx: input sparse index tensor
     * @return void
     */
    template <class SrcBTensor, class SparseIdxTensor>
    __aicore__ inline void CopyInB(int32_t nIndex, int baseN, int l1K, int32_t kOuterIndex,
                                        const SrcBTensor& b, const SparseIdxTensor& sparseIdx)
    {
        auto bTileHeight = baseN;
        auto bTileWidth = l1K;
        auto bL1Row = nIndex;
        auto bL1Col = kOuterIndex;
        using SrcBTrait = typename AscendC::tensor_trait<SrcBTensor>::trait_type;
        using SparseIdxTrait = typename AscendC::tensor_trait<SparseIdxTensor>::trait_type;
        typename TileCopy::template CopyGmToB1<BType, L1BTensorTrait, SrcBTrait> copyGmToB1;
        typename TileCopy::template CopyGmToB1<SparseIndexType, L1IdxTensorTrait, SparseIdxTrait> copyGmToB1Idx;

        auto l1BLayout = MakeLayoutByFormat<typename BType::T, CubeFormat::NZ>(
             bTileHeight, bTileWidth / DENSE_MATRIX_B_OFFSET);
        l1B_ = qidL1B_.template AllocTensor<L1BTensorTrait>();
        l1B_.SetTensorTrait(L1BTensorTrait(l1BLayout));

        auto l1BIndexLayout =
             MakeLayoutByFormat<int32_t, CubeFormat::NZ>(bTileHeight, bTileWidth / INDEX_MATRIX_OFFSET);
        l1BIdx_ = qidL1BIdx_.template AllocTensor<L1IdxTensorTrait>();
        l1BIdx_.SetTensorTrait(L1IdxTensorTrait(l1BIndexLayout));

        copyGmToB1(l1B_, const_cast<AscendC::GlobalTensor<SrcBTrait>&>(b),
                   AscendC::MakeCoord(bL1Row * L0_N, bL1Col * L1_K / DENSE_MATRIX_B_OFFSET));

        copyGmToB1Idx(l1BIdx_, const_cast<AscendC::GlobalTensor<SparseIdxTrait>&>(sparseIdx),
                      AscendC::MakeCoord(bL1Row * L0_N, bL1Col * L1_K / INDEX_MATRIX_OFFSET));
    }

    /**
     * @brief Load data from L1 cache to L0 cache for matrix A
     * @param [in] mIndex: row index of matrix A
     * @param [in] baseM: base row count of matrix A
     * @param [in] baseK: base column count of matrix A
     * @param [in] kInnerIndex: inner k loop index
     * @param [in] bufferPool: L0 cache buffer pool
     * @return void
     */
    __aicore__ inline void LoadDataA(int32_t mIndex, int baseM, int baseK, int32_t kInnerIndex, Tile::TBufPoolL0 bufferPool)
    {
        auto aBaseHeight = AType::isTrans ? baseK : baseM;
        auto aBaseWidth = AType::isTrans ? baseM : baseK;
        auto aL0Row = AType::isTrans ? kInnerIndex : mIndex;
        auto aL0Col = AType::isTrans ? mIndex : kInnerIndex;
        auto l0ALayout = MakeLayoutByFormat<typename AType::T, CubeFormat::ZZ>(aBaseHeight, aBaseWidth);
        l0A_ = bufferPool.template GetBuffer<AscendC::TPosition::A2, L0ATensorTrait>();
        l0A_.SetTensorTrait(L0ATensorTrait(l0ALayout));
        copyA1ToA2_(l0A_, l1A_,
                    AscendC::MakeCoord(aL0Row * (AType::isTrans ? L0_K : 0),
                                       aL0Col * (AType::isTrans ? 0 : L0_K)));
    }

    /**
     * @brief Load data from L1 cache to L0 cache for matrix B
     * @param [in] nIndex: row index of matrix B
     * @param [in] baseN: base row count of matrix B
     * @param [in] baseK: base column count of matrix B
     * @param [in] kInnerIndex: inner k loop index
     * @param [in] bufferPool: L0 cache buffer pool
     * @return void
     */
    __aicore__ inline void LoadDataB(int32_t nIndex, int baseN, int baseK, int32_t kInnerIndex, Tile::TBufPoolL0 bufferPool)
    {
        auto bBaseHeight = baseN;
        auto bBaseWidth = baseK;
        auto bL0Row = nIndex;
        auto bL0Col = kInnerIndex;
        auto l0BLayout = MakeLayoutByFormat<typename BType::T, CubeFormat::ZN>(
                         bBaseHeight, bBaseWidth / DENSE_MATRIX_B_OFFSET);
        l0B_ = bufferPool.template GetBuffer<AscendC::TPosition::B2, L0BTensorTrait>();
        l0B_.SetTensorTrait(L0BTensorTrait(l0BLayout));
        copyB1ToB2_(l0B_, l1B_, AscendC::MakeCoord(0, bL0Col * L0_K / DENSE_MATRIX_B_OFFSET), l1BIdx_);
    }

private:
    /**
     * @brief Calculate the used size of L1 cache
     * @param [in] dbL1Flag: flag indicating whether L1 ping pong double buffer is enables
     * @return The size of L1 cache that has been used
     */
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
            sharedl1Size += dbL1Flag * L0_N * (L1_K / DENSE_MATRIX_B_OFFSET + L1_K / INDEX_MATRIX_OFFSET) *
                            AscendC::GetBitSize<typename BType::T>() / AscendC::ONE_BYTE_BIT_SIZE;
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
        qidL1BIdx_.FreeAllEvent();
        qidL0C_.FreeAllEvent();
        event_t eventIDFixToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_M));
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventIDFixToM);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventIDFixToM);
    }

private:
    constexpr static uint16_t LIMIT_MNSIZE = 10;
    constexpr static uint16_t ALIGN_NUM = 16;
    constexpr static int32_t DENSE_MATRIX_B_OFFSET = 2;
    constexpr static int32_t INDEX_MATRIX_OFFSET = 8;
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
    AscendC::TQueBind<AscendC::TPosition::GM, AscendC::TPosition::B1, 1,
                      AscendC::GetNdNzMask(CubeFormat::NZ, SparseIndexType::format)>
        qidL1BIdx_;
    AscendC::TQue<AscendC::TPosition::CO1, 1> qidL0C_;
    Tile::TBufPoolL0 tbufPoolL0_;
    AscendC::LocalTensor<L1ATensorTrait> l1A_;
    AscendC::LocalTensor<L1BTensorTrait> l1B_;
    AscendC::LocalTensor<L0ATensorTrait> l0A_;
    AscendC::LocalTensor<L0BTensorTrait> l0B_;
    AscendC::LocalTensor<L1IdxTensorTrait> l1BIdx_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
