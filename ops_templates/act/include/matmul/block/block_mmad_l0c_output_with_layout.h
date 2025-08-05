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
 * \file block_mmad_l0c_output_with_layout.h
 * \brief
 */

#ifndef ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_L0C_OUTPUT_WITH_LAYOUT_H
#define ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_L0C_OUTPUT_WITH_LAYOUT_H

#include "lib/matmul/matmul.h"
#include "lib/matmul/tiling.h"
#include "lib/matmul/constant_tiling.h"
#include "./block_mmad.h"
#include "./block_mmad_utils.h"
#include "../../utils/layout_utils.h"
#include "../../utils/tensor_utils.h"
#include "../../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tile/tile_copy.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_,
          class TileCopy_>
class BlockMmad<MatmulL0COutputWithLayout<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_> {
public:
    using DispatchPolicy = MatmulL0COutputWithLayout<>;
    using L1Shape = L1TileShape_;
    using L0Shape = L0TileShape_;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using TileCopy = AscendC::Std::conditional_t<AscendC::Std::is_same_v<TileCopy_, void>,
                                                 Tile::TileCopy<Arch::Ascend910_95, Tile::CopyWithLayout>, TileCopy_>;

public:
    static_assert(AscendC::PhyPosIsL0C(CType::pos), "Only support L0C output");
    static_assert(IsF16F16F32<AType, BType, CType>() || IsBf16Bf16F32<AType, BType, CType>(), "Unsupported dtype");
    static_assert(IsND<AType>(), "Input A only support ND");
    static_assert(IsTileShapeValid<L1Shape, L0Shape>(), "L1Shape or L0Shape is invalid");
    static_assert(IsL1BufferValid<AType, BType, L1Shape>(), "L1 buffer overflow");
    static_assert(IsL0BufferValid<AType, BType, L0Shape>(), "L0 buffer overflow");

    __aicore__ BlockMmad() = default;
    __aicore__ ~BlockMmad()
    {
        matmul_.End();
    }

    constexpr static MatmulShapeParams shapeParams =
        GetMatmulShapeParams<typename DispatchPolicy::SingleShape, L0Shape>();
    constexpr static MatmulConfig cfg = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(
        shapeParams, GetFuncParams(DispatchPolicy::enableInputDataLenCheck), GetBiasParams(false));
    constexpr static MatmulApiStaticTiling staticTiling =
        AscendC::GetMatmulApiTiling<AType, BType, CType, BiasType, typename DispatchPolicy::SingleShape, L1Shape,
                                    L0Shape>(cfg);

public:
    /** \brief Init matmul object.
     */
    __aicore__ inline void Init()
    {
        matmul_.Init((TCubeTiling*)(nullptr), (AscendC::TPipe*)(nullptr));
    }

    /** \brief A complete matmul operation, will compute a matrix C of size actualShape.m * actualShape.n.
     * 
     * \param[out] c           Output matrix C based on Layout expression.
     * \param      a           Input matrix A based on Layout expression.
     * \param      b           Input matrix B based on Layout expression.
     * \param      actualShape The actual shape used in this calculation, real type is AscendC::Std::tuple(m, n, k).
     */
    template <class DstTensor, class SrcATensor, class SrcBTensor, class Shape>
    __aicore__ inline void IterateAll(DstTensor& c, const SrcATensor& a, const SrcBTensor& b, const Shape& actualShape)
    {
        using DstTrait = typename AscendC::tensor_trait<DstTensor>::trait_type;
        static_assert(AscendC::is_global_tensor_v<SrcATensor> && AscendC::is_global_tensor_v<SrcBTensor>,
                      "Only support GM input");
        static_assert(AscendC::is_local_tensor_v<DstTensor> && AscendC::PhyPosIsL0C(DstTrait::tPos),
                      "Output position only support l0c.");

        // Convert a tensor with layout to a normal tensor
        AscendC::GlobalTensor<typename AType::T> aGlobal;
        aGlobal.address_ = a.address_;
        AscendC::GlobalTensor<typename BType::T> bGlobal;
        bGlobal.address_ = b.address_;
        AscendC::LocalTensor<typename CType::T> cTmp;
        cTmp.SetAddr(c.address_);

        matmul_.SetTensorA(aGlobal, AType::isTrans);
        matmul_.SetTensorB(bGlobal, BType::isTrans);

        SetOrgShape(c, a, b);

        matmul_.SetSingleShape(Get<MNK_M>(actualShape), Get<MNK_N>(actualShape), Get<MNK_K>(actualShape));
        matmul_.IterateAll(cTmp);

        c.SetAddr(cTmp.address_);
    }

    template <class DstTensor, class SrcTensor, class Coord>
    __aicore__ inline void GetTensorC(DstTensor& ub, SrcTensor& l0c, const Coord& coord, uint8_t subIdx)
    {
        using DstTrait = typename AscendC::tensor_trait<DstTensor>::trait_type;
        using SrcTrait = typename AscendC::tensor_trait<SrcTensor>::trait_type;
        using DstType =
            AscendC::MatmulType<DstTrait::tPos,
                                LayoutToFormatV<typename DstTrait::LiteType, typename DstTrait::LiteLayoutType>,
                                typename DstTrait::LiteType>;
        static_assert(AscendC::is_local_tensor_v<DstTensor> && AscendC::is_local_tensor_v<SrcTensor>,
                      "Only support local tensor.");
        static_assert(AscendC::PhyPosIsL0C(SrcTrait::tPos) && AscendC::PhyPosIsUB(DstTrait::tPos),
                      "Only support l0c to ub.");
        static_assert(DstType::format == CubeFormat::ND || DstType::format == CubeFormat::NZ,
                      "Dst format only support ND or Nz.");

        typename TileCopy::template CopyCo1ToOut<DstType, DstTrait, SrcTrait> copyCo1ToOut;
        copyCo1ToOut(ub, l0c, coord, subIdx);
    }

private:
    __aicore__ inline void End()
    {
        matmul_.End();
    }

    template <class DstTensor, class SrcATensor, class SrcBTensor>
    __aicore__ inline void SetOrgShape(DstTensor& c, const SrcATensor& a, const SrcBTensor& b)
    {
        constexpr int mIdx = AType::isTrans ? 1 : 0;
        constexpr int nIdx = BType::isTrans ? 0 : 1;
        constexpr int kaIdx = AType::isTrans ? 0 : 1;
        constexpr int kbIdx = BType::isTrans ? 1 : 0;

        int orgM;
        int orgKa;
        const auto& aShape = a.GetTensorTrait().GetLayout().GetShape();
        if constexpr (AType::format == CubeFormat::ND) {
            orgM = Get<mIdx>(aShape);
            orgKa = Get<kaIdx>(aShape);
        } else {
            orgM = Get<mIdx, 0>(aShape) * Get<mIdx, 1>(aShape);
            orgKa = Get<kaIdx, 0>(aShape) * Get<kaIdx, 1>(aShape);
        }

        int orgN;
        int orgKb;
        const auto& bShape = b.GetTensorTrait().GetLayout().GetShape();
        if constexpr (BType::format == CubeFormat::ND) {
            orgN = Get<nIdx>(bShape);
            orgKb = Get<kbIdx>(bShape);
        } else {
            orgN = Get<nIdx, 0>(bShape) * Get<nIdx, 1>(bShape);
            orgKb = Get<kbIdx, 0>(bShape) * Get<kbIdx, 1>(bShape);
        }

        const auto& cShape = c.GetTensorTrait().GetLayout().GetShape();
        auto orgKc = Get<1, 0>(cShape) * Get<1, 1>(cShape); // Set if matrix C's N != matrix B's N
        matmul_.SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc);
    }

    AscendC::MatmulImpl<AType, BType, CType, BiasType, staticTiling> matmul_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
