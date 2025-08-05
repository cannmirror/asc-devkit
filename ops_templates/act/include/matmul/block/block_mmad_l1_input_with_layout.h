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
 * \file block_mmad_l1_input_with_layout.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_L1_INPUT_WITH_LAYOUT_H
#define ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_L1_INPUT_WITH_LAYOUT_H

#include "lib/matmul/matmul.h"
#include "lib/matmul/tiling.h"
#include "lib/matmul/constant_tiling.h"

#include "./block_mmad.h"
#include "./block_mmad_utils.h"
#include "../../utils/tensor_utils.h"
#include "../../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tile/tile_copy.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_>
class BlockMmad<MatmulL1InputWithLayout<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_,
                Tile::TileCopy<Arch::Ascend910_95, Tile::CopyNoGmIn>> {
public:
    using DispatchPolicy = MatmulL1InputWithLayout<>;
    using L1Shape = L1TileShape_;
    using L0Shape = L0TileShape_;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using TileCopy = Tile::TileCopy<Arch::Ascend910_95, Tile::CopyNoGmIn>;

public:
    static_assert(IsF16F16F16<AType, BType, CType>() || IsF16F16F32<AType, BType, CType>() ||
                      IsBf16Bf16Bf16<AType, BType, CType>() || IsBf16Bf16F32<AType, BType, CType>() ||
                      IsF32F32F32<AType, BType, CType>(),
                  "Unsupported dtype");
    static_assert(IsNz<AType>() && IsNz<BType>() && IsND<CType>(), "L1Load only support Nz input and ND output");
    static_assert(IsTileShapeValid<L1Shape, L0Shape>(), "L1Shape or L0Shape is invalid");
    static_assert(IsL1BufferValid<AType, BType, L1Shape>(), "L1 buffer overflow");
    static_assert(IsL0BufferValid<AType, BType, L0Shape>(), "L0 buffer overflow");

    __aicore__ BlockMmad() = default;
    __aicore__ ~BlockMmad()
    {
        matmul_.End();
    }

    template <const auto& MM_CFG, typename Impl, typename InputAType, typename InputBType, typename OutputCType,
              typename InputBiasType>
    struct MatmulPolicyNew
        : public AscendC::Impl::Detail::MatmulPolicy<MM_CFG, Impl, InputAType, InputBType, OutputCType, InputBiasType> {
    public:
        template <class InputType, class OutputType, typename T = void>
        using AdaptedCubeOut = typename TileCopy::template CopyCo1ToOut<InputType, OutputType>;
        using CopyCubeOut = AscendC::Impl::Detail::CopyCubeOut<Impl, InputAType, InputBType, OutputCType, MM_CFG,
                                                               AscendC::McgShfMode::RESERVED, void, AdaptedCubeOut>;
    };

    constexpr static MatmulShapeParams shapeParams =
        GetMatmulShapeParams<typename DispatchPolicy::SingleShape, L0Shape>();
    constexpr static MatmulConfig cfg = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(
        shapeParams, GetFuncParams(DispatchPolicy::enableInputDataLenCheck), GetBiasParams(false));
    constexpr static MatmulApiStaticTiling staticTiling =
        AscendC::GetMatmulApiTiling<AType, BType, CType, BiasType, typename DispatchPolicy::SingleShape, L1Shape,
                                    L0Shape>(cfg);

public:
    __aicore__ inline void Init()
    {
        matmul_.Init((TCubeTiling*)(nullptr), (AscendC::TPipe*)(nullptr));
    }

    template <class DstTensor, class SrcATensor, class SrcBTensor, class Shape>
    __aicore__ inline void IterateAll(DstTensor& c, const SrcATensor& a, const SrcBTensor& b, const Shape& actualShape)
    {
        static_assert(AscendC::is_local_tensor_v<SrcATensor> && AscendC::is_local_tensor_v<SrcBTensor>,
                      "Input only support local tensor");

        // Convert a tensor with layout to a normal tensor
        AscendC::LocalTensor<typename AType::T> aLocal;
        aLocal.SetAddr(a.address_);
        AscendC::LocalTensor<typename BType::T> bLocal;
        bLocal.SetAddr(b.address_);
        typename AscendC::Std::conditional_t<AscendC::is_global_tensor_v<DstTensor>,
                                             AscendC::GlobalTensor<typename CType::T>,
                                             AscendC::LocalTensor<typename CType::T>>
            cTmp;

        if constexpr (AscendC::is_global_tensor_v<DstTensor>) {
            cTmp.address_ = c.address_;
        } else {
            cTmp.SetAddr(c.address_);
        }

        matmul_.SetTensorA(aLocal, AType::isTrans);
        matmul_.SetTensorB(bLocal, BType::isTrans);

        SetOrgShape(c, a, b);

        matmul_.SetSingleShape(Get<MNK_M>(actualShape), Get<MNK_N>(actualShape), Get<MNK_K>(actualShape));
        matmul_.IterateAll(cTmp);

        if constexpr (AscendC::is_global_tensor_v<DstTensor>) {
            c.address_ = cTmp.address_;
        } else {
            c.SetAddr(cTmp.address_);
        }
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

        auto orgKc = Get<1>(c.GetTensorTrait().GetLayout().GetShape()); // Set if matrix C's N != matrix B's N
        matmul_.SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc);
    }

    AscendC::MatmulImpl<AType, BType, CType, BiasType, staticTiling,
                        AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>, MatmulPolicyNew>
        matmul_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
