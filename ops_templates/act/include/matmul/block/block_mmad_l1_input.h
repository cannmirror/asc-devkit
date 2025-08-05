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
 * \file block_mmad_l1_input.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_L1_INPUT_H
#define ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_L1_INPUT_H

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
class BlockMmad<MatmulL1Input<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_,
                Tile::TileCopy<Arch::Ascend910_95, Tile::CopyNoGmIn>> {
public:
    using DispatchPolicy = MatmulL1Input<>;
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
                      IsI8I8I32<AType, BType, CType>() || IsF32F32F32<AType, BType, CType>(),
                  "Unsupported dtype");
    static_assert(IsNz<AType>() && IsNz<BType>() && IsND<CType>(), "L1Load only support Nz input and ND output");
    static_assert(IsTileShapeValid<L1Shape, L0Shape>(), "L1Shape or L0Shape is invalid");
    static_assert(IsL1BufferValid<AType, BType, L1Shape>(), "L1 buffer overflow");
    static_assert(IsL0BufferValid<AType, BType, L0Shape>(), "L0 buffer overflow");

    __aicore__ BlockMmad() = default;
    __aicore__ ~BlockMmad() = default;

    template <const auto& MM_CFG, typename Impl, typename InputAType, typename InputBType, typename OutputCType,
              typename InputBiasType>
    struct MatmulPolicyNew
        : public AscendC::Impl::Detail::MatmulPolicy<MM_CFG, Impl, InputAType, InputBType, OutputCType, InputBiasType> {
    public:
        template <class InputType, class OutputType, typename T = void>
        using AdaptedCubeOut = typename TileCopy::template CopyCo1ToOut<InputType, OutputType>;
        using CopyCubeOut =
            AscendC::Impl::Detail::CopyCubeOut<Impl, InputAType, InputBType, OutputCType, MM_CFG,
                                               AscendC::McgShfMode::SINGLE_DST_MODE, void, AdaptedCubeOut>;
    };

    constexpr static MatmulShapeParams shapeParams =
        GetMatmulShapeParams<typename DispatchPolicy::SingleShape, L0Shape>();
    constexpr static MatmulConfig cfg = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(
        shapeParams, GetFuncParams(DispatchPolicy::enableInputDataLenCheck), GetBiasParams(false));
    constexpr static MatmulApiStaticTiling staticTiling =
        AscendC::GetMatmulApiTiling<AType, BType, CType, BiasType, typename DispatchPolicy::SingleShape, L1Shape,
                                    L0Shape>(cfg);

public:
    __aicore__ inline void Init(TCubeTiling* __restrict cubeTiling, AscendC::TPipe* tpipe = nullptr)
    {
        matmul_.Init(cubeTiling, tpipe);
    }
    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK)
    {
        matmul_.SetOrgShape(orgM, orgN, orgK);
    }
    __aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK)
    {
        matmul_.SetSingleShape(singleM, singleN, singleK);
    }
    __aicore__ inline void SetTensorA(const AscendC::LocalTensor<typename AType::T>& aLocal, bool isTransposeA = false)
    {
        matmul_.SetTensorA(aLocal, isTransposeA);
    }
    __aicore__ inline void SetTensorB(const AscendC::LocalTensor<typename BType::T>& bLocal, bool isTransposeB = false)
    {
        matmul_.SetTensorB(bLocal, isTransposeB);
    }
    __aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx)
    {
        matmul_.SetSubBlockIdx(subBlockIdx);
    }
    __aicore__ inline void IterateAll(const AscendC::GlobalTensor<typename CType::T>& gm, uint8_t enAtomic = 0)
    {
        matmul_.IterateAll(gm, enAtomic);
    }
    __aicore__ inline void IterateAll(const AscendC::LocalTensor<typename CType::T>& ubCmatrix, uint8_t enAtomic = 0)
    {
        matmul_.IterateAll(ubCmatrix, enAtomic);
    }
    __aicore__ inline bool Iterate(bool enPartialSum = false)
    {
        return matmul_.Iterate(enPartialSum);
    }
    __aicore__ inline void GetTensorC(const AscendC::GlobalTensor<typename CType::T>& gm, uint8_t enAtomic = 0)
    {
        matmul_.GetTensorC(gm, enAtomic);
    }
    __aicore__ inline void End()
    {
        matmul_.End();
    }

private:
    AscendC::MatmulImpl<AType, BType, CType, BiasType, staticTiling,
                        AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>, MatmulPolicyNew>
        matmul_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
