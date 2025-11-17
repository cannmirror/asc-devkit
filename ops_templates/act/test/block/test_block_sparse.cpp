/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "kernel_event.h"
#include "kernel_tiling/kernel_tiling.h"

#define __host_aicore__ __aicore__
#define ASCENDC_ASSERT(...)

#include "include/matmul/matmul_intf.h"
#include "include/matmul/policy/dispatch_policy.h"
#include "include/utils/layout_utils.h"

namespace Act{
namespace Gemm {
struct TilingParams {
    __aicore__ TilingParams() = default;
    __aicore__ TilingParams(uint32_t m, uint32_t n, uint32_t k, uint32_t isBias) : m_(m), n_(n), k_(k), isBias_(isBias)
    {}
    __aicore__ void GetTiling(TCubeTiling& tiling) const
    {
        tiling.usedCoreNum = 1;
        tiling.M = m_;
        tiling.N = n_;
        tiling.Ka = k_;
        tiling.Kb = k_;
        tiling.isBias = isBias_;
    }
    uint32_t m_{64};
    uint32_t n_{64};
    uint32_t k_{64};
    uint32_t isBias_{0};
};

template <class AType, class BType, class CType, class BiasType>
__aicore__ inline void BlockMmadKernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR bias, const TilingParams& tilingParam)
{
    if (AscendC::GetBlockIdx() >= 1) {
        return;
    }

    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;
    using BiasT = typename BiasType::T;

    AscendC::TPipe tpipe;

    TCubeTiling tiling;
    tilingParam.GetTiling(tiling);
    int m = tiling.M;
    int n = tiling.N;
    int ka = tiling.Ka;
    int kb = tiling.Kb;

    // A
    AscendC::Layout<AscendC::Shape<int, int>, AscendC::Stride<int, int>> aLayout;
    if constexpr (!AType::isTrans) {
        aLayout = AscendC::MakeLayout(AscendC::MakeShape(m, ka), AscendC::MakeStride(ka, 1));
    } else {
        aLayout = AscendC::MakeLayout(AscendC::MakeShape(ka, m), AscendC::MakeStride(m, 1));
    }
    auto aTensorTrait = AscendC::MakeTensorTrait<A_T, AscendC::TPosition::GM>(aLayout);
    AscendC::GlobalTensor<decltype(aTensorTrait)> aGlobal;
    aGlobal.SetTensorTrait(aTensorTrait);
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a), m * ka);

    // B
    AscendC::Layout<AscendC::Shape<int, int>, AscendC::Stride<int, int>> bLayout;
    if constexpr (!BType::isTrans) {
        bLayout = AscendC::MakeLayout(AscendC::MakeShape(kb, n), AscendC::MakeStride(n, 1));
    } else {
        bLayout = AscendC::MakeLayout(AscendC::MakeShape(n, kb), AscendC::MakeStride(kb, 1));
    }
    auto bTensorTrait = AscendC::MakeTensorTrait<B_T, AscendC::TPosition::GM>(bLayout);
    AscendC::GlobalTensor<decltype(bTensorTrait)> bGlobal;
    bGlobal.SetTensorTrait(bTensorTrait);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(b), kb * n);

    // C
    auto cLayout = AscendC::MakeLayout(AscendC::MakeShape(m, n), AscendC::MakeStride(n, 1));
    auto cTensorTrait = AscendC::MakeTensorTrait<C_T, AscendC::TPosition::GM>(cLayout);
    AscendC::GlobalTensor<decltype(cTensorTrait)> cGlobal;
    cGlobal.SetTensorTrait(cTensorTrait);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(c), m * n);

    // Init
    using A1B1Shape = AscendC::Shape<_128, _256, _128, _128>; // l1shape
    using A2B2Shape = AscendC::Shape<_128, _256, _64>; // baseshape
    typename Block::BlockMmad<MatmulNaivePipelineWithLayout<>,
        A1B1Shape, A2B2Shape, AType, BType, CType, BiasType> matmulObj;
    matmulObj.Init();

    auto actualShape = AscendC::MakeShape(m, n, ka);
    matmulObj.IterateAll(cGlobal, aGlobal, bGlobal, actualShape);
}

class BlockMmadMultiBlockSparse : public testing::Test {
protected:
    void SetUp() {
        AscendC::SetGCoreType(1);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

#define BLOCK_MMAD_MULTI_BLOCKSPARSE_TESTCASE(tilingParams, aT, bT, cT, biasT, transA, transB)                   \
    namespace BlockMmadMultiBlockSparse_Case_##tilingParams##_##aT##_##bT##_##cT##_##biasT##_##transA##_##transB   \
    {                                                                                                                  \
        using AType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aT, transA>;                         \
        using BType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bT, transB>;                         \
        using CType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cT>;                                 \
        using BiasType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasT>;                           \
        TEST_F(BlockMmadMultiBlockSparse,                                                                      \
               BlockMmadMultiBlockSparse_Case_##tilingParams##_##aT##_##bT##_##cT##_##biasT##_##transA##_##transB) \
        {                                                                                                              \
            uint8_t aGM[tilingParams.m_ * tilingParams.k_ * sizeof(aT)] = {0};                                         \
            uint8_t bGM[tilingParams.n_ * tilingParams.k_ * sizeof(bT)] = {0};                                         \
            uint8_t cGM[tilingParams.m_ * tilingParams.n_ * sizeof(cT)] = {0};                                         \
            uint8_t biasGM[tilingParams.n_ * sizeof(biasT)] = {0};                                                     \
            BlockMmadKernel<AType, BType, CType, BiasType>(aGM, bGM, cGM, biasGM, tilingParams);                       \
            for (int32_t i = 0; i < tilingParams.m_ * tilingParams.n_ * sizeof(cT); i++) {                             \
                EXPECT_EQ(cGM[i], 0x0);                                                                                \
            }                                                                                                          \
        }                                                                                                              \
    }

// m, n, k, isBias
TilingParams paramsA(128, 256, 128, 0);
TilingParams paramsB(311, 25, 67, 0);

BLOCK_MMAD_MULTI_BLOCKSPARSE_TESTCASE(paramsA, half, half, half, half, true, true);
BLOCK_MMAD_MULTI_BLOCKSPARSE_TESTCASE(paramsA, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true);

BLOCK_MMAD_MULTI_BLOCKSPARSE_TESTCASE(paramsB, half, half, half, half, false, false);
BLOCK_MMAD_MULTI_BLOCKSPARSE_TESTCASE(paramsB, half, half, half, half, false, true);
BLOCK_MMAD_MULTI_BLOCKSPARSE_TESTCASE(paramsB, half, half, half, half, true, false);
BLOCK_MMAD_MULTI_BLOCKSPARSE_TESTCASE(paramsB, half, half, half, half, true, true);
BLOCK_MMAD_MULTI_BLOCKSPARSE_TESTCASE(paramsB, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false);
BLOCK_MMAD_MULTI_BLOCKSPARSE_TESTCASE(paramsB, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true);
BLOCK_MMAD_MULTI_BLOCKSPARSE_TESTCASE(paramsB, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false);
BLOCK_MMAD_MULTI_BLOCKSPARSE_TESTCASE(paramsB, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true);

}
}