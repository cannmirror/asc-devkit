/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
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

#include "include/matmul/block/block_mmad_multi_block_with_layout.h"
#include "include/matmul/policy/dispatch_policy.h"
#include "include/utils/layout_utils.h"

namespace {
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
} // namespace

template <class AType, class BType, class CType, class BiasType>
__aicore__ inline void BlockMmadKernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR bias, const TilingParams& tilingParam)
{
    if (AscendC::GetBlockIdx() > 0) {
        return;
    }

    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;

    AscendC::TPipe tpipe;

    TCubeTiling tiling;
    tilingParam.GetTiling(tiling);
    uint32_t m = tiling.M;
    uint32_t n = tiling.N;
    uint32_t ka = tiling.Ka;
    uint32_t kb = tiling.Kb;

    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<C_T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(a), m * ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(b), n * kb);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(c), m * n);

    // A
    auto aLayout = Act::Gemm::MakeLayoutByFormat<A_T, AType::format>(AType::isTrans ? ka : m, AType::isTrans ? m : ka);
    auto aTensorTrait = AscendC::MakeTensorTrait<A_T, AType::pos>(aLayout);
    AscendC::GlobalTensor<decltype(aTensorTrait)> aWithLayout;
    aWithLayout.SetTensorTrait(aTensorTrait);
    aWithLayout.address_ = aGlobal.address_;

    // B
    auto bLayout = Act::Gemm::MakeLayoutByFormat<B_T, BType::format>(BType::isTrans ? n : kb, BType::isTrans ? kb : n);
    auto bTensorTrait = AscendC::MakeTensorTrait<B_T, BType::pos>(bLayout);
    AscendC::GlobalTensor<decltype(bTensorTrait)> bWithLayout;
    bWithLayout.SetTensorTrait(bTensorTrait);
    bWithLayout.address_ = bGlobal.address_;

    // C
    auto cLayout = Act::Gemm::MakeLayoutByFormat<C_T, CType::format>(m, n);
    auto cTensorTrait = AscendC::MakeTensorTrait<C_T, CType::pos>(cLayout);
    AscendC::GlobalTensor<decltype(cTensorTrait)> cWithLayout;
    cWithLayout.SetTensorTrait(cTensorTrait);
    cWithLayout.address_ = cGlobal.address_;

    // blockmmad
    using L1Shape = AscendC::Shape<Act::Gemm::_128, Act::Gemm::_256, Act::Gemm::_128>;
    using L0Shape = AscendC::Shape<Act::Gemm::_128, Act::Gemm::_256, Act::Gemm::_64>;
    typename Act::Gemm::Block::BlockMmad<Act::Gemm::MatmulMultiBlockWithLayout<>, L1Shape, L0Shape, AType, BType, CType,
                                         BiasType>
        matmulObj;
    matmulObj.Init();
    matmulObj.IterateAll(cWithLayout, aWithLayout, bWithLayout, AscendC::MakeShape(m, n, ka));
}

class TestBlockMmadMultiBlockWithLayout : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

#define BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(tilingParams, aT, bT, cT, biasT, transA, transB)                   \
    namespace BlockMmadMultiBlockWithLayout_Case_##tilingParams##_##aT##_##bT##_##cT##_##biasT##_##transA##_##transB   \
    {                                                                                                                  \
        using AType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aT, transA>;                         \
        using BType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bT, transB>;                         \
        using CType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cT>;                                 \
        using BiasType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasT>;                           \
        TEST_F(TestBlockMmadMultiBlockWithLayout,                                                                      \
               BlockMmadMultiBlockWithLayout_Case_##tilingParams##_##aT##_##bT##_##cT##_##biasT##_##transA##_##transB) \
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
TilingParams params1(128, 256, 128, 0);
TilingParams params2(311, 25, 67, 0);
TilingParams params3(1, 1, 11, 0);
TilingParams params4(2, 1, 2, 0);

BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params1, half, half, half, half, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params1, half, half, float, float, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params1, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params1, bfloat16_t, bfloat16_t, float, float, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params1, float, float, float, float, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params1, int8_t, int8_t, int32_t, int32_t, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, half, half, half, half, false, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, half, half, half, half, false, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, half, half, half, half, true, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, half, half, half, half, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, half, half, float, float, false, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, half, half, float, float, false, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, half, half, float, float, true, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, half, half, float, float, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, bfloat16_t, bfloat16_t, float, float, false, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, bfloat16_t, bfloat16_t, float, float, false, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, bfloat16_t, bfloat16_t, float, float, true, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, bfloat16_t, bfloat16_t, float, float, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, float, float, float, float, false, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, float, float, float, float, false, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, float, float, float, float, true, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, float, float, float, float, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, int8_t, int8_t, int32_t, int32_t, false, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, int8_t, int8_t, int32_t, int32_t, false, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, int8_t, int8_t, int32_t, int32_t, true, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params2, int8_t, int8_t, int32_t, int32_t, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params3, half, half, float, float, false, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params3, half, half, float, float, false, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params3, half, half, float, float, true, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params3, half, half, float, float, true, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params4, half, half, float, float, false, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params4, half, half, float, float, false, true);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params4, half, half, float, float, true, false);
BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_TESTCASE(params4, half, half, float, float, true, true);