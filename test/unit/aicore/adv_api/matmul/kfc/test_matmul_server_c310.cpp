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
#include "detail/matmul/kfc/matmul_server_impl_c310.h"

#include "kfc_fake_modules.h"
#include "../copy_cube_in/base_tiling_struct.h"

using namespace std;
using namespace AscendC;

namespace {
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using Scheduler = CustomMatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
};
} // namespace

class TestMatmulServerC310 : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
};

TEST_F(TestMatmulServerC310, GetLocalTensor)
{
    MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> mmServer;
    uint64_t ubAddr = 0x1111;
    uint64_t ubSize = 512U;
    const auto& ubLocal = mmServer.GetLocalTensor<typename A_TYPE::T, TPosition::VECCALC>(ubAddr, ubSize);
    uint64_t tscmAddr = 0x2222;
    uint64_t tscmSize = 256U;
    const auto& tscmLocal = mmServer.GetLocalTensor<typename A_TYPE::T, TPosition::TSCM>(tscmAddr, tscmSize);

    ASSERT_TRUE((uint64_t)(ubLocal.GetPhyAddr()) == ubAddr);
    ASSERT_TRUE((uint64_t)(tscmLocal.GetPhyAddr())
                == (uint64_t)(GetTPipePtr()->GetBaseAddr((uint8_t)(TPosition::TSCM))) + tscmAddr);
}

TEST_F(TestMatmulServerC310, GetTensorC)
{
    MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>,
        CustomMatmulPolicy>
        mmServer;
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN,
    // stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 384, 2048, 192, 384, 2048, 192, 128, 256, 64, 2, 8, 3, 2, 3, 3, 0, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);

    KfcMsg kfcInitMsg;
    MSG_POS TilingInfo* tilingSSbuf = reinterpret_cast<MSG_POS TilingInfo*>(GetTilingAddr(GetSubBlockIdxImpl()));
    tilingSSbuf->valid = 1;
    auto tempTilingSSbuf = reinterpret_cast<MSG_POS uint64_t*>(&(tilingSSbuf->tCubeTiling));
    auto tempTiling = reinterpret_cast<uint64_t*>(&tiling);
    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint64_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
        *tempTilingSSbuf = *tempTiling;
    }
    mmServer.Init(&kfcInitMsg);

    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    ASSERT_TRUE(!mmServer.GetTensorC(&kfcMsg));
}
