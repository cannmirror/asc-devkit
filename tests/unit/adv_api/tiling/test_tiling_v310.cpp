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
#include "graph/tensor.h"
#include <dlfcn.h>
#define private public
#define protected public
#include "include/adv_api/activation/softmax_tiling.h"
#include "tiling_api.h"
#include "platform_stub.h"
#include "impl/adv_api/tiling/matmul/math_util.h"
#include "impl/adv_api/tiling/matmul/matmul_tiling_algorithm.h"
#include "tiling/platform/platform_ascendc.h"
using namespace AscendC;
using namespace ge;
using namespace std;
using namespace matmul_tiling;

class TestTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(TestTiling, MultiCoreSmallMNFP4)
{
    matmul_tiling::MultiCoreMatmulTiling rnnMatmul3,rnnMatmul4,rnnMatmul5;
    rnnMatmul3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT4_E1M2);
    rnnMatmul3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT4_E1M2);
    rnnMatmul3.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType ::DT_FLOAT);
    rnnMatmul3.SetBiasType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT);
    rnnMatmul3.SetSingleRange(-1,-1,-1,-1,-1,-1);
    rnnMatmul3.EnableMultiCoreSplitK(true);
    auto ret = rnnMatmul3.EnableBias(true);
    ret = rnnMatmul3.SetDim(32);
    ret = rnnMatmul3.SetOrgShape(5, 40, 986);
    ret = rnnMatmul3.SetShape(5, 10, 986);
    ret = rnnMatmul3.SetBufferSpace(-1, -1, -1); // will use all buffer space if not explicitly specified
    optiling::TCubeTiling tilingData;
    ret = rnnMatmul3.GetTiling(tilingData);
    rnnMatmul3.PrintTilingData();
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, MultiCoreSmallMNFP8)
{
    matmul_tiling::MultiCoreMatmulTiling rnnMatmul3,rnnMatmul4,rnnMatmul5;
    rnnMatmul3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT8_E4M3FN);
    rnnMatmul3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT8_E4M3FN);
    rnnMatmul3.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType ::DT_FLOAT);
    rnnMatmul3.SetBiasType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT);
    rnnMatmul3.SetSingleRange(-1,-1,-1,-1,-1,-1);
    rnnMatmul3.EnableMultiCoreSplitK(true);
    auto ret = rnnMatmul3.EnableBias(true);
    ret = rnnMatmul3.SetDim(32);
    ret = rnnMatmul3.SetOrgShape(5, 40, 986);
    ret = rnnMatmul3.SetShape(5, 10, 986);
    ret = rnnMatmul3.SetBufferSpace(-1, -1, -1); // will use all buffer space if not explicitly specified
    optiling::TCubeTiling tilingData;
    ret = rnnMatmul3.GetTiling(tilingData);
    rnnMatmul3.PrintTilingData();
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMatmulTilingEnableL1BankConflictOptimiseFP16)
{
    MultiCoreMatmulTiling tiling;
    tiling.SetDim(1);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    int32_t m = 1024;
    int32_t n = 1024;
    int32_t k = 1024;

    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(true);
    tiling.SetDequantType(DequantType::TENSOR);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    int ret1 = tiling.EnableL1BankConflictOptimise();

    tiling.PrintTilingData();
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMatmulTilingEnableL1BankConflictOptimiseFP16TSCM)
{
    MultiCoreMatmulTiling tiling;
    tiling.SetDim(1);
    tiling.SetAType(TPosition::TSCM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tiling.SetBType(TPosition::TSCM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    int32_t m = 256;
    int32_t n = 256;
    int32_t k = 256;

    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    int ret1 = tiling.EnableL1BankConflictOptimise();

    tiling.PrintTilingData();
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMatmulTilingEnableL1BankConflictOptimiseFP16NORM)
{
    MultiCoreMatmulTiling tiling;
    tiling.SetDim(1);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    int32_t m = 768;
    int32_t n = 768;
    int32_t k = 768;

    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMatmulConfigParams(0);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    int ret1 = tiling.EnableL1BankConflictOptimise();

    tiling.PrintTilingData();
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMatmulTilingEnableL1BankConflictOptimiseFP32)
{
    MultiCoreMatmulTiling tiling;
    tiling.SetDim(1);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    int32_t m = 16;
    int32_t n = 16;
    int32_t k = 16;

    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    int ret1 = tiling.EnableL1BankConflictOptimise();

    tiling.PrintTilingData();
    EXPECT_EQ(ret, 0);
}
