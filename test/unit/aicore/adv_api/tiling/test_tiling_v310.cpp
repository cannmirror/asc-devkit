/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
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
#include "activation/softmax_tiling.h"
#include "tiling_api.h"
#include "platform_stub.h"
#include "tiling/matmul/math_util.h"
#include "tiling/matmul/matmul_tiling_algorithm.h"
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

TEST_F(TestTiling, TestMxMatmulFP4NDTiling)
{
    // need add PlatformInfo for ASCEND910D
    // matmul_tiling::PlatformInfo plat {.socVersion = platform_ascendc::SocVersion::ASCEND910D, .l1Size = 524288,
    //     .l0CSize = 262144, .ubSize = 262144, .l0ASize = 65536, .l0BSize = 65536};
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT4_E1M2);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT4_E1M2);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetShape(64, 1088, 64);
    tiling.SetOrgShape(64, 1088, 64);
    tiling.EnableBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    tiling.PrintTilingData();
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulFP4NZTiling)
{
    // need add PlatformInfo for ASCEND910D
    // matmul_tiling::PlatformInfo plat {.socVersion = platform_ascendc::SocVersion::ASCEND910D, .l1Size = 524288,
    //     .l0CSize = 262144, .ubSize = 262144, .l0ASize = 65536, .l0BSize = 65536};
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT4_E2M1, true);
    tiling.SetBType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT4_E2M1, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetShape(128, 4096, 7168);
    tiling.SetOrgShape(128, 4096, 7168);
    tiling.EnableBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    tiling.PrintTilingData();
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulFP8NDTilingCase1)
{
    // need add PlatformInfo for ASCEND910D
    // matmul_tiling::PlatformInfo plat {.socVersion = platform_ascendc::SocVersion::ASCEND910D, .l1Size = 524288,
    //     .l0CSize = 262144, .ubSize = 262144, .l0ASize = 65536, .l0BSize = 65536};
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E5M2);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E5M2);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetShape(64, 1088, 64);
    tiling.SetOrgShape(64, 1088, 64);
    tiling.EnableBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    tiling.PrintTilingData();
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulFP8NDTilingCase2)
{
    // need add PlatformInfo for ASCEND910D
    // matmul_tiling::PlatformInfo plat {.socVersion = platform_ascendc::SocVersion::ASCEND910D, .l1Size = 524288,
    //     .l0CSize = 262144, .ubSize = 262144, .l0ASize = 65536, .l0BSize = 65536};
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E4M3FN);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E4M3FN);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetShape(8960, 5120, 2048);
    tiling.SetOrgShape(8960, 5120, 2048);
    tiling.EnableBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    tiling.PrintTilingData();
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, MultiCoreSmallMNFP4)
{
    // need add PlatformInfo for ASCEND910D
    // matmul_tiling::PlatformInfo plat {.socVersion = platform_ascendc::SocVersion::ASCEND910D, .l1Size = 524288,
    //     .l0CSize = 262144, .ubSize = 262144, .l0ASize = 65536, .l0BSize = 65536};
    matmul_tiling::MultiCoreMatmulTiling rnnMatmul3, rnnMatmul4, rnnMatmul5;
    rnnMatmul3.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT4_E1M2);
    rnnMatmul3.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT4_E1M2);
    rnnMatmul3.SetCType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType ::DT_FLOAT);
    rnnMatmul3.SetBiasType(
        matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT);
    rnnMatmul3.SetSingleRange(-1, -1, -1, -1, -1, -1);
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
    // need add PlatformInfo for ASCEND910D
    // matmul_tiling::PlatformInfo plat {.socVersion = platform_ascendc::SocVersion::ASCEND910D, .l1Size = 524288,
    //     .l0CSize = 262144, .ubSize = 262144, .l0ASize = 65536, .l0BSize = 65536};
    matmul_tiling::MultiCoreMatmulTiling rnnMatmul3, rnnMatmul4, rnnMatmul5;
    rnnMatmul3.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT8_E4M3FN);
    rnnMatmul3.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT8_E4M3FN);
    rnnMatmul3.SetCType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType ::DT_FLOAT);
    rnnMatmul3.SetBiasType(
        matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT);
    rnnMatmul3.SetSingleRange(-1, -1, -1, -1, -1, -1);
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

TEST_F(TestTiling, TestMxMatmulFP8NDTiling_CaseTscm)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E5M2, true);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E5M2, false);
    tiling.SetScaleAType(TPosition::TSCM, CubeFormat::NZ, false);
    tiling.SetScaleBType(TPosition::TSCM, CubeFormat::NZ, true);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    tiling.SetShape(1, 1920, 64);
    tiling.SetOrgShape(1, 1920, 64);

    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    tiling.PrintTilingData();
    EXPECT_EQ(ret, 0);
}
