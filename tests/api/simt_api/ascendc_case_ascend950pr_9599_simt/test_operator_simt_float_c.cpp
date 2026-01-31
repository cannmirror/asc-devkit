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
#include <type_traits>
#include "simt_compiler_stub.h" // must include before simt api header
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace std;
using namespace AscendC;
using namespace AscendC::Simt;
namespace {
constexpr uint32_t THREAD_DIM = 128;
struct TranscendentalParams {
    int32_t mode;
};
}

template <typename T>
class KernelMath {
    public:
        __aicore__ KernelMath() {}
        __aicore__ inline void Process(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, const int mode);
};


template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__  void KernelMathCompute(__gm__ T* dst, __gm__ T* src0, __gm__ T* src1, const int mode)
{
    src0[0] = NAN;
    src1[0] = NAN;
    src0[1] = INFINITY;
    src1[1] = INFINITY;
    src0[2] = -INFINITY;
    src1[2] = INFINITY;
    src0[3] = INFINITY;
    src1[3] = -INFINITY;
    src0[4] = INFINITY;
    src0[5] = -INFINITY;
    src0[6] = 100.0f;
    src0[7] = -2.0f;
    src1[6] = INFINITY;
    src1[7] = -INFINITY;
    for(int idx = GetThreadIdx<0>() + block_idx*GetThreadNum<0>(); idx < 128; idx+=block_num*GetThreadNum<0>())
    {
        if (mode == 6) {
            dst[idx] = expf(src0[idx]);
        }
    }
}

template <typename T>
__aicore__  inline void KernelMath<T>::Process(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, const int mode)
{
    AscendC::Simt::VF_CALL<KernelMathCompute<T>>(Dim3(THREAD_DIM, 1, 1), out, src0, src1, mode);
}

class MathTestsuite : public testing::Test, public testing::WithParamInterface<TranscendentalParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TranscendentalTestCase, MathTestsuite,
    ::testing::Values(TranscendentalParams {0},
    TranscendentalParams {1},
    TranscendentalParams {2},
    TranscendentalParams {3},
    TranscendentalParams {4},
    TranscendentalParams {5},
    TranscendentalParams {6},
    TranscendentalParams {7},
    TranscendentalParams {8},
    TranscendentalParams {9},
    TranscendentalParams {10},
    TranscendentalParams {11},
    TranscendentalParams {12},
    TranscendentalParams {13},
    TranscendentalParams {14},
    TranscendentalParams {15},
    TranscendentalParams {16},
    TranscendentalParams {17},
    TranscendentalParams {18},
    TranscendentalParams {19},
    TranscendentalParams {20},
    TranscendentalParams {21},
    TranscendentalParams {22},
    TranscendentalParams {23},
    TranscendentalParams {24},
    TranscendentalParams {25},
    TranscendentalParams {26},
    TranscendentalParams {27},
    TranscendentalParams {28},
    TranscendentalParams {29},
    TranscendentalParams {30},
    TranscendentalParams {31},
    TranscendentalParams {32},
    TranscendentalParams {33},
    TranscendentalParams {34},
    TranscendentalParams {35},
    TranscendentalParams {36},
    TranscendentalParams {37}
                      ));

TEST_P(MathTestsuite, TranscendentalTestCase)
{
    auto param = GetParam();
    int32_t mode = param.mode;
    int fpByteSize = 4;
    int shapeSize = 128;

    uint8_t dstGm[shapeSize * fpByteSize] = {0};
    uint8_t src0Gm[shapeSize * fpByteSize] = {0};
    uint8_t src1Gm[shapeSize * fpByteSize] = {0};
    KernelMath<float> op;
    op.Process((__gm__ float*)dstGm, (__gm__ float*)src0Gm, (__gm__ float*)src1Gm, mode);
}

void VerifyFloatNumber1(float x, float xExpected, float epsilon = 1e-4)
{
    if (std::isnan(xExpected)) {
        EXPECT_TRUE(std::isnan(x));
    } else if (std::isinf(xExpected)) {
        EXPECT_TRUE(std::isinf(x));
        if (xExpected > 0.0) {
            EXPECT_GT(x, 0.0);
        } else {
            EXPECT_LT(x, 0.0);
        }
    } else {
        EXPECT_NEAR(x, xExpected, epsilon);
    }
}

// ================================ Test atanf start ================================
struct atanfTestParam {
    float x;
    float yExpected;
};
 
class atanfTestSuite : public ::testing::TestWithParam<atanfTestParam> {
public:
    void SetUp() override {}
    void TearDown() override {}
};

INSTANTIATE_TEST_CASE_P(atanfTestCaseFloat, atanfTestSuite, ::testing::Values(
    atanfTestParam {INFINITY, 1.57079632679489661923132169163975144f},
    atanfTestParam {-INFINITY, -1.57079632679489661923132169163975144f},
    atanfTestParam {NAN, NAN}
));

TEST_P(atanfTestSuite, atanfTestCaseFloat)
{
    const auto param = this->GetParam();
    float x = param.x;
    float yExpected = param.yExpected;
    float y = atanf(x);
    VerifyFloatNumber1(y, yExpected);
}
// ================================ Test atanf end ==================================