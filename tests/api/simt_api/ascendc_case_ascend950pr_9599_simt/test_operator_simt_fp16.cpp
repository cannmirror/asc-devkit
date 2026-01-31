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
#include <random>
#include "simt_compiler_stub.h"
#include "kernel_operator.h"
#include "simt_api/asc_fp16.h"

using namespace std;
using namespace AscendC;
using namespace AscendC::Simt;

#define THREAD_DIM 128
template <typename T>
class KernelFP16 {
    public:
        __aicore__ KernelFP16() {}
        __aicore__ inline void Process(__gm__ T* dst, int32_t mode);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelFP16Compute(__gm__ T* dst, int32_t mode)
{
    for(int idx=GetThreadIdx<0>()+block_idx*GetThreadNum<0>();idx < 1; idx+=block_num*GetThreadNum<0>()) {
        if(mode == 0) {
            half a = 2.3;
            a++;
            dst[idx] = ++a;
        } else if(mode == 1) {
            half a = 2.1;
            a--;
            dst[idx] = --a;
        } else if(mode == 2) {
            half a = 2.1;
            half b = 0.0;
            dst[idx] = a && b;
        } else if(mode == 3) {
            half a = 2.1;
            half b = 0.0;
            dst[idx] = a || b;
        } else if(mode == 4) {
            half a = 2.1;
            dst[idx] = Floor(a);
        } else if(mode == 5) {
            half a = 2.1;
            dst[idx] = Rint(a);
        } else if(mode == 6) {
            half a = 2.1;
            dst[idx] = Ceil(a);
        }
    }
}

template <typename T>
__aicore__ inline void KernelFP16<T>::Process(__gm__ T* dst, int32_t mode)
{
    AscendC::Simt::VF_CALL<KernelFP16Compute<T>>(AscendC::Simt::Dim3(THREAD_DIM, 1, 1), dst, mode);
}

// ================================ Test half start ===============================
struct FP16Params {
    int32_t mode;
};

class FP16Testsuite : public testing::Test, public testing::WithParamInterface<FP16Params> {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(FP16Testsuite, MathApiTest)
{
    float x = static_cast<float>(rand()) / RAND_MAX;
    half x_h = half(x);

    half result = hcos(x_h);
    half expect = half(cosf(x_h.ToFloat()));
    EXPECT_EQ(expect, result);
    
    result = hsin(x_h);
    expect = half(sinf(x_h.ToFloat()));
    EXPECT_EQ(expect, result);

    result = htanh(x_h);
    expect = half(tanhf(x_h.ToFloat()));
    EXPECT_EQ(expect, result);

    result = hexp(x_h);
    expect = half(expf(x_h.ToFloat()));
    EXPECT_EQ(expect, result);

    result = hexp2(x_h);
    expect = half(exp2f(x_h.ToFloat()));
    EXPECT_EQ(expect, result);
    
    result = hexp10(x_h);
    expect = half(powf(10.0, x_h.ToFloat()));
    EXPECT_EQ(expect, result);

    result = hrcp(x_h);
    if (x_h != half(0)) {
        expect = half(1.0f / x_h.ToFloat());
        EXPECT_EQ(expect, result);
    }
}
// ================================ Test half end ===============================

// ================================ Test half2 start ===============================
struct FP162Params {
    int32_t mode;
};

class FP162Testsuite : public testing::Test, public testing::WithParamInterface<FP162Params> {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(FP162Testsuite, MathApiTest_half2)
{
    float x = static_cast<float>(rand()) / RAND_MAX + 0.1f;
    float y = static_cast<float>(rand()) / RAND_MAX + 0.1f;
    half x_h = half(x);
    half y_h = half(y);
    half2 xy_h2 = {x_h, y_h};

    half2 result = h2cos(xy_h2);
    half expect1 = half(cosf(x_h.ToFloat()));
    half expect2 = half(cosf(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);
    
    result = h2sin(xy_h2);
    expect1 = half(sinf(x_h.ToFloat()));
    expect2 = half(sinf(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2tanh(xy_h2);
    expect1 = half(tanhf(x_h.ToFloat()));
    expect2 = half(tanhf(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2exp(xy_h2);
    expect1 = half(expf(x_h.ToFloat()));
    expect2 = half(expf(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2exp2(xy_h2);
    expect1 = half(exp2f(x_h.ToFloat()));
    expect2 = half(exp2f(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);
    
    result = h2exp10(xy_h2);
    expect1 = half(powf(10.0, x_h.ToFloat()));
    expect2 = half(powf(10.0, y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2log(xy_h2);
    if (x_h > half(0) && y_h > half(0)) {
        expect1 = half(log(x_h.ToFloat()));
        expect2 = half(log(y_h.ToFloat()));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2log2(xy_h2);
    if (x_h > half(0) && y_h > half(0)) {
        expect1 = half(log2(x_h.ToFloat()));
        expect2 = half(log2(y_h.ToFloat()));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2log10(xy_h2);
    if (x_h > half(0) && y_h > half(0)) {
        expect1 = half(log10(x_h.ToFloat()));
        expect2 = half(log10(y_h.ToFloat()));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2rcp(xy_h2);
    if (x_h != half(0) && y_h != half(0)) {
        expect1 = half(1.0f / x_h.ToFloat());
        expect2 = half(1.0f / y_h.ToFloat());
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }
    
    result = h2sqrt(xy_h2);
    if (x_h > half(0) && y_h > half(0)) {
        expect1 = half(sqrt(x_h.ToFloat()));
        expect2 = half(sqrt(y_h.ToFloat()));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2rsqrt(xy_h2);
    if (x_h > half(0) &&y_h > half(0)) {
        expect1 = half(1.0f / sqrt(x_h.ToFloat()));
        expect2 = half(1.0f / sqrt(y_h.ToFloat()));
    }
    
    result = h2ceil(xy_h2);
    expect1 = half(ceil(x_h.ToFloat()));
    expect2 = half(ceil(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2floor(xy_h2);
    expect1 = half(floor(x_h.ToFloat()));
    expect2 = half(floor(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2rint(xy_h2);
    expect1 = half(rint(x_h.ToFloat()));
    expect2 = half(rint(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2trunc(xy_h2);
    expect1 = half(trunc(x_h.ToFloat()));
    expect2 = half(trunc(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);
}
// ================================ Test half2 end ===============================