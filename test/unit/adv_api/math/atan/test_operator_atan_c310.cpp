/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @brief vector atan instruction ut for ascend910
 *
 */
#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "kernel_utils.h"

using namespace std;
using namespace AscendC;

template <typename T, uint8_t algorithm>
void AtanKernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    SetVectorMask<uint8_t, MaskMode::NORMAL>(256);

    DataCopy(inputLocal, inputGlobal, dataSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetVectorMask<uint8_t, MaskMode::NORMAL>(128);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    static constexpr AtanConfig atanConfig = {static_cast<AtanAlgo>(algorithm)};
    Atan<T, false, atanConfig>(outputLocal, inputLocal, dataSize);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    PipeBarrier<PIPE_ALL>();
}

struct AtanTestParams {
    int32_t dataSize;
    int32_t dataBitSize;
    void (*calFunc)(uint8_t*, uint8_t*, int32_t);
};

class AtanTestsuite : public testing::Test, public testing::WithParamInterface<AtanTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_ATAN, AtanTestsuite,
    ::testing::Values(AtanTestParams { 256, 2, AtanKernel<half, 0> },
        AtanTestParams { 256, 4, AtanKernel<float, 0> },
        AtanTestParams { 256, 4, AtanKernel<float, 1> }));

TEST_P(AtanTestsuite, AtanTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.dataBitSize] = {0};
    uint8_t dstGm[param.dataSize * param.dataBitSize] = {0};

    param.calFunc(srcGm, dstGm, param.dataSize);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
