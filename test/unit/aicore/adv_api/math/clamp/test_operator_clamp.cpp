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
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;

static const uint32_t tmpBufferSize = 1537;

enum TestMode {
    CLAMP_MAX,
    CLAMP_MIN,
};

enum TmpMode {
    MODE_NORMAL,
    MODE_TMPBUFFER,
};

class TEST_CLAMP : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2);
    }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

template <typename T>
void main_vec_clamp_demo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm, uint32_t dataSize,
    TestMode testMode, TmpMode tmpMode)
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

    DataCopy(inputLocal, inputGlobal, dataSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    T scalar = 1;
    if (tmpMode == MODE_NORMAL) {
        if (testMode == CLAMP_MAX) {
            ClampMax<T, false>(outputLocal, inputLocal, scalar, dataSize);
        } else if (testMode == CLAMP_MIN) {
            ClampMin<T, false>(outputLocal, inputLocal, scalar, dataSize);
        }
    } else if (tmpMode == MODE_TMPBUFFER) {
        LocalTensor<uint8_t> sharedTmpBuffer;
        bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
        sharedTmpBuffer.SetSize(tmpBufferSize);
        if (testMode == CLAMP_MAX) {
            ClampMax<T, false>(outputLocal, inputLocal, sharedTmpBuffer, scalar, dataSize);
        } else if (testMode == CLAMP_MIN) {
            ClampMin<T, false>(outputLocal, inputLocal, sharedTmpBuffer, scalar, dataSize);
        }
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    PipeBarrier<PIPE_ALL>();
}
#define VEC_CLAMP_TESTCASE(DATA_TYPE, TEST_MODE, CALCOUNT, TMP_MODE)                                                   \
    TEST_F(TEST_CLAMP, Clamp##_##DATA_TYPE##_##CALCOUNT##_##TEST_MODE##_##TMP_MODE##_##Case)                           \
    {                                                                                                                  \
        uint32_t dataSize = CALCOUNT;                                                                                  \
        uint8_t inputGm[dataSize * sizeof(DATA_TYPE)];                                                                 \
        uint8_t outputGm[dataSize * sizeof(DATA_TYPE)];                                                                \
                                                                                                                       \
        main_vec_clamp_demo<DATA_TYPE>(outputGm, inputGm, dataSize, TEST_MODE, TMP_MODE);                              \
                                                                                                                       \
        for (uint32_t i = 0; i < dataSize; i++) { EXPECT_EQ(outputGm[i], 0x00); }                                      \
    }
VEC_CLAMP_TESTCASE(float, CLAMP_MAX, 256, MODE_NORMAL);
VEC_CLAMP_TESTCASE(float, CLAMP_MIN, 256, MODE_NORMAL);
VEC_CLAMP_TESTCASE(half, CLAMP_MAX, 256, MODE_NORMAL);
VEC_CLAMP_TESTCASE(half, CLAMP_MIN, 256, MODE_NORMAL);
VEC_CLAMP_TESTCASE(half, CLAMP_MIN, 2048, MODE_TMPBUFFER);
