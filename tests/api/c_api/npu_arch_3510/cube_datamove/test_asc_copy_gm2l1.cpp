/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

__aicore__ inline void copy_gm_to_cbuf_v2_stub(__cbuf__ void* dst, __gm__ void* src, uint8_t sid, uint32_t n_burst, uint32_t len_burst, uint8_t pad_func_mode,
                                            uint64_t src_stride, uint32_t dst_stride) {
    EXPECT_EQ(dst, reinterpret_cast<__cbuf__ void *>(11));
    EXPECT_EQ(src, reinterpret_cast<__gm__ void *>(22));
    EXPECT_EQ(sid, static_cast<uint8_t>(0));
    EXPECT_EQ(n_burst, static_cast<uint32_t>(33));
    EXPECT_EQ(len_burst, static_cast<uint32_t>(44));
    EXPECT_EQ(pad_func_mode, static_cast<uint8_t>(55));
    EXPECT_EQ(src_stride, static_cast<uint64_t>(66));
    EXPECT_EQ(dst_stride, static_cast<uint64_t>(77));
}

class TEST_COPY_GM_TO_L1 : public testing::Test {
protected:
    void SetUp() {
        g_coreType = C_API_AIC_TYPE;
    }
    void TearDown() {
        g_coreType = C_API_AIV_TYPE;
    }
};

TEST_F(TEST_COPY_GM_TO_L1, TEST_COPY_GM_TO_L1)
{
    MOCKER_CPP(copy_gm_to_cbuf_v2, void(__cbuf__ void*, __gm__ void*, uint8_t, uint32_t, uint32_t, uint8_t,
                                    uint64_t, uint32_t))
            .times(1)
            .will(invoke(copy_gm_to_cbuf_v2_stub));

    __cbuf__ void *dst = reinterpret_cast<__cbuf__ void *>(11);
    __gm__ void *src = reinterpret_cast<__gm__ void *>(22);

    uint16_t n_burst = static_cast<uint32_t>(33);
    uint16_t len_burst = static_cast<uint32_t>(44);
    uint16_t pad_func_mode = static_cast<uint8_t>(55);
    uint16_t src_stride = static_cast<uint64_t>(66);
    uint16_t dst_stride = static_cast<uint64_t>(77);

    asc_copy_gm2l1(dst, src, n_burst, len_burst, pad_func_mode, src_stride, dst_stride);
    GlobalMockObject::verify();
}

TEST_F(TEST_COPY_GM_TO_L1, TEST_COPY_GM_TO_L1_SYNC)
{
    MOCKER_CPP(copy_gm_to_cbuf_v2, void(__cbuf__ void*, __gm__ void*, uint8_t, uint32_t, uint32_t, uint8_t,
                                    uint64_t, uint32_t))
            .times(1)
            .will(invoke(copy_gm_to_cbuf_v2_stub));

    __cbuf__ void *dst = reinterpret_cast<__cbuf__ void *>(11);
    __gm__ void *src = reinterpret_cast<__gm__ void *>(22);

    uint16_t n_burst = static_cast<uint32_t>(33);
    uint16_t len_burst = static_cast<uint32_t>(44);
    uint16_t pad_func_mode = static_cast<uint8_t>(55);
    uint16_t src_stride = static_cast<uint64_t>(66);
    uint16_t dst_stride = static_cast<uint64_t>(77);

    asc_copy_gm2l1_sync(dst, src, n_burst, len_burst, pad_func_mode, src_stride, dst_stride);
    GlobalMockObject::verify();
}