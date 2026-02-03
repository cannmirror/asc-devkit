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
#include <mockcpp/mockcpp.hpp>
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

#define TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF(class_name, c_api_name, cce_name)            \
                                                                                                \
class TestCubeDmamove##class_name : public testing::Test {                                   \
protected:                                                                                      \
    void SetUp() {                                                                              \
        g_c_api_core_type = C_API_AIC_TYPE;                                                     \
    }                                                                                           \
    void TearDown() {                                                                           \
        g_c_api_core_type = C_API_AIV_TYPE;                                                     \
    }                                                                                           \
};                                                                                              \
                                                                                                \
namespace {                                                                                     \
                                                                                                \
void cce_name##_uint64_t_Stub(__cbuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t n_burst, \
                uint16_t burst_len, uint16_t src_stride, uint16_t dst_stride, pad_t pad_mode)   \
{                                                                                               \
    EXPECT_EQ(dst, reinterpret_cast<__cbuf__ void *>(11));                                        \
    EXPECT_EQ(src, reinterpret_cast<__gm__ void *>(22));                                      \
    EXPECT_EQ(sid, static_cast<uint8_t>(0));                                                    \
    EXPECT_EQ(n_burst, static_cast<uint16_t>(1));                                               \
    EXPECT_EQ(burst_len, static_cast<uint16_t>(1));                                             \
    EXPECT_EQ(src_stride, static_cast<uint16_t>(0));                                               \
    EXPECT_EQ(dst_stride, static_cast<uint16_t>(0));                                               \
}                                                                                               \
                                                                                                \
}                                                                                               \
                                                                                                \
TEST_F(TestCubeDmamove##class_name, c_api_name##_CopyConfig_Succ)                               \
{                                                                                               \
    __cbuf__ void *dst = reinterpret_cast<__cbuf__ void *>(11);                                  \
    __gm__ void *src = reinterpret_cast<__gm__ void *>(22);                                     \
                                                                                                \
    uint16_t n_burst = static_cast<uint16_t>(1);                                                  \
    uint16_t burst_len = static_cast<uint16_t>(1);                                                \
    uint16_t src_stride = static_cast<uint16_t>(0);                                              \
    uint16_t dst_stride = static_cast<uint16_t>(0);                                             \
                                                                                                \
    MOCKER_CPP(cce_name, void(__cbuf__ void *, __gm__ void *,                                   \
                uint8_t, uint16_t, uint16_t, uint16_t, uint16_t, pad_t))                        \
            .times(1)                                                                           \
            .will(invoke(cce_name##_uint64_t_Stub));                                              \
                                                                                                \
    c_api_name(dst, src, n_burst, burst_len, src_stride, dst_stride, (pad_t)0);             \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \
                                                                                                \
TEST_F(TestCubeDmamove##class_name, c_api_name##_size_Succ)                                     \
{                                                                                               \
    __cbuf__ void *dst = reinterpret_cast<__cbuf__ void *>(11);                                 \
    __gm__ void *src = reinterpret_cast<__gm__ void *>(22);                                     \
    uint32_t size = static_cast<uint32_t>(44);                                                  \
                                                                                                \
    MOCKER_CPP(cce_name, void(__cbuf__ void *, __gm__ void *,                                   \
            uint8_t, uint16_t, uint16_t, uint16_t, uint16_t, pad_t))                             \
            .times(1)                                                                           \
            .will(invoke(cce_name##_uint64_t_Stub));                                             \
                                                                                                \
    c_api_name(dst, src, size);                                                                 \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \
                                                                                                \
TEST_F(TestCubeDmamove##class_name, c_api_name##_sync_Succ)                                     \
{                                                                                               \
    __cbuf__ void *dst = reinterpret_cast<__cbuf__ void *>(11);                                 \
    __gm__ void *src = reinterpret_cast<__gm__ void *>(22);                                     \
    uint32_t size = static_cast<uint32_t>(44);                                                  \
                                                                                                \
    MOCKER_CPP(cce_name, void(__cbuf__ void *, __gm__ void *,                                   \
            uint8_t, uint16_t, uint16_t, uint16_t, uint16_t, pad_t))                            \
            .times(1)                                                                           \
            .will(invoke(cce_name##_uint64_t_Stub));                                            \
    c_api_name##_sync(dst, src, size);                                                          \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \

// ==========asc_copy_gm2l1==========
TEST_CUBE_DMAMOVE_COPY_GM_TO_CBUF(CopyGM2L1, asc_copy_gm2l1, copy_gm_to_cbuf);