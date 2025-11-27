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
#include "c_api/c_api_interf_util.h"

#define TEST_VECTOR_COMPUTE_REPATE_REDUCE_INSTR(class_name, c_api_name, cce_name, data_type)    \
                                                                                                \
class TestVectorCompute##class_name##data_type : public testing::Test {                         \
protected:                                                                                      \
    void SetUp() {}                                                                             \
    void TearDown() {}                                                                          \
};                                                                                              \
                                                                                                \
namespace {                                                                                     \
                                                                                                \
void cce_name##_##data_type##_uint8_t_uint16_t_uint16_t_uint16_t_Stub(__ubuf__ data_type *dst,  \
                __ubuf__ data_type *src, uint8_t repeat, uint16_t dst_repeat_stride,            \
                uint16_t src0_stride, uint16_t src1_stride, order_t order)                      \
{                                                                                               \
    EXPECT_EQ(dst, reinterpret_cast<__ubuf__ data_type *>(11));                                 \
    EXPECT_EQ(src, reinterpret_cast<__ubuf__ data_type *>(22));                                 \
    EXPECT_EQ(repeat, static_cast<uint8_t>(1));                                                 \
    EXPECT_EQ(dst_repeat_stride, static_cast<uint16_t>(8));                                     \
    EXPECT_EQ(src0_stride, static_cast<uint16_t>(1));                                           \
    EXPECT_EQ(src1_stride, static_cast<uint16_t>(8));                                           \
    EXPECT_EQ(order, order_t::ONLY_VALUE);                                                      \
}                                                                                               \
                                                                                                \
void cce_name##_##data_type##_##data_type##_uint64_t_Stub(__ubuf__ data_type *dst,              \
                __ubuf__ data_type *src, uint8_t repeat,                                        \
                uint16_t dst_repeat_stride, uint16_t src0_stride, uint16_t src1_stride,         \
                order_t order)                                                                  \
{                                                                                               \
    EXPECT_EQ(dst, reinterpret_cast<__ubuf__ data_type *>(11));                                 \
    EXPECT_EQ(src, reinterpret_cast<__ubuf__ data_type *>(22));                                 \
    EXPECT_EQ(order, order_t::ONLY_VALUE);                                                      \
}                                                                                               \
                                                                                                \
void cce_name##_##data_type##_set_vector_mask_Stub(uint64_t mask1, uint64_t mask0)              \
{                                                                                               \
    EXPECT_EQ(mask1, static_cast<uint64_t>(0));                                                 \
    EXPECT_EQ(mask0, static_cast<uint64_t>(44));                                                \
}                                                                                               \
                                                                                                \
}                                                                                               \
                                                                                                \
TEST_F(TestVectorCompute##class_name##data_type, c_api_name##_half_half_half_ReduceConfig_Succ) \
{                                                                                               \
    __ubuf__ data_type *dst =  reinterpret_cast<__ubuf__ data_type *>(11);                      \
    __ubuf__ data_type *src =  reinterpret_cast<__ubuf__ data_type *>(22);                      \
                                                                                                \
    asc_repeat_reduce_config config;                                                            \
    config.dst_repeat_stride = static_cast<uint64_t>(8);                                        \
    config.src_block_stride = static_cast<uint64_t>(1);                                         \
    config.src_repeat_stride = static_cast<uint64_t>(8);                                        \
    config.repeat = static_cast<uint64_t>(1);                                                   \
    order_t order = order_t::ONLY_VALUE;                                                        \
                                                                                                \
    MOCKER_CPP(cce_name, void(__ubuf__ data_type *, __ubuf__ data_type *,                       \
                uint8_t, uint16_t, uint16_t, uint16_t, order_t))                                \
            .times(1)                                                                           \
            .will(invoke(cce_name##_##data_type##_uint8_t_uint16_t_uint16_t_uint16_t_Stub));    \
                                                                                                \
    c_api_name(dst, src, config, order);                                                        \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \
                                                                                                \
TEST_F(TestVectorCompute##class_name##data_type, c_api_name##_half_half_half_int32_t_Succ)      \
{                                                                                               \
    __ubuf__ data_type *dst =  reinterpret_cast<__ubuf__ data_type *>(11);                      \
    __ubuf__ data_type *src =  reinterpret_cast<__ubuf__ data_type *>(22);                      \
    uint32_t count = static_cast<uint32_t>(44);                                                 \
    order_t order = order_t::ONLY_VALUE;                                                        \
    MOCKER_CPP(set_vector_mask, void(uint64_t, uint64_t))                                       \
                .times(1)                                                                       \
                .will(invoke(cce_name##_##data_type##_set_vector_mask_Stub));                   \
                                                                                                \
    MOCKER_CPP(cce_name, void(__ubuf__ data_type *, __ubuf__ data_type *,                       \
                uint8_t, uint16_t, uint16_t, uint16_t, order_t))                                \
            .times(1)                                                                           \
            .will(invoke(cce_name##_##data_type##_##data_type##_uint64_t_Stub));                \
                                                                                                \
    c_api_name(dst, src, count, order);                                                         \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \
                                                                                                \
TEST_F(TestVectorCompute##class_name##data_type, c_api_name##_sync_half_half_half_int32_t_Succ) \
{                                                                                               \
    __ubuf__ data_type *dst =  reinterpret_cast<__ubuf__ data_type *>(11);                      \
    __ubuf__ data_type *src =  reinterpret_cast<__ubuf__ data_type *>(22);                      \
    uint32_t count = static_cast<uint32_t>(44);                                                 \
    order_t order = order_t::ONLY_VALUE;                                                        \
    MOCKER_CPP(set_vector_mask, void(uint64_t, uint64_t))                                       \
                .times(1)                                                                       \
                .will(invoke(cce_name##_##data_type##_set_vector_mask_Stub));                   \
                                                                                                \
    MOCKER_CPP(cce_name, void(__ubuf__ data_type *, __ubuf__ data_type *,                       \
                uint8_t, uint16_t, uint16_t, uint16_t, order_t))                                \
            .times(1)                                                                           \
            .will(invoke(cce_name##_##data_type##_##data_type##_uint64_t_Stub));                \
    c_api_name##_sync(dst, src, count, order);                                                  \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \

// ==========asc_repeat_reduce(half/float)==========
TEST_VECTOR_COMPUTE_REPATE_REDUCE_INSTR(RepeatBlockReduceMax, asc_repeat_reduce_max, vcmax, half);
TEST_VECTOR_COMPUTE_REPATE_REDUCE_INSTR(RepeatBlockReduceMax, asc_repeat_reduce_max, vcmax, float);

TEST_VECTOR_COMPUTE_REPATE_REDUCE_INSTR(RepeatBlockReduceMin, asc_repeat_reduce_min, vcmin, half);
TEST_VECTOR_COMPUTE_REPATE_REDUCE_INSTR(RepeatBlockReduceMin, asc_repeat_reduce_min, vcmin, float);