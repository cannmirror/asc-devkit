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
#include "c_api/utils_intf.h"

#define TEST_VECTOR_COMPUTE_RELU_CONV_INSTR(class_name, c_api_name, cce_name, dst_data_type, src_data_type)          \
                                                                                               \
class TestVectorCompute##class_name##src_data_type : public testing::Test {                        \
protected:                                                                                     \
    void SetUp() {}                                                                            \
    void TearDown() {}                                                                         \
};                                                                                             \
                                                                                               \
namespace {                                                                                    \
                                                                                               \
void cce_name##_##dst_data_type##_##src_data_type##_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_bool_Stub(__ubuf__ dst_data_type *dst,   \
                __ubuf__ src_data_type *src0, __ubuf__ src_data_type *src1, uint8_t repeat,                                   \
                uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride,                       \
                uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride, bool h) \
{                                                                                               \
    EXPECT_EQ(dst, reinterpret_cast<__ubuf__ dst_data_type *>(11));                                 \
    EXPECT_EQ(src0, reinterpret_cast<__ubuf__ src_data_type *>(22));                                \
    EXPECT_EQ(src1, reinterpret_cast<__ubuf__ src_data_type *>(33));                                \
    EXPECT_EQ(repeat, static_cast<uint8_t>(1));                                                 \
    EXPECT_EQ(dst_block_stride, static_cast<uint8_t>(1));                                       \
    EXPECT_EQ(src0_block_stride, static_cast<uint8_t>(1));                                      \
    EXPECT_EQ(src1_block_stride, static_cast<uint8_t>(1));                                      \
    EXPECT_EQ(dst_repeat_stride, static_cast<uint8_t>(8));                                      \
    EXPECT_EQ(src0_repeat_stride, static_cast<uint8_t>(8));                                     \
    EXPECT_EQ(src1_repeat_stride, static_cast<uint8_t>(8));                                     \
    EXPECT_EQ(h, false);                                                               \
}                                                                                               \
                                                                                                \
void  cce_name##_##dst_data_type##_##src_data_type##_uint64_t_bool_Stub(__ubuf__ dst_data_type *dst,   \
                __ubuf__ src_data_type *src0, __ubuf__ src_data_type *src1, uint8_t repeat,                 \
                uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride,     \
                uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride, bool h)  \
{                                                                                               \
    EXPECT_EQ(dst, reinterpret_cast<__ubuf__ dst_data_type *>(11));                                 \
    EXPECT_EQ(src0, reinterpret_cast<__ubuf__ src_data_type *>(22));                                \
    EXPECT_EQ(src1, reinterpret_cast<__ubuf__ src_data_type *>(33));                                \
}                                                                                               \
                                                                                                \
void cce_name##_set_vector_mask_Stub(uint64_t mask1, uint64_t mask0)              \
{                                                                                               \
    EXPECT_EQ(mask1, static_cast<uint64_t>(0));                                                 \
    EXPECT_EQ(mask0, static_cast<uint64_t>(44));                                                \
}                                                                                               \
                                                                                                \
}                                                                                               \
                                                                                                \
TEST_F(TestVectorCompute##class_name##src_data_type, c_api_name##_half_half_half_BinaryConfig_Succ) \
{                                                                                               \
    __ubuf__ dst_data_type *dst = reinterpret_cast<__ubuf__ dst_data_type *>(11);                       \
    __ubuf__ src_data_type *src0 = reinterpret_cast<__ubuf__ src_data_type *>(22);                      \
    __ubuf__ src_data_type *src1 = reinterpret_cast<__ubuf__ src_data_type *>(33);                      \
                                                                                                        \
    uint8_t dst_block_stride = 1;                                         \
    uint8_t src0_block_stride = 1;                                        \
    uint8_t src1_block_stride = 1;                                        \
    uint8_t dst_repeat_stride = 8;                                        \
    uint8_t src0_repeat_stride = 8;                                       \
    uint8_t src1_repeat_stride = 8;                                       \
    uint8_t repeat = 1;                                                   \
                                                                                                \
    MOCKER_CPP(cce_name, void(__ubuf__ dst_data_type *,__ubuf__ src_data_type *,                        \
                __ubuf__ src_data_type *, uint8_t, uint8_t, uint8_t, uint8_t,                       \
                uint8_t, uint8_t, uint8_t, bool))                                                     \
            .times(1)                                                                           \
            .will(invoke(cce_name##_##dst_data_type##_##src_data_type##_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_uint8_t_bool_Stub));   \
                                                                                                \
    c_api_name(dst, src0, src1, repeat, dst_block_stride, src0_block_stride, src1_block_stride, \
                dst_repeat_stride, src0_repeat_stride, src1_repeat_stride);                                                        \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \
                                                                                                \
TEST_F(TestVectorCompute##class_name##src_data_type, c_api_name##_half_half_half_int32_t_Succ)      \
{                                                                                               \
    __ubuf__ dst_data_type *dst = reinterpret_cast<__ubuf__ dst_data_type *>(11);                       \
    __ubuf__ src_data_type *src0 = reinterpret_cast<__ubuf__ src_data_type *>(22);                      \
    __ubuf__ src_data_type *src1 = reinterpret_cast<__ubuf__ src_data_type *>(33);                      \
    uint32_t count = static_cast<uint32_t>(44);                                                 \
    MOCKER_CPP(set_vector_mask, void(uint64_t, uint64_t))                                       \
            .times(1)                                                                           \
            .will(invoke(cce_name##_set_vector_mask_Stub));                                    \
                                                                                                \
    MOCKER_CPP(cce_name, void(__ubuf__ dst_data_type *,__ubuf__ src_data_type *,                        \
                __ubuf__ src_data_type *, uint8_t, uint8_t, uint8_t, uint8_t,                       \
                uint8_t, uint8_t, uint8_t, bool))                                                     \
            .times(1)                                                                           \
            .will(invoke(cce_name##_##dst_data_type##_##src_data_type##_uint64_t_bool_Stub));  \
                                                                                                \
    c_api_name(dst, src0, src1, count);                                                         \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \
                                                                                                \
TEST_F(TestVectorCompute##class_name##src_data_type, c_api_name##_sync_half_half_half_int32_t_Succ) \
{                                                                                               \
    __ubuf__ dst_data_type *dst = reinterpret_cast<__ubuf__ dst_data_type *>(11);                       \
    __ubuf__ src_data_type *src0 = reinterpret_cast<__ubuf__ src_data_type *>(22);                      \
    __ubuf__ src_data_type *src1 = reinterpret_cast<__ubuf__ src_data_type *>(33);                      \
    uint32_t count = static_cast<uint32_t>(44);                                                 \
    MOCKER_CPP(set_vector_mask, void(uint64_t, uint64_t))                                       \
            .times(1)                                                                           \
            .will(invoke(cce_name##_set_vector_mask_Stub));                       \
                                                                                                \
    MOCKER_CPP(cce_name, void(__ubuf__ dst_data_type *,__ubuf__ src_data_type *,                        \
                __ubuf__ src_data_type *, uint8_t, uint8_t, uint8_t, uint8_t,                       \
                uint8_t, uint8_t, uint8_t, bool))                                                     \
            .times(1)                                                                           \
            .will(invoke(cce_name##_##dst_data_type##_##src_data_type##_uint64_t_bool_Stub));  \
    c_api_name##_sync(dst, src0, src1, count);                                                  \
    GlobalMockObject::verify();                                                                 \
}

// ==========asc_add_relu_cast_f162s8==========
TEST_VECTOR_COMPUTE_RELU_CONV_INSTR(AddReluF162S8, asc_add_relu, vaddreluconv_f162s8, int8_t, half);
// ==========asc_add_relu_cast_f322f16==========
TEST_VECTOR_COMPUTE_RELU_CONV_INSTR(AddReluF322F16, asc_add_relu, vaddreluconv_f322f16, half, float);
// ==========asc_add_relu_cast_s162s8==========
TEST_VECTOR_COMPUTE_RELU_CONV_INSTR(AddReluS162S8, asc_add_relu, vaddreluconv_s162s8, int8_t, int16_t);

// ==========asc_sub_relu_cast_f162s8==========
TEST_VECTOR_COMPUTE_RELU_CONV_INSTR(SubReluF162S8, asc_sub_relu, vsubreluconv_f162s8, int8_t, half);
// ==========asc_sub_relu_cast_f322f16==========
TEST_VECTOR_COMPUTE_RELU_CONV_INSTR(SubReluF322F16, asc_sub_relu, vsubreluconv_f322f16, half, float);
// ==========asc_sub_relu_cast_s162s8==========
TEST_VECTOR_COMPUTE_RELU_CONV_INSTR(SubReluS162S8, asc_sub_relu, vsubreluconv_s162s8, int8_t, int16_t);
