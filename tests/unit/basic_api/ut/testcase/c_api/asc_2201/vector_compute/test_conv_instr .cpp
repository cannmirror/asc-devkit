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

#define TEST_VECTOR_COMPUTE_CONV_INSTR(class_name, c_api_name, cce_name, dst_data_type, src_data_type)       \
                                                                                                \
class TestVectorCompute##class_name##_##dst_data_type##_##src_data_type : public testing::Test {\
protected:                                                                                      \
    void SetUp() {}                                                                             \
    void TearDown() {}                                                                          \
};                                                                                              \
                                                                                                \
namespace {                                                                                     \
                                                                                                \
void cce_name##_##dst_data_type##_##src_data_type##_UnaryCfg_Stub(__ubuf__ dst_data_type *dst, \
                __ubuf__ src_data_type *src, uint8_t repeat,                                    \
                uint16_t dst_block_stride, uint16_t src_block_stride,                           \
                uint16_t dst_repeat_stride, uint16_t src_repeat_stride)                         \  
{                                                                                               \
    EXPECT_EQ(dst, reinterpret_cast<__ubuf__ dst_data_type *>(11));                             \
    EXPECT_EQ(src, reinterpret_cast<__ubuf__ src_data_type *>(22));                             \
    EXPECT_EQ(repeat, static_cast<uint8_t>(1));                                                 \
    EXPECT_EQ(dst_block_stride, static_cast<uint16_t>(1));                                      \
    EXPECT_EQ(src_block_stride, static_cast<uint16_t>(1));                                      \
    EXPECT_EQ(dst_repeat_stride, static_cast<uint16_t>(8));                                     \
    EXPECT_EQ(src_repeat_stride, static_cast<uint16_t>(8));                                     \
}                                                                                               \
                                                                                                \
void  cce_name##_##dst_data_type##_##src_data_type##_uint32_t_Stub(__ubuf__ dst_data_type *dst, \
                __ubuf__ src_data_type *src, uint8_t repeat,                                    \
                uint16_t dst_block_stride, uint16_t src_block_stride,                           \
                uint16_t dst_repeat_stride, uint16_t src_repeat_stride)                         \
{                                                                                               \
    EXPECT_EQ(dst, reinterpret_cast<__ubuf__ dst_data_type *>(11));                             \
    EXPECT_EQ(src, reinterpret_cast<__ubuf__ src_data_type *>(22));                             \
}                                                                                               \
                                                                                                \
void cce_name##_##dst_data_type##_##src_data_type##_set_vector_mask_Stub(uint64_t mask1, uint64_t mask0)              \
{                                                                                               \
    EXPECT_EQ(mask1, static_cast<uint64_t>(0));                                                 \
    EXPECT_EQ(mask0, static_cast<uint64_t>(44));                                                \
}                                                                                               \
                                                                                                \
}                                                                                               \
                                                                                                \
TEST_F(TestVectorCompute##class_name##_##dst_data_type##_##src_data_type, c_api_name##_##dst_data_type##_##src_data_type##_UnaryConfig_Succ)  \
{                                                                                               \
    __ubuf__ dst_data_type *dst = reinterpret_cast<__ubuf__ dst_data_type *>(11);               \
    __ubuf__ src_data_type *src = reinterpret_cast<__ubuf__ src_data_type *>(22);               \
                                                                                                \
    asc_unary_config config;                                                                    \
    config.dst_block_stride = static_cast<uint64_t>(1);                                         \
    config.src_block_stride = static_cast<uint64_t>(1);                                         \
    config.dst_repeat_stride = static_cast<uint64_t>(8);                                        \
    config.src_repeat_stride = static_cast<uint64_t>(8);                                        \
    config.repeat = static_cast<uint64_t>(1);                                                   \
                                                                                                \
    MOCKER_CPP(cce_name, void(__ubuf__ dst_data_type *, __ubuf__ src_data_type *,               \
                uint8_t, uint16_t, uint16_t, uint16_t, uint16_t))                               \
            .times(1)                                                                           \
            .will(invoke(cce_name##_##dst_data_type##_##src_data_type##_UnaryCfg_Stub));        \
                                                                                                \
    c_api_name(dst, src, config);                                                               \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \
                                                                                                \
TEST_F(TestVectorCompute##class_name##_##dst_data_type##_##src_data_type, c_api_name##_##dst_data_type##_##src_data_type##_uint32_t_Succ)      \
{                                                                                               \
    __ubuf__ dst_data_type *dst = reinterpret_cast<__ubuf__ dst_data_type *>(11);               \
    __ubuf__ src_data_type *src = reinterpret_cast<__ubuf__ src_data_type *>(22);               \
    uint32_t count = static_cast<uint32_t>(44);                                                 \
    MOCKER_CPP(set_vector_mask, void(uint64_t, uint64_t))                                       \
            .times(1)                                                                           \
            .will(invoke(cce_name##_##dst_data_type##_##src_data_type##_set_vector_mask_Stub)); \
                                                                                                \
    MOCKER_CPP(cce_name, void(__ubuf__ dst_data_type *,__ubuf__ src_data_type *,                \
                uint8_t, uint16_t, uint16_t, uint16_t, uint16_t))                               \
            .times(1)                                                                           \
            .will(invoke(cce_name##_##dst_data_type##_##src_data_type##_uint32_t_Stub));        \
                                                                                                \
    c_api_name(dst, src, count);                                                                \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \
                                                                                                \
TEST_F(TestVectorCompute##class_name##_##dst_data_type##_##src_data_type, c_api_name##_sync_##dst_data_type##_##src_data_type##_uint32_t_Succ) \
{                                                                                               \
    __ubuf__ dst_data_type *dst = reinterpret_cast<__ubuf__ dst_data_type *>(11);               \
    __ubuf__ src_data_type *src = reinterpret_cast<__ubuf__ src_data_type *>(22);               \
    uint32_t count = static_cast<uint32_t>(44);                                                 \
    MOCKER_CPP(set_vector_mask, void(uint64_t, uint64_t))                                       \
            .times(1)                                                                           \
            .will(invoke(cce_name##_##dst_data_type##_##src_data_type##_set_vector_mask_Stub)); \
                                                                                                \
    MOCKER_CPP(cce_name, void(__ubuf__ dst_data_type *, __ubuf__ src_data_type *,               \
                uint8_t, uint16_t, uint16_t, uint16_t, uint16_t))                               \
            .times(1)                                                                           \
            .will(invoke(cce_name##_##dst_data_type##_##src_data_type##_uint32_t_Stub));        \
    c_api_name##_sync(dst, src, count);                                                         \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \

// ==========asc_bf162float==========
// ==========asc_bf162int32(a/c/f/r/z)==========
// ==========asc_float2bf16(r/a/f/c/z)==========
// ==========asc_float2float(r/f/c/a/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Fr, asc_float2float_r, vconv_f322f32r, float, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Ff, asc_float2float_f, vconv_f322f32f, float, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Fc, asc_float2float_c, vconv_f322f32c, float, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Fa, asc_float2float_a, vconv_f322f32a, float, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Fz, asc_float2float_z, vconv_f322f32z, float, float);
// ==========asc_float2half(NA/r/a/f/c/z/o)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2H, asc_float2half, vconv_f322f16, half, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Hr, asc_float2half_r, vconv_f322f16r, half, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Hf, asc_float2half_f, vconv_f322f16f, half, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Hc, asc_float2half_c, vconv_f322f16c, half, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Hz, asc_float2half_z, vconv_f322f16z, half, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Ho, asc_float2half_o, vconv_f322f16o, half, float);
// ==========asc_half2float(NA)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2F, asc_half2float, vconv_f162f32, float, half);
// ==========asc_half2int4(NA/a/c/f/r/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I4, asc_half2int4, vconv_f162s4, void, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I4a, asc_half2int4_a, vconv_f162s4a, void, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I4c, asc_half2int4_c, vconv_f162s4c, void, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I4f, asc_half2int4_f, vconv_f162s4f, void, half);
// ==========asc_half2int16(a/c/f/r/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I16a, asc_half2int16_a, vconv_f162s16a, int16_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I16c, asc_half2int16_c, vconv_f162s16c, int16_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I16f, asc_half2int16_f, vconv_f162s16f, int16_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I16r, asc_half2int16_r, vconv_f162s16r, int16_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I16z, asc_half2int16_z, vconv_f162s16z, int16_t, half);
// ==========asc_half2int32(a/c/f/r/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I32a, asc_half2int32_a, vconv_f162s32a, int32_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I32c, asc_half2int32_c, vconv_f162s32c, int32_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I32f, asc_half2int32_f, vconv_f162s32f, int32_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I32r, asc_half2int32_r, vconv_f162s32r, int32_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I32z, asc_half2int32_z, vconv_f162s32z, int32_t, half);