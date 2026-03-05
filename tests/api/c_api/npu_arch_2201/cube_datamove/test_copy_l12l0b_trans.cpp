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

#define TEST_CUBE_DATAMOVE_UNARY_SCALAR_INSTR(class_name, c_api_name, cce_name, data_type)     \
                                                                                                \
class TestCubeDatamove##class_name##data_type : public testing::Test {                         \
protected:                                                                                      \
    void SetUp() { \
        g_coreType = C_API_AIC_TYPE; \
    }                                                                             \
    void TearDown() { g_coreType = C_API_AIV_TYPE;}                                                                          \
};                                                                                              \
                                                                                                \
namespace {                                                                                     \
                                                                                                \
void cce_name##_##data_type##_uint16_t_uint8_t_uint16_t_uint16_t_bool_uint16_t_Stub(__cb__ data_type *dst,   \
                __cbuf__ data_type *src,  uint16_t index_id, uint8_t repeat,                     \
                uint16_t src_stride, uint16_t dst_stride, bool addrmode,                        \
                uint16_t frac_stride)                                                           \
{                                                                                               \
    EXPECT_EQ(dst, reinterpret_cast<__cb__ data_type *>(11));                                 \
    EXPECT_EQ(src, reinterpret_cast<__cbuf__ data_type *>(22));                                 \
    EXPECT_EQ(repeat, static_cast<uint8_t>(1));                                                 \
    EXPECT_EQ(index_id, static_cast<uint16_t>(1));                                            \
    EXPECT_EQ(src_stride, static_cast<uint16_t>(1));                                            \
    EXPECT_EQ(dst_stride, static_cast<uint16_t>(8));                                           \
    EXPECT_EQ(frac_stride, static_cast<uint16_t>(8));                                               \
    EXPECT_EQ(addrmode, false);                                                                 \
}                                                                                               \
                                                                                                \
                                                                                                \
}                                                                                               \
                                                                                                \
TEST_F(TestCubeDatamove##class_name##data_type, c_api_name##_data_type_data_type_Succ)  \
{                                                                                               \
    __cb__ data_type *dst = reinterpret_cast<__cb__ data_type *>(11);                           \
    __cbuf__ data_type *src = reinterpret_cast<__cbuf__ data_type *>(22);                       \
                                                                                                \
    uint16_t index_id = 1;                                                              \
    uint16_t src_stride = 1;                                                              \
    uint16_t dst_stride = 8;                                                             \
    uint16_t frac_stride = 8;                                                             \
    uint8_t repeat = 1;                                                                         \
    bool addrmode = false;                                                                         \
                                                                                                \
    MOCKER_CPP(cce_name, void(__cb__ data_type *,__cbuf__ data_type *,                          \
                uint16_t, uint8_t, uint16_t, uint16_t, bool, uint16_t))                               \
            .times(1)                                                                           \
            .will(invoke(cce_name##_##data_type##_uint16_t_uint8_t_uint16_t_uint16_t_bool_uint16_t_Stub));   \
                                                                                                \
    c_api_name(dst, src, index_id, repeat, src_stride, dst_stride, addrmode, frac_stride);       \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \

// ==========asc_copy_l12l0b_trans(half/float/int32_t/int8_t/uint32_t/uint8_t)==========
TEST_CUBE_DATAMOVE_UNARY_SCALAR_INSTR(CopyL12l0bTrans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, half);
TEST_CUBE_DATAMOVE_UNARY_SCALAR_INSTR(CopyL12l0bTrans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, float);
TEST_CUBE_DATAMOVE_UNARY_SCALAR_INSTR(CopyL12l0bTrans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, int32_t);
TEST_CUBE_DATAMOVE_UNARY_SCALAR_INSTR(CopyL12l0bTrans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, int8_t);
TEST_CUBE_DATAMOVE_UNARY_SCALAR_INSTR(CopyL12l0bTrans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, uint32_t);
TEST_CUBE_DATAMOVE_UNARY_SCALAR_INSTR(CopyL12l0bTrans, asc_copy_l12l0b_trans, load_cbuf_to_cb_transpose, uint8_t);
