/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "../utils/test_binary_instr_utils.h"

// ==========asc_sub_relu(half/float/int16_t)==========
TEST_VECTOR_COMPUTE_BINARY_INSTR(SubReluCAPI, asc_sub_relu, vsubrelu, half);
TEST_VECTOR_COMPUTE_BINARY_INSTR(SubReluCAPI, asc_sub_relu, vsubrelu, float);
TEST_VECTOR_COMPUTE_BINARY_INSTR(SubReluCAPI, asc_sub_relu, vsubrelu, int16_t);

// ==========vsubreluconv_vdeqs162b8==========
TEST_VECTOR_COMPUTE_RELU_CONV_INSTR(SubReluConvVdeqs162b8_int8_t, asc_sub_relu_vdeq, vsubreluconv_vdeqs162b8, int8_t, int16_t);
TEST_VECTOR_COMPUTE_RELU_CONV_INSTR(SubReluConvVdeqs162b8_uint8_t, asc_sub_relu_vdeq, vsubreluconv_vdeqs162b8, uint8_t, int16_t);