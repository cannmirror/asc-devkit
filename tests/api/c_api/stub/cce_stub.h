/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __TESTS_C_API_STUB__
#define __TESTS_C_API_STUB__
#include <cstdint>
#include "stub_fun.h"

void vsts(vector_f8e4m3 data, __ubuf__ fp8_e4m3fn_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_f8e5m2 data, __ubuf__ fp8_e5m2_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_f8e8m0 data, __ubuf__ fp8_e8m0_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_f4e2m1x2 data, __ubuf__ fp4x2_e2m1_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_f4e1m2x2 data, __ubuf__ fp4x2_e1m2_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);

void vsts(vector_s8 src0, vector_s8 src1, __ubuf__ int8_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_u8 src0, vector_u8 src1, __ubuf__ uint8_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_s16 src0, vector_s16 src1, __ubuf__ int16_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_u16 src0, vector_u16 src1, __ubuf__ uint16_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_s32 src0, vector_s32 src1, __ubuf__ int32_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_u32 src0, vector_u32 src1, __ubuf__ uint32_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_f16 src0, vector_f16 src1, __ubuf__ half* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_bf16 src0, vector_bf16 src1, __ubuf__ bfloat16_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_f8e4m3 src0, vector_f8e4m3 src1, __ubuf__ fp8_e4m3fn_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_f8e5m2 src0, vector_f8e5m2 src1, __ubuf__ fp8_e5m2_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_f8e8m0 src0, vector_f8e8m0 src1, __ubuf__ fp8_e8m0_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_f4e2m1x2 src0, vector_f4e2m1x2 src1, __ubuf__ fp4x2_e2m1_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);
void vsts(vector_f4e1m2x2 src0, vector_f4e1m2x2 src1, __ubuf__ fp4x2_e1m2_t* base, int32_t offset, Literal dist, vector_bool mask, Literal mode);

#if defined(__DAV_CUBE__)
    inline int32_t g_coreType = 1;
#else
    inline int32_t g_coreType = 2;
#endif

typedef std::integral_constant<Pos, Pos::LOWEST> Lowest_Type;
typedef std::integral_constant<Pos, Pos::HIGHEST> Highest_Type;
constexpr Lowest_Type POS_LOWEST = Lowest_Type();
constexpr Highest_Type POS_HIGHEST = Highest_Type();

#endif
