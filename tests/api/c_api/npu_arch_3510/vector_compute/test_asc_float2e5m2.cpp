 /**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "tests/api/c_api/npu_arch_3510/utils/test_vcvt_instr_utils.h"

//================asc_float2e5m2_rd================
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rd, asc_float2e5m2_rd, vcvt, vector_f8e5m2, vector_float, 0);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rd_v2, asc_float2e5m2_rd_v2, vcvt, vector_f8e5m2, vector_float, 1);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rd_v3, asc_float2e5m2_rd_v3, vcvt, vector_f8e5m2, vector_float, 2);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rd_v4, asc_float2e5m2_rd_v4, vcvt, vector_f8e5m2, vector_float, 3);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rd_sat, asc_float2e5m2_rd_sat, vcvt, vector_f8e5m2, vector_float, 4);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rd_sat_v2, asc_float2e5m2_rd_sat_v2, vcvt, vector_f8e5m2, vector_float, 5);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rd_sat_v3, asc_float2e5m2_rd_sat_v3, vcvt, vector_f8e5m2, vector_float, 6);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rd_sat_v4, asc_float2e5m2_rd_sat_v4, vcvt, vector_f8e5m2, vector_float, 7);

//================asc_float2e5m2_rn================
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn, asc_float2e5m2_rn, vcvt, vector_f8e5m2, vector_float, 8);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_v2, asc_float2e5m2_rn_v2, vcvt, vector_f8e5m2, vector_float, 9);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_v3, asc_float2e5m2_rn_v3, vcvt, vector_f8e5m2, vector_float, 10);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_v4, asc_float2e5m2_rn_v4, vcvt, vector_f8e5m2, vector_float, 11);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_sat, asc_float2e5m2_rn_sat, vcvt, vector_f8e5m2, vector_float, 12);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_sat_v2, asc_float2e5m2_rn_sat_v2, vcvt, vector_f8e5m2, vector_float, 13);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_sat_v3, asc_float2e5m2_rn_sat_v3, vcvt, vector_f8e5m2, vector_float, 14);
TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rn_sat_v4, asc_float2e5m2_rn_sat_v4, vcvt, vector_f8e5m2, vector_float, 15);

//================asc_float2e5m2_rna================
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rna, asc_float2e5m2_rna, vcvt, vector_f8e5m2, vector_float, 16);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rna_v2, asc_float2e5m2_rna_v2, vcvt, vector_f8e5m2, vector_float, 17);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rna_v3, asc_float2e5m2_rna_v3, vcvt, vector_f8e5m2, vector_float, 18);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rna_v4, asc_float2e5m2_rna_v4, vcvt, vector_f8e5m2, vector_float, 19);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rna_sat, asc_float2e5m2_rna_sat, vcvt, vector_f8e5m2, vector_float, 20);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rna_sat_v2, asc_float2e5m2_rna_sat_v2, vcvt, vector_f8e5m2, vector_float, 21);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rna_sat_v3, asc_float2e5m2_rna_sat_v3, vcvt, vector_f8e5m2, vector_float, 22);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rna_sat_v4, asc_float2e5m2_rna_sat_v4, vcvt, vector_f8e5m2, vector_float, 23);

//================asc_float2e5m2_ru================
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_ru, asc_float2e5m2_ru, vcvt, vector_f8e5m2, vector_float, 24);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_ru_v2, asc_float2e5m2_ru_v2, vcvt, vector_f8e5m2, vector_float, 25);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_ru_v3, asc_float2e5m2_ru_v3, vcvt, vector_f8e5m2, vector_float, 26);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_ru_v4, asc_float2e5m2_ru_v4, vcvt, vector_f8e5m2, vector_float, 27);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_ru_sat, asc_float2e5m2_ru_sat, vcvt, vector_f8e5m2, vector_float, 28);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_ru_sat_v2, asc_float2e5m2_ru_sat_v2, vcvt, vector_f8e5m2, vector_float, 29);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_ru_sat_v3, asc_float2e5m2_ru_sat_v3, vcvt, vector_f8e5m2, vector_float, 30);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_ru_sat_v4, asc_float2e5m2_ru_sat_v4, vcvt, vector_f8e5m2, vector_float, 31);

//================asc_float2e5m2_rz================
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rz, asc_float2e5m2_rz, vcvt, vector_f8e5m2, vector_float, 32);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rz_v2, asc_float2e5m2_rz_v2, vcvt, vector_f8e5m2, vector_float, 33);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rz_v3, asc_float2e5m2_rz_v3, vcvt, vector_f8e5m2, vector_float, 34);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rz_v4, asc_float2e5m2_rz_v4, vcvt, vector_f8e5m2, vector_float, 35);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rz_sat, asc_float2e5m2_rz_sat, vcvt, vector_f8e5m2, vector_float, 36);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rz_sat_v2, asc_float2e5m2_rz_sat_v2, vcvt, vector_f8e5m2, vector_float, 37);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rz_sat_v3, asc_float2e5m2_rz_sat_v3, vcvt, vector_f8e5m2, vector_float, 38);
// TEST_VECTOR_COMPUTE_VCVT_INSTR_1(Vcvt_rz_sat_v4, asc_float2e5m2_rz_sat_v4, vcvt, vector_f8e5m2, vector_float, 39);