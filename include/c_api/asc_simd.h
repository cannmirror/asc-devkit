/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)  
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS  
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H  
#endif     

#ifndef INCLUDE_C_API_ASC_SIMD_H
#define INCLUDE_C_API_ASC_SIMD_H

#include "instr_impl/npu_arch_2201/utils_impl/utils_impl.h"
#include "instr_impl/npu_arch_2201/utils_impl/debug_utils.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)

#include "atomic/atomic.h"
#include "cache_ctrl/cache_ctrl.h"
#include "cube_compute/cube_compute.h"
#include "cube_datamove/cube_datamove.h"
#include "misc/misc.h"
#include "scalar_compute/scalar_compute.h"
#include "sync/sync.h"
#include "sys_var/sys_var.h"
#include "vector_compute/vector_compute.h"
#include "vector_datamove/vector_datamove.h"
#include "utils/debug/asc_dump.h"
#include "utils/debug/asc_printf.h"
#include "utils/debug/asc_assert.h"

#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
#include "vector_compute/vector_compute.h"
#include "include/c_api/reg_compute/reg_convert.h"
#include "include/c_api/reg_compute/reg_load.h"
#include "include/c_api/reg_compute/reg_store.h"
#include "include/c_api/reg_compute/reg_vector.h"

#endif

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H)  
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS  
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H  
#endif    
