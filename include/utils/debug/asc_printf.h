/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file asc_printf.h
 * \brief
 */

#ifndef INCLUDE_UTILS_DEBUG_ASC_PRINTF_H
#define INCLUDE_UTILS_DEBUG_ASC_PRINTF_H

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_PRINTF_H__
#endif

#include "simt_api/device_types.h"

#ifndef __CHECK_FEATURE_AT_PRECOMPILE

namespace __asc_simt_vf {
template <class... Args>
#ifndef __NPU_COMPILER_INTERNAL_PURE_SIMT__
__attribute__((always_inline)) inline __SIMT_DEVICE_FUNCTIONS_DECL__ void printf(const __gm__ char* fmt, Args&&... args);
#else
__attribute__((always_inline)) inline __SIMT_DEVICE_FUNCTIONS_DECL__ void printf(const char* fmt, Args&&... args);
#endif
}   // namespase __asc_simt_vf

#endif

#include "impl/utils/debug/asc_printf_simt_impl.h"

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_PRINTF_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_PRINTF_H__
#endif

#endif