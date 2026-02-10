/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file asc_utils_macros.h
 * \brief
 */
#ifndef IMPL_UTILS_DEBUG_ASC_UTILS_MACROS_H
#define IMPL_UTILS_DEBUG_ASC_UTILS_MACROS_H

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_UTILS_MACROS__
#warning "asc_utils_macros.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future."
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif // __aicore__

constexpr int32_t MIX = 0;
constexpr int32_t AIC = 1;
constexpr int32_t AIV = 2;

#if (defined(__DAV_CUBE__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3101))
#define SPLIT_CORE_CUBE
#endif

#if (defined(__DAV_VEC__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3101))
#define SPLIT_CORE_VEC
#endif

#if defined(ASCENDC_CPU_DEBUG)
extern int32_t g_matmulCount;
extern int32_t g_coreType;
#define ASCEND_IS_AIV (g_coreType == AIV)
#define ASCEND_IS_AIC (g_coreType == AIC)
#define ASCEND_IS_NOT_AIV (g_coreType != AIV)
#define ASCEND_IS_NOT_AIC (g_coreType != AIC)
#else
#if defined(SPLIT_CORE_CUBE)
constexpr int32_t g_coreType = AIC;
#elif defined(SPLIT_CORE_VEC)
constexpr int32_t g_coreType = AIV;
#else
constexpr int32_t g_coreType = MIX;
#endif
#define ASCEND_IS_AIV constexpr(g_coreType == AIV)
#define ASCEND_IS_AIC constexpr(g_coreType == AIC)
#define ASCEND_IS_NOT_AIV constexpr(g_coreType != AIV)
#define ASCEND_IS_NOT_AIC constexpr(g_coreType != AIC)
#endif

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_UTILS_MACROS__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_UTILS_MACROS__
#endif

#endif // IMPL_UTILS_DEBUG_ASC_UTILS_MACROS_H