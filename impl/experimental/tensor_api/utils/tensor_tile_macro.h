/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
* \file tensor_tile_macro.h
* \brief
*/
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_UTILS_TENSOR_TILE_MACRO_H
#define IMPL_EXPERIMENTAL_TENSOR_API_UTILS_TENSOR_TILE_MACRO_H

#include "impl/utils/sys_constants.h"
#include "impl/utils/sys_macros.h"

#define GM_ADDR __gm__ uint8_t*

#include <cstdint>
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#define ASSERT(x) assert(x)
#define DEBUG_CODE(T) T

#else

#ifndef ASSERT
#define ASSERT(x)
#endif
#define DEBUG_CODE(T)

#if (__CCE__)
#define _ASCENDC_HAS_BISHENG_COMPILER 1
#endif

#if (_ASCENDC_HAS_BISHENG_COMPILER)
#define ASCENDC_HOST __host__
#define ASCENDC_AICORE __aicore__
#define ASCENDC_HOST_AICORE __host_aicore__
#else
#define ASCENDC_HOST
#define ASCENDC_AICORE
#define ASCENDC_HOST_AICORE
#endif

#if __NPU_ARCH__ == 2002
#ifndef __BLOCK_LOCAL__
#define __BLOCK_LOCAL__ [[block_local]]
#endif // __BLOCK_LOCAL__
#else
#ifndef __WORKGROUP_LOCAL__
#define __WORKGROUP_LOCAL__ [[workgroup_local]]
#endif // __WORKGROUP_LOCAL__
#endif

#ifndef __BLOCK_LOCAL__
#define __BLOCK_LOCAL__ [[block_local]]
#endif // __BLOCK_LOCAL__
#endif // ASCENDC_CPU_DEBUG

#ifndef QBUF_MAX_LEN
#define QBUF_MAX_LEN 64
#endif

#ifndef QBUFPOOL_MAX_LEN
#define QBUFPOOL_MAX_LEN 16
#endif

#ifndef MAX_MSG_COUNT
#define MAX_MSG_COUNT 64
#endif

#ifndef QBUF_L0A_RESERVED_LEN
#define QBUF_L0A_RESERVED_LEN 2
#endif

#ifndef QBUF_L0B_RESERVED_LEN
#define QBUF_L0B_RESERVED_LEN 2
#endif
#ifndef QBUF_TOTAL_RESERVED_LEN
#define QBUF_TOTAL_RESERVED_LEN 4
#endif

#ifndef TPIPE_MAX_TYPE
#define TPIPE_MAX_TYPE 4
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
#define ASCENDC_MATMUL_AICORE
#endif

#if (defined(__NPU_ARCH__) &&                                               \
     ((__NPU_ARCH__ == 1001) || (__NPU_ARCH__ == 2002) ||                   \
      (__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3002) ||                   \
      (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)))
#ifndef ASCENDC_DUMP
#define ASCENDC_DUMP 1
#endif
#endif

#if defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 0)
    #undef ASCENDC_DUMP
#endif

namespace AscendC {
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2002) || (__NPU_ARCH__ == 2201) ||           \
    (__NPU_ARCH__ == 3002) || (__NPU_ARCH__ == 3102) || (__NPU_ARCH__ == 3101) ||           \
    (__NPU_ARCH__ == 5102)) // Available for V200 and V210
constexpr int32_t QUE_MAX_EVENT = 8;
#else
constexpr int32_t QUE_MAX_EVENT = 4;
#endif
constexpr int32_t HF32_MODE_BIT = 46;
constexpr int32_t HF32_TRANS_MODE_BIT = 47;
constexpr int32_t MM_LAYOUT_MODE_BIT = 51;
constexpr int32_t LEAKY_RELU_MODE_BIT = 50;
constexpr int32_t CAST_MODE_BIT = 59;

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102) || \
    (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3113)) || defined(__ASC_NPU_HOST__)
__aicore__ constexpr inline int32_t CeilDivision(int32_t num1, int32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1 || !defined(SPLIT_CORE_VEC)
    return (num1 + num2 - 1) / num2;
#else
    return get_repeat_ceiling(num1, num2);
#endif
}
#else
__aicore__ inline int32_t CeilDivision(int32_t num1, int32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}
#endif

#if !defined(__NPU_HOST__) && !defined(__ASCC_HOST__)

#ifndef ASCC_PARAM_BLOCK_CUBE
#define ASCC_PARAM_BLOCK_CUBE
const int32_t BLOCK_CUBE = 16;
#endif

#else // defined(__NPU_HOST__) || defined(__ASCC_HOST__)

#ifndef ASCC_PARAM_BLOCK_CUBE
#define ASCC_PARAM_BLOCK_CUBE
constexpr uint32_t BLOCK_CUBE = 16;
#endif
#endif // !defined(__NPU_HOST__) && !defined(__ASCC_HOST__)
} // namespace AscendC

#if defined(ASCENDC_CPU_DEBUG)
extern int32_t g_matmulCount;
#endif

#endif //IMPL_EXPERIMENTAL_TENSOR_API_UTILS_TENSOR_TILE_MACRO_H
