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
 * \file kernel_macros.h
 * \brief
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message("impl/basic_api/kernel_macros.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"basic_api/kernel_tpipe.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_MACROS_H__
#endif
#ifndef ASCENDC_KERNEL_MACROS_H
#define ASCENDC_KERNEL_MACROS_H

#include "impl/utils/sys_macros.h"
#include "impl/utils/sys_constants.h"

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

#if defined(__ASC_DISABLE_RESERVED_UBUF__)
#define __ASC_RESERVED_UBUF_UNAVAILABLE_ATTR__(msg) __attribute__((unavailable(msg)))
#else
#define __ASC_RESERVED_UBUF_UNAVAILABLE_ATTR__(msg)
#endif

#define __ASC_RESERVED_UBUF_CAT__(a, b) __ASC_RESERVED_UBUF_CAT_IMPL__(a, b)
#define __ASC_RESERVED_UBUF_CAT_IMPL__(a, b) a##b
// Probe whether an architecture has been registered below. A registered marker expands to
// "~, 1", so CHECK_N returns 1; an unregistered marker stays as one token and returns 0.
#define __ASC_RESERVED_UBUF_PROBE__() ~, 1
#define __ASC_RESERVED_UBUF_CHECK_N__(_0, n, ...) n
#define __ASC_RESERVED_UBUF_CHECK__(...) __ASC_RESERVED_UBUF_CHECK_N__(__VA_ARGS__, 0)
#define __ASC_RESERVED_UBUF_IS_PROBE__(...) __ASC_RESERVED_UBUF_CHECK__(__VA_ARGS__)

// Architecture whitelist for __ASC_USE_RESERVED_UBUF__. Add new chips here first so typos
// report a clear unsupported-architecture error even when reserved UBUF disabling is off.
#define __ASC_RESERVED_UBUF_ARCH_2201 __ASC_RESERVED_UBUF_PROBE__()
#define __ASC_RESERVED_UBUF_ARCH_3510 __ASC_RESERVED_UBUF_PROBE__()
#define __ASC_RESERVED_UBUF_IS_SUPPORTED_ARCH__(arch) \
    __ASC_RESERVED_UBUF_IS_PROBE__(__ASC_RESERVED_UBUF_CAT__(__ASC_RESERVED_UBUF_ARCH_, arch))

#define __ASC_RESERVED_UBUF_PRAGMA__(x) _Pragma(#x)
#define __ASC_RESERVED_UBUF_UNSUPPORTED_ARCH_ERROR__() \
    __ASC_RESERVED_UBUF_PRAGMA__(GCC error \
        "unsupported chip architecture in __ASC_USE_RESERVED_UBUF__; supported architectures are 2201 and 3510")

#if defined(__ASC_DISABLE_RESERVED_UBUF__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#define __ASC_USE_RESERVED_UBUF_2201(msg) __ASC_RESERVED_UBUF_UNAVAILABLE_ATTR__(msg)
#else
#define __ASC_USE_RESERVED_UBUF_2201(msg)
#endif

#if defined(__ASC_DISABLE_RESERVED_UBUF__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define __ASC_USE_RESERVED_UBUF_3510(msg) __ASC_RESERVED_UBUF_UNAVAILABLE_ATTR__(msg)
#else
#define __ASC_USE_RESERVED_UBUF_3510(msg)
#endif

// Dispatch one architecture: unsupported chips fail fast, supported chips expand to the
// architecture-specific unavailable attribute only when __ASC_DISABLE_RESERVED_UBUF__ matches.
#define __ASC_USE_RESERVED_UBUF_SUPPORTED_0(arch, msg) __ASC_RESERVED_UBUF_UNSUPPORTED_ARCH_ERROR__()
#define __ASC_USE_RESERVED_UBUF_SUPPORTED_1(arch, msg) \
    __ASC_RESERVED_UBUF_CAT__(__ASC_USE_RESERVED_UBUF_, arch)(msg)
#define __ASC_USE_RESERVED_UBUF_CALL__(supported, arch, msg) \
    __ASC_USE_RESERVED_UBUF_CALL_IMPL__(supported, arch, msg)
#define __ASC_USE_RESERVED_UBUF_CALL_IMPL__(supported, arch, msg) \
    __ASC_RESERVED_UBUF_CAT__(__ASC_USE_RESERVED_UBUF_SUPPORTED_, supported)(arch, msg)
#define __ASC_USE_RESERVED_UBUF_IMPL__(arch, msg) \
    __ASC_USE_RESERVED_UBUF_CALL__(__ASC_RESERVED_UBUF_IS_SUPPORTED_ARCH__(arch), arch, msg)

// Public entry supports one to five architecture arguments followed by the diagnostic message.
// Example: __ASC_USE_RESERVED_UBUF__(2201, 3510, "API is forbidden ...")
#define __ASC_USE_RESERVED_UBUF_1(arch1, msg) __ASC_USE_RESERVED_UBUF_IMPL__(arch1, msg)
#define __ASC_USE_RESERVED_UBUF_2(arch1, arch2, msg) \
    __ASC_USE_RESERVED_UBUF_IMPL__(arch1, msg)        \
    __ASC_USE_RESERVED_UBUF_IMPL__(arch2, msg)
#define __ASC_USE_RESERVED_UBUF_3(arch1, arch2, arch3, msg) \
    __ASC_USE_RESERVED_UBUF_2(arch1, arch2, msg)             \
    __ASC_USE_RESERVED_UBUF_IMPL__(arch3, msg)
#define __ASC_USE_RESERVED_UBUF_4(arch1, arch2, arch3, arch4, msg) \
    __ASC_USE_RESERVED_UBUF_3(arch1, arch2, arch3, msg)             \
    __ASC_USE_RESERVED_UBUF_IMPL__(arch4, msg)
#define __ASC_USE_RESERVED_UBUF_5(arch1, arch2, arch3, arch4, arch5, msg) \
    __ASC_USE_RESERVED_UBUF_4(arch1, arch2, arch3, arch4, msg)             \
    __ASC_USE_RESERVED_UBUF_IMPL__(arch5, msg)

#define __ASC_USE_RESERVED_UBUF_PICK__( \
    _1, _2, _3, _4, _5, _6, NAME, ...) NAME
#define __ASC_USE_RESERVED_UBUF__(...)                                                \
    __ASC_USE_RESERVED_UBUF_PICK__(                                                   \
        __VA_ARGS__, __ASC_USE_RESERVED_UBUF_5, __ASC_USE_RESERVED_UBUF_4,            \
        __ASC_USE_RESERVED_UBUF_3, __ASC_USE_RESERVED_UBUF_2,                         \
        __ASC_USE_RESERVED_UBUF_1)(__VA_ARGS__)

namespace AscendC {
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2002) || (__NPU_ARCH__ == 2201) ||           \
    (__NPU_ARCH__ == 3002) || (__NPU_ARCH__ == 3102) || (__NPU_ARCH__ == 3510) ||           \
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

} // namespace AscendC

#if defined(ASCENDC_CPU_DEBUG)
extern int32_t g_matmulCount;
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 1001 || __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3102)
#define ASCEND_IS_AICORE constexpr(true)
#else
#define ASCEND_IS_AICORE constexpr(false)
#endif

namespace AscendC {
namespace Reg {
} // namespace Reg
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 3003) || \
    ((__NPU_ARCH__ == 3113)))
namespace MicroAPI = Reg;
#endif
} // namespace AscendC
#endif // ASCENDC_KERNEL_MACROS_H
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_MACROS_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_MACROS_H__
#endif
