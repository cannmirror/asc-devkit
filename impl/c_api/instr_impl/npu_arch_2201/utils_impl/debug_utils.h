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
#warning "impl/c_api/instr_impl/npu_arch_2201/utils_impl/debug_utils.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "c_api/asc_simd.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_UTILS_IMPL_DEBUG_UTILS_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_UTILS_IMPL_DEBUG_UTILS_H

#ifndef __host_aicore__
#define __host_aicore__ [host, aicore]
#endif // __host_aicore__

#if (_ASCENDC_HAS_BISHENG_COMPILER)
#define ASCENDC_HOST __host__
#define ASCENDC_AICORE __aicore__
#define ASCENDC_HOST_AICORE __host_aicore__
#else
#define ASCENDC_HOST
#define ASCENDC_AICORE
#define ASCENDC_HOST_AICORE
#endif

#include "include/utils/std/tuple.h"
#include "include/utils/std/algorithm.h"
#include "include/utils/std/type_traits.h"
#include "include/utils/std/utility.h"

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 8
#endif

inline __gm__ uint8_t* g_sysPrintFifoSpace = nullptr;

constexpr uint16_t ASC_ONE_PARAM_SIZE = 8;
constexpr uint32_t ASC_DUMP_CORE_COUNT = 75;
constexpr uint32_t ASC_MAGIC_NUM_CHECK = 0xAE86;
constexpr uint64_t ASC_TIMEOUT_CYCLE = 50 * 1000 * 1000 * 5; // 5s

struct arch_version {
    static constexpr uint32_t v2201 = 2201;
};

struct get_arrch_version {
    __aicore__ inline constexpr uint32_t operator()() const {
#ifdef __NPU_ARCH__
        return __NPU_ARCH__;
#else
        return 0;
#endif
    }
};

constexpr uint32_t CURRENT_ARCH_VERSION = get_arrch_version{}();

struct OpSystemRunCfg {
    uint64_t l2Cacheoffset;
};

enum FuncMetaType { // 函数级TLV类型
    F_TYPE_KTYPE = 1, // kernel type tlv
    F_TYPE_CROSS_CORE_SYNC = 2, // cross core sync
    F_TYPE_MIX_TASK_RATION = 3, // MIX CORE TYPE
    F_TYPE_L0_EXCEPTION_DFX = 4, // DFX tlv for header
    F_TYPE_L0_EXCEPTION_DFX_ARGSINFO = 5, // DFX tlv for args info
    F_TYPE_L0_EXCEPTION_DFX_IS_TIK = 6, // DFX tlv mark for TIK
    F_TYPE_DETERMINISTIC_INFO = 13,
    F_TYPE_FUNCTION_ENTRY_INFO= 14,
    F_TYPE_BLOCK_DIM_INFO = 15,
    F_TYPE_MAX
};

enum KernelType {
    K_TYPE_AICORE = 1,              // c100/m200
    K_TYPE_AIC = 2,                 // v220-cube
    K_TYPE_AIV = 3,                 // v220-vec
    K_TYPE_MIX_AIC_MAIN = 4,        // v220 mix cube/vector 1:2
    K_TYPE_MIX_AIV_MAIN = 5,        // v220 mix vector/cube 1:2
    K_TYPE_AIC_ROLLBACK = 6,        // v220-cube，aic rollback
    K_TYPE_AIV_ROLLBACK = 7,        // v220-vec，aiv rollback
    K_TYPE_MAX
};

struct BaseTlv {  // TLV头部定义
    unsigned short type;
    unsigned short len;
};

struct FunMetaKType {
    BaseTlv head;
    unsigned int ktype;
};

struct FunLevelKType {
    struct FunMetaKType ktypeMeta;
};

struct BinaryMetaAscFeature {
    BaseTlv head;
    uint32_t feature;
};

struct BinaryMetaVersion {
    BaseTlv head;      // B_TYPE_BIN_VERSION = 0
    uint32_t version;  // version info
};

__aicore__ inline void enable_printf()
{
#ifdef __ENABLE_ASCENDC_PRINTF__
#if defined (ASCENDC_DUMP) && (ASCENDC_DUMP == 1)
    static const struct BinaryMetaAscFeature __asc_feature_print__ __attribute__ ((used, section (".ascend.meta"))) =
        {4, 4, 1};
#endif
#endif
}

enum class DumpType : uint8_t {
    DUMP_DEFAULT = 0,
    DUMP_SCALAR,
    DUMP_TENSOR,
    DUMP_SHAPE,
    DUMP_ASSERT,
    DUMP_META,
    DUMP_TIME_STAMP,
    DUMP_SIMT,
    DUMP_BUFI,
    DUMP_BUFO,
    DUMP_SKIP
};

struct BlockRingBufInfo {
    uint32_t length = 0U;        // total size per block (include head and r/w info)
    uint32_t coreId = 0U;        // current core id
    uint32_t blockNum = 0U;      // total core num
    uint32_t ringBufLen = 0U;    // fifo buff size (print tlv storage)
    uint16_t magic = 0U;         // magic number
    uint16_t flag = 0U;          // 0: simd, 1: simt
    uint32_t rsv = 0U;           // reserve
    uint64_t ringBufAddr = 0U;   // start addr of fifo buff
    uint32_t resvMem[6];        // reserved
};

struct RingBufWriteInfo {
    uint32_t type = static_cast<uint32_t>(DumpType::DUMP_BUFI); // DumpType = DUMP_BUFI
    uint32_t length = 0U;       // u64 + u64
    uint64_t bufOffset = 0U;    // the offset of write addr relative to ringBufAddr
    uint64_t packIdx = 0U;      // print pack counter
};

struct RingBufReadInfo {
    uint32_t type = static_cast<uint32_t>(DumpType::DUMP_BUFO); // DumpType = DUMP_BUFO
    uint32_t length = 0U;       // u64 + u64
    uint64_t bufOffset = 0U;    // the offset of read addr relative to ringBufAddr
    uint64_t resv = 0U;
};

struct SkipTlvInfo {
    uint32_t type = static_cast<uint32_t>(DumpType::DUMP_SKIP); // DumpType = DUMP_SKIP
    uint32_t length = 0U;
};

struct PrintTlvInfoHead {
    uint32_t type = static_cast<uint32_t>(DumpType::DUMP_SCALAR);
    uint32_t length = 0U;
    uint32_t resvMem[2];         // reserved
    uint64_t fmtOffset = 0U;     // offset of fmt string from the start of fmtOffset addr
};

constexpr uint32_t MIN_TLV_LEN = sizeof(SkipTlvInfo);

struct DumpTensorTlvInfoHead {
    uint32_t type = static_cast<uint32_t>(DumpType::DUMP_TENSOR); // DumpType = DUMP_TENSOR
    uint32_t length = 0U;            // Length of (addr dataType desc bufferId position dumpSize dumpData align)
    uint32_t tensorAddr = 0U;        // Address of Tensor
    uint32_t dataType = 0U;          // Data type: int32_t/half/...
    uint32_t desc = 0U;              // Usr id
    uint32_t bufferId = 0U;          // 0
    uint16_t position = 0U;          // Position GM,UB,L1,L0C
    uint16_t resv0 = 0U;             // reserved
    uint32_t dim = 0U;               // shape dim
    uint32_t shape[K_MAX_SHAPE_DIM]; // dim <= 8
    uint32_t resv1 = 0U;             // reserved
    uint32_t dumpSize = 0U;          // Length of dumpData
                                     // dumpData[dumpSize], Tensor data
};

enum class DumpTensorDataType : uint32_t {
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_INT8 = 2,
    ACL_INT32 = 3,
    ACL_UINT8 = 4,
    ACL_INT16 = 6,
    ACL_UINT16 = 7,
    ACL_UINT32 = 8,
    ACL_INT64 = 9,
    ACL_UINT64 = 10,
    ACL_DOUBLE = 11,
    ACL_BOOL = 12,
    ACL_STRING = 13,
    ACL_COMPLEX64 = 16,
    ACL_COMPLEX128 = 17,
    ACL_BF16 = 27,
    ACL_INT4 = 29,
    ACL_UINT1 = 30,
    ACL_COMPLEX32 = 33,
    ACL_HIFLOAT8 = 34,
    ACL_FLOAT8_E5M2 = 35,
    ACL_FLOAT8_E4M3FN = 36,
    ACL_FLOAT8_E8M0 = 37,
    ACL_FLOAT6_E3M2 = 38,
    ACL_FLOAT6_E2M3 = 39,
    ACL_FLOAT4_E2M1 = 40,
    ACL_FLOAT4_E1M2 = 41,
    ACL_MAX = 42,
};

template <typename T>
__aicore__ constexpr inline DumpTensorDataType get_dump_datatype()
{
    if constexpr (AscendC::Std::is_same<T, bool>::value) {
        return DumpTensorDataType::ACL_BOOL;
    } else if (AscendC::Std::is_same<T, uint8_t>::value) {
        return DumpTensorDataType::ACL_UINT8;
    } else if (AscendC::Std::is_same<T, int8_t>::value) {
        return DumpTensorDataType::ACL_INT8;
    } else if (AscendC::Std::is_same<T, int16_t>::value) {
        return DumpTensorDataType::ACL_INT16;
    } else if (AscendC::Std::is_same<T, uint16_t>::value) {
        return DumpTensorDataType::ACL_UINT16;
    } else if (AscendC::Std::is_same<T, int32_t>::value) {
        return DumpTensorDataType::ACL_INT32;
    } else if (AscendC::Std::is_same<T, uint32_t>::value) {
        return DumpTensorDataType::ACL_UINT32;
    } else if (AscendC::Std::is_same<T, uint64_t>::value) {
        return DumpTensorDataType::ACL_UINT64;
    } else if (AscendC::Std::is_same<T, int64_t>::value) {
        return DumpTensorDataType::ACL_INT64;
    } else if (AscendC::Std::is_same<T, float>::value) {
        return DumpTensorDataType::ACL_FLOAT;
    } else if (AscendC::Std::is_same<T, half>::value) {
        return DumpTensorDataType::ACL_FLOAT16;
    } else if (AscendC::Std::is_same<T, bfloat16_t>::value) {
        return DumpTensorDataType::ACL_BF16;
    } else {
        return DumpTensorDataType::ACL_MAX;
    }
}

enum class Hardware : uint8_t { GM, UB, L1, L0A, L0B, L0C, BIAS, FIXBUF, MAX };
__aicore__ constexpr inline uint32_t div_ceil(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ constexpr inline uint32_t align_up(uint32_t a, uint32_t b)
{
    return div_ceil(a, b) * b;
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif