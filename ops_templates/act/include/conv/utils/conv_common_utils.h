/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_common_utils.h
 * \brief
 */

#ifndef UTILS_CONV_COMMON_UTILS_H
#define UTILS_CONV_COMMON_UTILS_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <iomanip>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "conv_layout_utils.h"

namespace Act {
namespace Conv {
struct ConvShape {
    int64_t batch_ = 0;
    int64_t cin_ = 0;
    int64_t din_ = 0;
    int64_t hin_ = 0;
    int64_t win_ = 0;
    int64_t cout_ = 0;
    int32_t kh_ = 0;
    int32_t kw_ = 0;
    int32_t group_ = 0;
    int64_t do_ = 0;
    int64_t ho_ = 0;
    int64_t wo_ = 0;
    int32_t strideh_ = 0;
    int32_t stridew_ = 0;
    int32_t padTop_ = 0;
    int32_t padBottom_ = 0;
    int32_t padLeft_ = 0;
    int32_t padRight_ = 0;
    int32_t dilationh_ = 0;
    int32_t dilationw_ = 0;
    bool hasbias_ = false;
    bool enable_hf32_ = false;
};

struct ConvDim {
    uint32_t batchDim = 0;
    uint32_t mDim = 0;
    uint32_t nDim = 0;
};

struct ConvInterate {
    uint8_t loadAL1Flag = 0; // 是否载入AL1的标志
    uint8_t loadBL1Flag = 0; // 是否载入BL1的标志

    uint64_t mAL1Iter = 0; // AL1上m方向迭代器
    uint64_t nBL1Iter = 0; // BL1上n方向迭代器
    uint64_t batchIter = 0; // batch方向迭代器
};

struct TempIters {
    uint64_t kAL1Iter = 0;
    uint64_t kBL1Iter = 0;
    uint64_t kAL0Iter = 0;
    uint64_t kBL0Iter = 0;
    uint64_t kIter = 0;
};

struct ConvInterateMax {
    uint64_t ddr2l1LoopBatch = 0;
    uint64_t ddr2l1LoopN = 0;
    uint64_t ddr2l1LoopM = 0;
};

struct SingleCoreShape {
    uint64_t singleCoreBatch = 0;
    uint64_t singleCoreCi = 0;
    uint64_t singleCoreM = 0;
    uint64_t singleCoreN = 0;
};

constexpr int64_t MAX_N_BF16_SHAPE = 1000000;
constexpr int64_t MAX_FM_D_BF16_SHAPE = 1000000;
constexpr int64_t MAX_FM_H_BF16_SHAPE = 1000000;
constexpr int64_t MAX_FM_W_BF16_SHAPE = 1000000;
constexpr int64_t MAX_KD_BF16_SHAPE = 1000000;
constexpr int64_t MAX_KH_BF16_SHAPE = 1000000;
constexpr int64_t MAX_KW_BF16_SHAPE = 1000000;
constexpr int64_t MAX_CIN_BF16_SHAPE = 1000000;
constexpr int64_t MAX_COUT_BF16_SHAPE = 1000000;
constexpr uint64_t L0A_HALF_SIZE = 32768;
constexpr uint64_t L0B_HALF_SIZE = 32768;


// parameters mask value
constexpr uint64_t MASK_16 = 0xffff;
constexpr uint64_t MASK_8 = 0xff;
constexpr uint64_t MASK_6 = 0x3f;

// Load2dv2 parameters offset
constexpr uint8_t DST_STRIDE_OFFSET = 16;
constexpr uint8_t K_START_OFFSET = 16;
constexpr uint8_t M_STEP_OFFSET = 32;
constexpr uint8_t K_STEP_OFFSET = 40;

// Load3d parameters offset
constexpr uint8_t MSTEP_OFFSET = 16;
constexpr uint8_t POSM_OFFSET = 48;
constexpr uint8_t POSK_OFFSET = 32;
constexpr uint8_t STRIDEH_OFFSET = 6;
constexpr uint8_t KERNELW_OFFSET = 12;
constexpr uint8_t KERNELH_OFFSET = 20;
constexpr uint8_t DILATIONW_OFFSET = 28;
constexpr uint8_t DILATIONH_OFFSET = 36;
constexpr uint8_t CIN_OFFSET = 48;

constexpr uint32_t PAD_SIZE = 4;
constexpr uint64_t PAD_IDX_T = 2;
constexpr uint64_t PAD_IDX_B = 3;
constexpr uint64_t PAD_IDX_L = 0;
constexpr uint64_t PAD_IDX_R = 1;
constexpr uint64_t MAX_PAD_R = 255;

constexpr uint64_t PADDING_ALIGN_SIZE = 32;
constexpr uint32_t BT_BLOCK_SIZE = 32;
constexpr uint32_t BLOCK_L0_N = 16;
constexpr uint32_t BLOCK_L0_M = 16;

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

__aicore__ inline  uint64_t ConvCeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}
__aicore__ inline uint32_t ConvAlign(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

__aicore__ inline int64_t Max(const int64_t numA, const int64_t numB)
{
    return numA > numB ? numA : numB;
}

__aicore__ inline int64_t Min(const int64_t numA, const int64_t numB)
{
    return numA < numB ? numA : numB;
}

template <size_t N, typename Tp>
__aicore__ constexpr inline decltype(auto) GetIntegralConstant()
{
    static_assert(AscendC::Std::is_tuple_v<Tp>, "Input must be a AscendC::Std::tuple type");
    return AscendC::Std::tuple_element<N, Tp>::type::value;
}

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0)

template <typename T>
struct GetDstType {
    using Type = T;
};

template <>
struct GetDstType<float> {
    using Type = float;
};

template <>
struct GetDstType<half> {
    using Type = float;
};

template <>
struct GetDstType<bfloat16_t> {
    using Type = float;
};

template <>
struct GetDstType<int8_t> {
    using Type = int32_t;
};

template <>
struct GetDstType<hifloat8_t> {
    using Type = float;
};

template <>
struct GetDstType<fp8_e4m3fn_t> {
    using Type = float;
};

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
__aicore__ inline void set_mark(uint64_t v)
{
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1

#else
    __asm__ __volatile__("");
    asm volatile("MOV COND, %0\n" : "+l"(v));
    __asm__ __volatile__("");
#endif
}
#endif

} // namespace Conv
} // namespace Act
#endif