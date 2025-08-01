/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_bp_tiling_util.h
 * \brief
 */

#ifndef AICORE_ADV_API_TILING_CONV_BACKPROP_CONV3D_BP_TILING_UTIL_H
#define AICORE_ADV_API_TILING_CONV_BACKPROP_CONV3D_BP_TILING_UTIL_H

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <cstddef>

namespace ConvBackpropApi {

constexpr uint32_t DB_ON = 2;
constexpr uint64_t L1_SIZE = 524160; // 512 * 1024 - 128;
constexpr int64_t L0_SIZE = 65536;
constexpr uint32_t BEST_BASE_M = 128;
constexpr uint32_t BEST_BASE_K = 128;
constexpr uint32_t BEST_BASE_N = 256;
constexpr int64_t WORKSIZE = 16777216; // 16 * 1024 * 1024;
constexpr int32_t BUFFER_NUM_L1 = 4;

// dw
constexpr uint32_t C04 = 4;
constexpr uint32_t C16 = 16;
constexpr int32_t MIN_BATCHDIM = 4;
constexpr uint32_t SECOND_BASE_M = 64;
constexpr uint32_t SECOND_BASE_K = 64;
constexpr uint32_t SECOND_BASE_N = 512;
constexpr uint32_t MIN_STEPK = 2;
constexpr uint32_t ROW_NUM = 2;

constexpr uint32_t C0_BYTE_SIZE = 32;
constexpr uint32_t LOAD3D_MAX_STRIDE_H_W = 63;
constexpr uint32_t LOAD3D_MAX_DILATION_H_W = 255;
constexpr uint32_t LOAD3D_MAX_PAD = 255;
constexpr uint32_t LOAD3D_MAX_FILTER_H_W = 511;
constexpr uint32_t LOAD3D_MAX_DDR2L1_SIZE = 65535;

constexpr uint32_t DB_OFF = 1;
constexpr const uint32_t ROW_FIRST = 1;
constexpr const uint32_t COL_FIRST = 2;

constexpr int32_t KERNEL_HW_4 = 4;
constexpr int32_t KERNEL_HW_9 = 9;
constexpr int32_t KERNEL_HW_16 = 16;
constexpr uint32_t NUM_HALF = 2;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_64 = 64;
constexpr uint32_t SPLIT_M_K = 1;
constexpr uint32_t SPLIT_N_K = 2;
constexpr uint32_t SPLIT_M_N = 3;
constexpr uint32_t L1_DEPTH_16 = 16;
constexpr uint32_t L1_DEPTH_8 = 8;
constexpr uint32_t L1_DEPTH_4 = 4;
constexpr uint32_t L1_DEPTH_2 = 2;
constexpr uint32_t STEP_2 = 2;

// dx only
constexpr size_t OUTPUT_PADDING_DIM = 5;
constexpr int32_t BYTE_BLOCK = 32;
constexpr uint32_t B16_BITS = 4;
constexpr uint32_t FP32_BITS = 3;
constexpr uint32_t FP32_DATA_SIZE = 4;
constexpr uint32_t F16_DATA_SIZE = 2; // BF16和FP16共用
constexpr uint32_t NUM_FP32_C1OUT = 2;
constexpr int32_t FMAP_H_NUM = 2;

const size_t Y_INDEX = 0;
const size_t FILTER_INDEX = 1;
const size_t OUTPUT_BP_INDEX = 2;
const size_t BAIS_INDEX = 3;
const size_t OFFSET_W_INDEX = 4;
const size_t OUTPUT_PADDING_INDEX = 5;
const size_t OFFSET_X_INDEX = 6;

const int32_t DIM_LOW = 1;
const int32_t PAD_DIM_LOW = 0;
const int32_t PAD_DIM_UP = 255;
const int32_t STRIDES_DIM_HW_UP = 63;
const int32_t STRIDES_DIM_DEPTH_UP = 255;
const int32_t GROUPS_LOW = 1;
const int32_t GROUPS_UP = 65535;
const int32_t K_START_POSITION_MAX = 65535;
const int32_t K_NUM_TWO = 2;

#define OP_TILING_CHECK(cond, log_func, expr)                                                                          \
    do {                                                                                                               \
        if (cond) {                                                                                                    \
            log_func;                                                                                                  \
            expr;                                                                                                      \
        }                                                                                                              \
    } while (0)

#define D_OP_LOGE(opname, fmt, ...) OpLogSub(OP, DLOG_ERROR, opname, fmt, ##__VA_ARGS__)
#define OP_LOGE_WITHOUT_REPORT(opname, ...) D_OP_LOGE(get_op_info(opname), __VA_ARGS__)

#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...)                                                                   \
    do {                                                                                                               \
        OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);                                                       \
        REPORT_INNER_ERROR("E69999", "op[%s], " err_msg, get_cstr(get_op_info(op_name)), ##__VA_ARGS__);               \
    } while (0)

/**
 * if y is 0, return x
 */
template <typename T>
typename std::enable_if<std::is_signed<T>::value, T>::type CeilDiv(T x, T y)
{
    if (y != 0 && x != 0) {
        const T quotient = x / y;
        return (x % y != 0 && ((x > 0) == (y > 0))) ? (quotient + 1) : quotient;
    }

    return x;
}

/**
 * if y is 0, return x
 */
template <typename T>
typename std::enable_if<std::is_unsigned<T>::value, T>::type CeilDiv(T x, T y)
{
    if (y != 0 && x != 0) {
        const T quotient = x / y;
        return (x % y != 0) ? (quotient + 1) : quotient;
    }

    return x;
}

template <typename T>
inline T CeilDivision(T num1, T num2)
{
    return CeilDiv(num1, num2);
}

inline int64_t CeilDivision(int64_t num1, int32_t num2)
{
    return CeilDiv(num1, static_cast<int64_t>(num2));
}

template <typename T>
inline T Min(T num1, T num2)
{
    return std::min(num1, num2);
}

inline int32_t Min(int64_t num1, int32_t num2)
{
    return static_cast<int32_t>(std::min(num1, static_cast<int64_t>(num2)));
}

inline int32_t Min(int32_t num1, int64_t num2)
{
    return static_cast<int32_t>(std::min(num2, static_cast<int64_t>(num1)));
}

inline int32_t CalcHi(int32_t ho, int32_t stride_h, int32_t kernel_h_dilation, int32_t ori_hi)
{
    return Min(static_cast<int64_t>(ho - 1) * stride_h + kernel_h_dilation, ori_hi);
}

inline int32_t CalcHo(int64_t k, int32_t wo)
{
    if (k == 0 || wo == 0) {
        return 0;
    }
    //  完整K是ho*wo，k可能超int32，但是下面除完以后的wo不可能超int32
    int32_t ho = static_cast<int32_t>(CeilDivision(k, wo));
    if (k % wo == 0 || wo % k == 0) {
        return ho;
    } else {
        return ho + 1;
    }
}

inline int32_t GetGcd(int32_t param1, int32_t param2)
{
    // get greatest common divisor of param1 and param2
    if (param1 < param2) {
        std::swap(param1, param2);
    }
    if (param2 == 0) {
        return 0;
    }
    if (param1 % param2 == 0) {
        return param2;
    } else {
        return GetGcd(param2, param1 - param2);
    }
}

inline void GetFactors(std::vector<int32_t>& factor_list, int64_t src_num, int32_t max_factor)
{
    int32_t max_num = Min(src_num, max_factor);
    for (int32_t factor = 1; factor <= max_num; factor++) {
        if (src_num % factor == 0) {
            factor_list.push_back(factor);
        }
    }
}

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type CeilAlign(T x, T align)
{
    return CeilDiv(x, align) * align;
}

inline int64_t Lcm(int64_t param1, int64_t param2)
{
    int64_t pram1_lcm = param1;
    int64_t pram2_lcm = param2;
    int64_t temp = pram1_lcm * pram2_lcm;
    int64_t param1_temp = pram1_lcm;
    while (pram1_lcm % pram2_lcm != 0) {
        param1_temp = pram1_lcm;
        pram1_lcm = pram2_lcm;
        pram2_lcm = param1_temp % pram2_lcm;
    }
    return temp / pram2_lcm;
}

inline int64_t Lcm(int32_t param1, int32_t param2)
{
    return Lcm(static_cast<int64_t>(param1), static_cast<int64_t>(param2));
}

inline int64_t Lcm(int64_t param1, int32_t param2)
{
    return Lcm(param1, static_cast<int64_t>(param2));
}

/**
 * if align is 0, return 0
 */
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type FloorAlign(T x, T align)
{
    return align == 0 ? 0 : x / align * align;
}

} // namespace ConvBackpropApi
#endif // AICORE_ADV_API_TILING_CONV_BACKPROP_CONV3D_BP_TILING_UTIL_H