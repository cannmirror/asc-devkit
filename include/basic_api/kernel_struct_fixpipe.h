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
 * \file kernel_struct_fixpipe.h
 * \brief
 */
#ifndef ASCENDC_MODULE_STRUCT_FIXPIPE_H
#define ASCENDC_MODULE_STRUCT_FIXPIPE_H

namespace AscendC {
enum class CO2Layout : uint8_t {
    NZ = 0,
    ROW_MAJOR, // ND Row
    COLUMN_MAJOR // ND Column
};

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
struct FixpipeConfig {
    CO2Layout format;
    bool isToUB;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    bool enableFixVal = false;
#endif
};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
constexpr FixpipeConfig CFG_NZ_FIX = {CO2Layout::NZ, false, true};
#endif

constexpr FixpipeConfig CFG_NZ = {CO2Layout::NZ, false};
constexpr FixpipeConfig CFG_ROW_MAJOR = {CO2Layout::ROW_MAJOR, false};
constexpr FixpipeConfig CFG_COLUMN_MAJOR = {CO2Layout::COLUMN_MAJOR, false};
#else
struct FixpipeConfig {
    CO2Layout format;
};

constexpr FixpipeConfig CFG_NZ = {CO2Layout::NZ};
constexpr FixpipeConfig CFG_ROW_MAJOR = {CO2Layout::ROW_MAJOR};
constexpr FixpipeConfig CFG_COLUMN_MAJOR = {CO2Layout::COLUMN_MAJOR};
#endif

struct FixpipeParamsV220 {
    __aicore__ FixpipeParamsV220() {}

    __aicore__ FixpipeParamsV220(const uint16_t nSizeIn, const uint16_t mSizeIn, const uint16_t srcStrideIn,
        const uint32_t dstStrideIn, const bool reluEnIn)
        : nSize(nSizeIn),
          mSize(mSizeIn),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          reluEn(reluEnIn)
    {}

    __aicore__ FixpipeParamsV220(const uint16_t nSizeIn, const uint16_t mSizeIn, const uint16_t srcStrideIn,
        const uint32_t dstStrideIn, const bool reluEnIn, const QuantMode_t quantPreIn, const int64_t deqScalarIn,
        const uint16_t ndNumIn, const uint16_t srcNdStrideIn, const uint16_t dstNdStrideIn, const uint8_t unitFlagIn)
        : nSize(nSizeIn),
          mSize(mSizeIn),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          reluEn(reluEnIn),
          quantPre(quantPreIn),
          deqScalar(deqScalarIn),
          ndNum(ndNumIn),
          srcNdStride(srcNdStrideIn),
          dstNdStride(dstNdStrideIn),
          unitFlag(unitFlagIn)
    {}

    __aicore__ FixpipeParamsV220(const uint16_t nSizeIn, const uint16_t mSizeIn, const uint16_t srcStrideIn,
        const uint32_t dstStrideIn, const bool reluEnIn, const QuantMode_t quantPreIn, const int64_t deqScalarIn,
        const uint16_t ndNumIn, const uint16_t srcNdStrideIn, const uint16_t dstNdStrideIn, const uint8_t unitFlagIn,
        const bool isChannelSplitIn)
        : nSize(nSizeIn),
          mSize(mSizeIn),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          reluEn(reluEnIn),
          quantPre(quantPreIn),
          deqScalar(deqScalarIn),
          ndNum(ndNumIn),
          srcNdStride(srcNdStrideIn),
          dstNdStride(dstNdStrideIn),
          unitFlag(unitFlagIn),
          isChannelSplit(isChannelSplitIn)
    {}

    uint16_t nSize = 0;
    uint16_t mSize = 0;  // M-DirectionSize
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
    // Params: used for Quant
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    uint64_t deqScalar;
    // Params: used for nz2nd
    uint16_t ndNum = 1;
    uint16_t srcNdStride = 0;
    uint16_t dstNdStride = 0;
    bool reluEn = false;
    uint8_t unitFlag = 0;
    bool isChannelSplit = false;
};

#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
// 根据模板参数选结构体
template <CO2Layout format>
struct TransformParams {};

template <>
struct TransformParams<CO2Layout::NZ> {
    __aicore__ inline TransformParams(){};
    using PARAMS = uint8_t;
};

template <>
struct TransformParams<CO2Layout::ROW_MAJOR> {
    __aicore__ inline TransformParams(){};
    using PARAMS = Nz2NdParams;
};

template <>
struct TransformParams<CO2Layout::COLUMN_MAJOR> {
    __aicore__ inline TransformParams(){};
    using PARAMS = Nz2DnParams;
};

template <CO2Layout format = CO2Layout::ROW_MAJOR>
struct FixpipeParamsC310 {
    __aicore__ FixpipeParamsC310() {}

    __aicore__ FixpipeParamsC310(const uint16_t nSizeIn, const uint16_t mSizeIn, const uint16_t srcStrideIn,
        const uint32_t dstStrideIn)
    {
        nSize = nSizeIn;
        mSize = mSizeIn;
        srcStride = srcStrideIn;
        dstStride = dstStrideIn;
    }

    uint16_t nSize = 0;
    uint16_t mSize = 0;  // M-DirectionSize
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
    // Params: used for Quant
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    uint64_t deqScalar;
    bool reluEn = false;
    uint8_t unitFlag = 0;
    // c310 extend param
    uint8_t dualDstCtl = 0;
    bool subBlockId = false;
    typename TransformParams<format>::PARAMS params;
    bool isChannelSplit = false;
};
#endif

using FixpipeParamsM300 = FixpipeParamsV220;
using FixpipeParamsM310 = FixpipeParamsV220;
} // namespace AscendC

/* **************************************************************************************************
 * Fixpipe(Layout) API Level2                                              *
 * ************************************************************************************************* */
namespace AscendC {

struct FixpipeTrait {
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    bool enableRelu = false;
    bool enableChannleSplit = false;
    uint8_t unitFlag = false;
    uint8_t dualDstCtl = false;
};
constexpr FixpipeTrait DEFAULT_FIXPIPE_TRAIT;

}
#endif // ASCENDC_MODULE_STRUCT_FIXPIPE_H