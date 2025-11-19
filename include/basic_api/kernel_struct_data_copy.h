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
 * \file kernel_struct_data_copy.h
 * \brief
 */
#ifndef ASCENDC_MODULE_STRUCT_DATA_COPY_H
#define ASCENDC_MODULE_STRUCT_DATA_COPY_H
#include "utils/kernel_utils_mode.h"

namespace AscendC {
#ifndef ASCC_ENUM_DATAFORMAT
#define ASCC_ENUM_DATAFORMAT
enum class DataFormat : uint8_t {
    ND = 0,
    NZ,
    NCHW,
    NC1HWC0,
    NHWC,
    NCDHW,
    NDC1HWC0,
    FRACTAL_Z_3D,
};
#endif // ASCC_ENUM_DATAFORMAT

#if defined (__NPU_ARCH__) && (__NPU_ARCH__ == 3113)
struct DataCopyParamsL311 {
    __aicore__ DataCopyParamsL311() {}

    __aicore__ DataCopyParamsL311(const uint16_t count, const uint16_t len, const uint16_t srcStrideIn,
        const uint16_t dstStrideIn, const uint16_t sidIn = 0)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          sid(sidIn)
    {}
    __aicore__ inline void SetBlockCount(const uint16_t blockCount_)
    {
        blockCount = blockCount_;
    }

    __aicore__ inline void SetBlockLen(const uint16_t blockLen_)
    {
        blockLen = blockLen_;
    }

    __aicore__ inline void SetSrcStride(const uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstStride(const uint16_t dstStride_)
    {
        dstStride = dstStride_;
    }

    __aicore__ inline void SetSid(const uint16_t sid_)
    {
        sid = sid_;
    }

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint16_t blockLen = 0;
    uint16_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t dstStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t sid = 0;
};
#endif

#if defined (__NPU_ARCH__) && ((__NPU_ARCH__ == 3113))
using DataCopyParams = DataCopyParamsL311;
#elif defined (__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103))
struct DataCopyParams {
    __aicore__ DataCopyParams() {}

    __aicore__ DataCopyParams(const uint16_t count, const uint16_t len, const uint16_t srcStrideIn,
        const uint16_t dstStrideIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn)
    {}
    __aicore__ inline void SetBlockCount(const uint16_t blockCount_)
    {
        blockCount = blockCount_;
    }

    __aicore__ inline void SetBlockLen(const uint16_t blockLen_)
    {
        blockLen = blockLen_;
    }

    __aicore__ inline void SetSrcStride(const uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstStride(const uint16_t dstStride_)
    {
        dstStride = dstStride_;
    }

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint16_t blockLen = 0;
    uint16_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t dstStride = DEFAULT_DATA_COPY_STRIDE;
};
#else
struct DataCopyParams {
    __aicore__ DataCopyParams() {}

    __aicore__ DataCopyParams(const uint16_t count, const uint16_t len, const uint16_t srcStrideIn,
        const uint16_t dstStrideIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn)
    {}

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint16_t blockLen = 0;
    // srcStride and dstStride will be deprecated, use srcGap and dstGap instead.
    union {
        uint16_t srcGap = 0;
        // srcStride will be deprecated, use srcGap instead
        uint16_t srcStride;
    };

    union {
        uint16_t dstGap = 0;
        // dstStride will be deprecated, use dstGap instead
        uint16_t dstStride;
    };
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
enum class DataCopyMVType : uint8_t {
    UB_TO_OUT = 0,
    OUT_TO_UB = 1,
};

struct NdDmaConfig {
    static constexpr uint16_t unsetPad = 0xffff;
    bool isNearestValueMode = false;
    uint16_t loopLpSize = unsetPad; // Left padding size of all dimensions, must be less than 256.
    uint16_t loopRpSize = unsetPad; // Right padding size of all dimensions, must be less than 256.
    bool ascOptimize = false;       // used for AscendC optimization on special senario.
};
using MultiCopyConfig = NdDmaConfig;  // reserve old name
constexpr NdDmaConfig kDefaultNdDmaConfig = { false, NdDmaConfig::unsetPad, NdDmaConfig::unsetPad,
    false };
constexpr NdDmaConfig kDefaultMultiCopyConfig = { false, NdDmaConfig::unsetPad, NdDmaConfig::unsetPad,
    false };  // reserve old name

template <uint8_t dim>
struct MultiCopyLoopInfo  {
    static_assert(dim >= 1 && dim <= 5, "MultiCopy Dims must be between 1 and 5.");

    // Index [0, dim) represents lowerest dimension to highest dimension accordingly.
    uint64_t loopSrcStride[dim] = {0}; // src stride info per loop.
    uint32_t loopDstStride[dim] = {0}; // dst stride info per loop.
    uint32_t loopSize[dim] = {0}; // Loop size per loop.
    uint8_t loopLpSize[dim] = {0}; // Left padding size per loop.
    uint8_t loopRpSize[dim] = {0}; // Right padding size per loop.
};

template <typename T, uint8_t dim>
struct MultiCopyParams  {
    MultiCopyLoopInfo<dim> loopInfo;
    T constantValue;
};
#endif

struct DataCopyEnhancedParams {
    __aicore__ DataCopyEnhancedParams() {}

    __aicore__ DataCopyEnhancedParams(const BlockMode blockModeIn, const DeqScale deqScaleIn, const uint64_t deqValueIn,
        const uint8_t sidStoreModeIn, const bool isReluIn, const pad_t padModeIn, const uint64_t padValueIn)
        : blockMode(blockModeIn),
          deqScale(deqScaleIn),
          deqValue(deqValueIn),
          sidStoreMode(sidStoreModeIn),
          isRelu(isReluIn),
          padMode(padModeIn),
          padValue(padValueIn)
    {}

    BlockMode blockMode = BlockMode::BLOCK_MODE_NORMAL;
    DeqScale deqScale = DeqScale::DEQ_NONE;
    uint64_t deqValue = 0;
    uint8_t sidStoreMode = 0;
    bool isRelu = false;
    pad_t padMode = pad_t::PAD_NONE;
    uint64_t padValue = 0;
    uint64_t deqTensorAddr = 0;
};

struct DataCopyCO12DstParams {
    __aicore__ DataCopyCO12DstParams() {}

    __aicore__ DataCopyCO12DstParams(const uint16_t nSizeIn, const uint16_t mSizeIn, const uint32_t dstStrideIn,
        const uint16_t srcStrideIn, const QuantMode_t quantPreIn, const uint8_t reluPreIn, const bool channelSplitIn,
        const bool nz2ndEnIn)
        : nSize(nSizeIn),
          mSize(mSizeIn),
          dstStride(dstStrideIn),
          srcStride(srcStrideIn),
          quantPre(quantPreIn),
          reluPre(reluPreIn),
          channelSplit(channelSplitIn),
          nz2ndEn(nz2ndEnIn)
    {}

    uint8_t sid = 0;
    uint16_t nSize = 0;
    uint16_t mSize = 0;
    uint32_t dstStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint8_t unitFlag = 0;
    uint8_t clipReluPre = 0;
    uint8_t eltWiseOp = 0;
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    uint8_t reluPre = 0;
    bool channelSplit = false;
    bool nz2ndEn = false;
};

#if defined (__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
struct DataCopyPadParams {
    __aicore__ DataCopyPadParams() {}

    __aicore__ DataCopyPadParams(const bool isPadValue, const uint8_t leftPadValue, const uint8_t rightPadValue,
        const uint64_t padValue)
        : isPad(isPadValue),
          leftPadding(leftPadValue),
          rightPadding(rightPadValue),
          paddingValue(padValue)
    {}
    __aicore__ inline void SetIsPad(const bool isPad_)
    {
        isPad = isPad_;
    }

    __aicore__ inline void SetLeftPadding(const uint8_t leftPadding_)
    {
        leftPadding = leftPadding_;
    }

    __aicore__ inline void SetRightPadding(const uint8_t rightPadding_)
    {
        rightPadding = rightPadding_;
    }

    __aicore__ inline void SetPaddingValue(const uint64_t paddingValue_)
    {
        paddingValue = paddingValue_;
    }

    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    uint64_t paddingValue = 0;
};
#else
struct DataCopyPadParams {
    __aicore__ DataCopyPadParams() {}

    __aicore__ DataCopyPadParams(const bool isPadValue, const uint8_t leftPadValue, const uint8_t rightPadValue,
        const uint64_t padValue)
        : isPad(isPadValue),
          leftPadding(leftPadValue),
          rightPadding(rightPadValue),
          paddingValue(padValue)
    {}

    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    uint64_t paddingValue = 0;
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
struct DataCopyExtParams {
    __aicore__ DataCopyExtParams() {}

    __aicore__ DataCopyExtParams(const uint16_t count, const uint32_t len, const int64_t srcStrideIn,
        const int64_t dstStrideIn, const uint32_t rsvIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          rsv(rsvIn)
    {}

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint32_t blockLen = 0;
    int64_t srcStride = static_cast<int64_t>(DEFAULT_DATA_COPY_STRIDE);
    int64_t dstStride = static_cast<int64_t>(DEFAULT_DATA_COPY_STRIDE);
    uint32_t rsv = 0; // reserved information
};
#elif defined (__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
      (__NPU_ARCH__ == 3113))
struct DataCopyExtParams {
    __aicore__ DataCopyExtParams() {}

    __aicore__ DataCopyExtParams(const uint16_t count, const uint32_t len, const uint32_t srcStrideIn,
        const uint32_t dstStrideIn, const uint32_t rsvIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          rsv(rsvIn)
    {}
    __aicore__ inline void SetBlockCount(const uint16_t blockCount_)
    {
        blockCount = blockCount_;
    }

    __aicore__ inline void SetBlockLen(const uint32_t blockLen_)
    {
        blockLen = blockLen_;
    }

    __aicore__ inline void SetSrcStride(const uint32_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstStride(const uint32_t dstStride_)
    {
        dstStride = dstStride_;
    }

    __aicore__ inline void SetRsv(const uint32_t rsv_)
    {
        rsv = rsv_;
    }

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint32_t blockLen = 0;
    uint32_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint32_t dstStride = DEFAULT_DATA_COPY_STRIDE;
    uint32_t rsv = 0; // reserved information
};
#else
struct DataCopyExtParams {
    __aicore__ DataCopyExtParams() {}

    __aicore__ DataCopyExtParams(const uint16_t count, const uint32_t len, const uint32_t srcStrideIn,
        const uint32_t dstStrideIn, const uint32_t rsvIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          rsv(rsvIn)
    {}

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint32_t blockLen = 0;
    uint32_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint32_t dstStride = DEFAULT_DATA_COPY_STRIDE;
    uint32_t rsv = 0; // reserved information
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
template <typename T> struct DataCopyPadExtParams {
    using TYPE = typename GetPadValueType<T>::Type;
    __aicore__ DataCopyPadExtParams()
    {
        isPad = false;
        leftPadding = 0;
        rightPadding = 0;
        paddingValue = 0;
    }
    __aicore__ DataCopyPadExtParams(const bool isPadValue, const uint8_t leftPadValue, const uint8_t rightPadValue,
        T padValue)
    {
        isPad = isPadValue;
        leftPadding = leftPadValue;
        rightPadding = rightPadValue;
        paddingValue = *(reinterpret_cast<TYPE *>(&padValue));
    }
    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    TYPE paddingValue = 0;
};
#elif defined (__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
      (__NPU_ARCH__ == 3113))
template <typename T>
struct DataCopyPadExtParams {
    __aicore__ DataCopyPadExtParams() {}

    __aicore__ DataCopyPadExtParams(const bool isPadValue, const uint8_t leftPadValue, const uint8_t rightPadValue,
        T padValue)
        : isPad(isPadValue),
          leftPadding(leftPadValue),
          rightPadding(rightPadValue),
          paddingValue(padValue)
    {}
    __aicore__ inline void SetIsPad(const bool isPad_)
    {
        isPad = isPad_;
    }

    __aicore__ inline void SetLeftPadding(const uint8_t leftPadding_)
    {
        leftPadding = leftPadding_;
    }

    __aicore__ inline void SetRightPadding(const uint8_t rightPadding_)
    {
        rightPadding = rightPadding_;
    }

    __aicore__ inline void SetPaddingValue(const T paddingValue_)
    {
        paddingValue = paddingValue_;
    }

    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    T paddingValue = 0;
};      
#else
template <typename T>
struct DataCopyPadExtParams {
    __aicore__ DataCopyPadExtParams() {}

    __aicore__ DataCopyPadExtParams(const bool isPadValue, const uint8_t leftPadValue, const uint8_t rightPadValue,
        T padValue)
        : isPad(isPadValue),
          leftPadding(leftPadValue),
          rightPadding(rightPadValue),
          paddingValue(padValue)
    {}

    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    T paddingValue = 0;
};
#endif

#if defined (__NPU_ARCH__) && (__NPU_ARCH__ == 3113)
struct Nd2NzParamsL311 {
    __aicore__ Nd2NzParamsL311() {}

    __aicore__ Nd2NzParamsL311(const uint16_t ndNumIn, const uint16_t nValueIn, const uint32_t dValueIn,
        const uint64_t srcNdMatrixStrideIn, const uint64_t srcDValueIn, const uint16_t dstNzC0StrideIn,
        const uint16_t dstNzNStrideIn, const uint32_t dstNzMatrixStrideIn, const uint16_t sidIn = 0)
        : ndNum(ndNumIn),
          nValue(nValueIn),
          dValue(dValueIn),
          srcNdMatrixStride(srcNdMatrixStrideIn),
          srcDValue(srcDValueIn),
          dstNzC0Stride(dstNzC0StrideIn),
          dstNzNStride(dstNzNStrideIn),
          dstNzMatrixStride(dstNzMatrixStrideIn),
          sid(sidIn)
    {}
    __aicore__ inline void SetNdNum(const uint16_t ndNum_)
    {
        ndNum = ndNum_;
    }

    __aicore__ inline void SetNValue(const uint16_t nValue_)
    {
        nValue = nValue_;
    }

    __aicore__ inline void SetDValue(const uint32_t dValue_)
    {
        dValue = dValue_;
    }

    __aicore__ inline void SetSrcNdMatrixStride(const uint64_t srcNdMatrixStride_)
    {
        srcNdMatrixStride = srcNdMatrixStride_;
    }

    __aicore__ inline void SetSrcDValue(const uint64_t srcDValue_)
    {
        srcDValue = srcDValue_;
    }

    __aicore__ inline void SetDstNzC0Stride(const uint16_t dstNzC0Stride_)
    {
        dstNzC0Stride = dstNzC0Stride_;
    }

    __aicore__ inline void SetDstNzNStride(const uint16_t dstNzNStride_)
    {
        dstNzNStride = dstNzNStride_;
    }

    __aicore__ inline void SetDstNzMatrixStride(const uint32_t dstNzMatrixStride_)
    {
        dstNzMatrixStride = dstNzMatrixStride_;
    }

    __aicore__ inline void SetSid(const uint16_t sid_)
    {
        sid = sid_;
    }

    uint16_t ndNum = 0;
    uint16_t nValue = 0;
    uint32_t dValue = 0;
    uint64_t srcNdMatrixStride = 0;
    uint64_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint32_t dstNzMatrixStride = 0;
    uint16_t sid = 0;
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
struct Nd2NzParams {
    __aicore__ Nd2NzParams() {}

    __aicore__ Nd2NzParams(const uint16_t ndNumIn, const uint16_t nValueIn, const uint32_t dValueIn,
        const uint64_t srcNdMatrixStrideIn, const uint64_t srcDValueIn, const uint16_t dstNzC0StrideIn,
        const uint16_t dstNzNStrideIn, const uint32_t dstNzMatrixStrideIn)
        : ndNum(ndNumIn),
          nValue(nValueIn),
          dValue(dValueIn),
          srcNdMatrixStride(srcNdMatrixStrideIn),
          srcDValue(srcDValueIn),
          dstNzC0Stride(dstNzC0StrideIn),
          dstNzNStride(dstNzNStrideIn),
          dstNzMatrixStride(dstNzMatrixStrideIn)
    {}

    uint16_t ndNum = 0;
    uint16_t nValue = 0;
    uint32_t dValue = 0;
    uint64_t srcNdMatrixStride = 0;
    uint64_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint32_t dstNzMatrixStride = 0;
};
#elif defined (__NPU_ARCH__) && ((__NPU_ARCH__ == 3113))
using Nd2NzParams = Nd2NzParamsL311;
#elif defined (__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103))
struct Nd2NzParams {
    __aicore__ Nd2NzParams() {}

    __aicore__ Nd2NzParams(const uint16_t ndNumIn, const uint16_t nValueIn, const uint16_t dValueIn,
        const uint16_t srcNdMatrixStrideIn, const uint16_t srcDValueIn, const uint16_t dstNzC0StrideIn,
        const uint16_t dstNzNStrideIn, const uint16_t dstNzMatrixStrideIn)
        : ndNum(ndNumIn),
          nValue(nValueIn),
          dValue(dValueIn),
          srcNdMatrixStride(srcNdMatrixStrideIn),
          srcDValue(srcDValueIn),
          dstNzC0Stride(dstNzC0StrideIn),
          dstNzNStride(dstNzNStrideIn),
          dstNzMatrixStride(dstNzMatrixStrideIn)
    {}
    __aicore__ inline void SetNdNum(const uint16_t ndNum_)
    {
        ndNum = ndNum_;
    }

    __aicore__ inline void SetNValue(const uint16_t nValue_)
    {
        nValue = nValue_;
    }

    __aicore__ inline void SetDValue(const uint16_t dValue_)
    {
        dValue = dValue_;
    }

    __aicore__ inline void SetSrcNdMatrixStride(const uint16_t srcNdMatrixStride_)
    {
        srcNdMatrixStride = srcNdMatrixStride_;
    }

    __aicore__ inline void SetSrcDValue(const uint16_t srcDValue_)
    {
        srcDValue = srcDValue_;
    }

    __aicore__ inline void SetDstNzC0Stride(const uint16_t dstNzC0Stride_)
    {
        dstNzC0Stride = dstNzC0Stride_;
    }

    __aicore__ inline void SetDstNzNStride(const uint16_t dstNzNStride_)
    {
        dstNzNStride = dstNzNStride_;
    }

    __aicore__ inline void SetDstNzMatrixStride(const uint16_t dstNzMatrixStride_)
    {
        dstNzMatrixStride = dstNzMatrixStride_;
    }

    uint16_t ndNum = 0;
    uint16_t nValue = 0;
    uint16_t dValue = 0;
    uint16_t srcNdMatrixStride = 0;
    uint16_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint16_t dstNzMatrixStride = 0;
};
#else
struct Nd2NzParams {
    __aicore__ Nd2NzParams() {}

    __aicore__ Nd2NzParams(const uint16_t ndNumIn, const uint16_t nValueIn, const uint16_t dValueIn,
        const uint16_t srcNdMatrixStrideIn, const uint16_t srcDValueIn, const uint16_t dstNzC0StrideIn,
        const uint16_t dstNzNStrideIn, const uint16_t dstNzMatrixStrideIn)
        : ndNum(ndNumIn),
          nValue(nValueIn),
          dValue(dValueIn),
          srcNdMatrixStride(srcNdMatrixStrideIn),
          srcDValue(srcDValueIn),
          dstNzC0Stride(dstNzC0StrideIn),
          dstNzNStride(dstNzNStrideIn),
          dstNzMatrixStride(dstNzMatrixStrideIn)
    {}

    uint16_t ndNum = 0;
    uint16_t nValue = 0;
    uint16_t dValue = 0;
    uint16_t srcNdMatrixStride = 0;
    uint16_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint16_t dstNzMatrixStride = 0;
};
#endif

#if defined (__NPU_ARCH__) && ((__NPU_ARCH__ == 3113))
struct Nz2NdParamsFullL311 {
    __aicore__ Nz2NdParamsFullL311() {}

    __aicore__ Nz2NdParamsFullL311(const uint16_t ndNumIn, const uint16_t nValueIn, const uint16_t dValueIn,
        const uint16_t srcNdMatrixStrideIn, const uint16_t srcNStrideIn, const uint32_t dstDStrideIn,
        const uint16_t dstNdMatrixStrideIn, const uint16_t sidIn)
        : ndNum(ndNumIn),
          nValue(nValueIn),
          dValue(dValueIn),
          srcNdMatrixStride(srcNdMatrixStrideIn),
          srcNStride(srcNStrideIn),
          dstDStride(dstDStrideIn),
          dstNdMatrixStride(dstNdMatrixStrideIn),
          sid(sidIn)
    {}
    __aicore__ inline void SetNdNum(const uint16_t ndNum_)
    {
        ndNum = ndNum_;
    }

    __aicore__ inline void SetNValue(const uint16_t nValue_)
    {
        nValue = nValue_;
    }

    __aicore__ inline void SetDValue(const uint16_t dValue_)
    {
        dValue = dValue_;
    }

    __aicore__ inline void SetSrcNdMatrixStride(const uint16_t srcNdMatrixStride_)
    {
        srcNdMatrixStride = srcNdMatrixStride_;
    }

    __aicore__ inline void SetSrcNStride(const uint16_t srcNStride_)
    {
        srcNStride = srcNStride_;
    }

    __aicore__ inline void SetDstDStride(const uint32_t dstDStride_)
    {
        dstDStride = dstDStride_;
    }

    __aicore__ inline void SetDstNdMatrixStride(const uint16_t dstNdMatrixStride_)
    {
        dstNdMatrixStride = dstNdMatrixStride_;
    }

    __aicore__ inline void SetSid(const uint16_t sid_)
    {
        sid = sid_;
    }

    uint16_t ndNum = 1;
    uint16_t nValue = 0;
    uint16_t dValue = 0;
    uint16_t srcNdMatrixStride = 1;
    uint16_t srcNStride = 0;
    uint32_t dstDStride = 0; // u16->u32, N=70016����u16�����
    uint16_t dstNdMatrixStride = 1;
    uint16_t sid = 0;
};
#endif

#if defined (__NPU_ARCH__) && ((__NPU_ARCH__ == 3113))
using Nz2NdParamsFull = Nz2NdParamsFullL311;
#elif defined (__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103))
struct Nz2NdParamsFull {
    __aicore__ Nz2NdParamsFull() {}

    __aicore__ Nz2NdParamsFull(const uint16_t ndNumIn, const uint16_t nValueIn, const uint16_t dValueIn,
        const uint16_t srcNdMatrixStrideIn, const uint16_t srcNStrideIn, const uint16_t dstDStrideIn,
        const uint16_t dstNdMatrixStrideIn)
        : ndNum(ndNumIn),
          nValue(nValueIn),
          dValue(dValueIn),
          srcNdMatrixStride(srcNdMatrixStrideIn),
          srcNStride(srcNStrideIn),
          dstDStride(dstDStrideIn),
          dstNdMatrixStride(dstNdMatrixStrideIn)
    {}
    __aicore__ inline void SetNdNum(const uint16_t ndNum_)
    {
        ndNum = ndNum_;
    }

    __aicore__ inline void SetNValue(const uint16_t nValue_)
    {
        nValue = nValue_;
    }

    __aicore__ inline void SetDValue(const uint16_t dValue_)
    {
        dValue = dValue_;
    }

    __aicore__ inline void SetSrcNdMatrixStride(const uint16_t srcNdMatrixStride_)
    {
        srcNdMatrixStride = srcNdMatrixStride_;
    }

    __aicore__ inline void SetSrcNStride(const uint16_t srcNStride_)
    {
        srcNStride = srcNStride_;
    }

    __aicore__ inline void SetDstDStride(const uint32_t dstDStride_)
    {
        dstDStride = dstDStride_;
    }

    __aicore__ inline void SetDstNdMatrixStride(const uint16_t dstNdMatrixStride_)
    {
        dstNdMatrixStride = dstNdMatrixStride_;
    }

    uint16_t ndNum = 1;
    uint16_t nValue = 0;
    uint16_t dValue = 0;
    uint16_t srcNdMatrixStride = 1;
    uint16_t srcNStride = 0;
    uint16_t dstDStride = 0;
    uint16_t dstNdMatrixStride = 1;
};
#else
struct Nz2NdParamsFull {
    __aicore__ Nz2NdParamsFull() {}

    __aicore__ Nz2NdParamsFull(const uint16_t ndNumIn, const uint16_t nValueIn, const uint16_t dValueIn,
        const uint16_t srcNdMatrixStrideIn, const uint16_t srcNStrideIn, const uint16_t dstDStrideIn,
        const uint16_t dstNdMatrixStrideIn)
        : ndNum(ndNumIn),
          nValue(nValueIn),
          dValue(dValueIn),
          srcNdMatrixStride(srcNdMatrixStrideIn),
          srcNStride(srcNStrideIn),
          dstDStride(dstDStrideIn),
          dstNdMatrixStride(dstNdMatrixStrideIn)
    {}

    uint16_t ndNum = 1;
    uint16_t nValue = 0;
    uint16_t dValue = 0;
    uint16_t srcNdMatrixStride = 1;
    uint16_t srcNStride = 0;
    uint16_t dstDStride = 0;
    uint16_t dstNdMatrixStride = 1;
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
struct Dn2NzParams {
    __aicore__ Dn2NzParams() {}

    __aicore__ Dn2NzParams(const uint16_t dnNumIn, const uint16_t nValueIn, const uint32_t dValueIn,
        const uint64_t srcDnMatrixStrideIn, const uint64_t srcDValueIn, const uint16_t dstNzC0StrideIn,
        const uint16_t dstNzNStrideIn, const uint32_t dstNzMatrixStrideIn)
    {
        dnNum = dnNumIn;
        nValue = nValueIn;
        dValue = dValueIn;
        srcDnMatrixStride = srcDnMatrixStrideIn;
        srcDValue = srcDValueIn;
        dstNzC0Stride = dstNzC0StrideIn;
        dstNzNStride = dstNzNStrideIn;
        dstNzMatrixStride = dstNzMatrixStrideIn;
    }

    uint16_t dnNum = 0;
    uint16_t nValue = 0;
    uint32_t dValue = 0;
    uint64_t srcDnMatrixStride = 0;
    uint64_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint32_t dstNzMatrixStride = 0;
};

struct LoopModeParams {
    __aicore__ LoopModeParams()
    {
        loop1Size = 0;
        loop2Size = 0;
        loop1SrcStride = 0;
        loop1DstStride = 0;
        loop2SrcStride = 0;
        loop2DstStride = 0;
    }

    __aicore__ LoopModeParams(const uint32_t loop1SizeIn, const uint32_t loop2SizeIn, const uint64_t loop1SrcStrideIn,
    const uint64_t loop1DstStrideIn, const uint64_t loop2SrcStrideIn, const uint64_t loop2DstStrideIn)
    {
        loop1Size = loop1SizeIn;
        loop2Size = loop2SizeIn;
        loop1SrcStride = loop1SrcStrideIn;
        loop1DstStride = loop1DstStrideIn;
        loop2SrcStride = loop2SrcStrideIn;
        loop2DstStride = loop2DstStrideIn;
    }

    uint32_t loop1Size = 0;
    uint32_t loop2Size = 0;
    uint64_t loop1SrcStride = 0;
    uint64_t loop1DstStride = 0;
    uint64_t loop2SrcStride = 0;
    uint64_t loop2DstStride = 0;
};
#endif

struct SliceInfo {
    __aicore__ SliceInfo() {}

    __aicore__ SliceInfo(const uint32_t startIndexIn, const uint32_t endIndexIn, const uint32_t strideIn,
        const uint32_t burstLenIn, const uint32_t shapeValueIn = 0)
        : startIndex(startIndexIn),
          endIndex(endIndexIn),
          stride(strideIn),
          burstLen(burstLenIn),
          shapeValue(shapeValueIn)
    {}

    uint32_t startIndex = 0;
    uint32_t endIndex = ONE_BLK_SIZE - 1;
    uint32_t stride = 0;
    uint32_t burstLen = ONE_BLK_SIZE;
    uint32_t shapeValue = 0;
};

struct CopyRepeatParams {
    __aicore__ CopyRepeatParams() {}

    __aicore__ CopyRepeatParams(const uint16_t dstStrideIn, const uint16_t srcStrideIn, uint16_t dstRepeatSizeIn,
        uint16_t srcRepeatSizeIn)
        : dstStride(dstStrideIn),
          srcStride(srcStrideIn),
          dstRepeatSize(dstRepeatSizeIn),
          srcRepeatSize(srcRepeatSizeIn)
    {}

    uint16_t dstStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t dstRepeatSize = DEFAULT_REPEAT_STRIDE;
    uint16_t srcRepeatSize = DEFAULT_REPEAT_STRIDE;
};

#if defined (__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
struct Dn2NzParams {
    __aicore__ Dn2NzParams()
    {
        dnNum = 0;
        nValue = 0;
        dValue = 0;
        srcDnMatrixStride = 0;
        srcDValue = 0;
        dstNzC0Stride = 0;
        dstNzNStride = 0;
        dstNzMatrixStride = 0;
    }
 
    __aicore__ Dn2NzParams(const uint16_t dnNumIn, const uint16_t nValueIn, const uint32_t dValueIn,
        const uint64_t srcDnMatrixStrideIn, const uint64_t srcDValueIn, const uint16_t dstNzC0StrideIn,
        const uint16_t dstNzNStrideIn, const uint32_t dstNzMatrixStrideIn)
    {
        dnNum = dnNumIn;
        nValue = nValueIn;
        dValue = dValueIn;
        srcDnMatrixStride = srcDnMatrixStrideIn;
        srcDValue = srcDValueIn;
        dstNzC0Stride = dstNzC0StrideIn;
        dstNzNStride = dstNzNStrideIn;
        dstNzMatrixStride = dstNzMatrixStrideIn;
    }
 
    __aicore__ inline void SetDnNum(const uint16_t dnNum_)
    {
        dnNum = dnNum_;
    }
 
    __aicore__ inline void SetSrcDnMatrixStride(const uint16_t srcDnMatrixStride_)
    {
        srcDnMatrixStride = srcDnMatrixStride_;
    }
 
    __aicore__ inline void SetSrcDValue(const uint16_t srcDValue_)
    {
        srcDValue = srcDValue_;
    }
 
    __aicore__ inline void SetDstNzC0Stride(const uint16_t dstNzC0Stride_)
    {
        dstNzC0Stride = dstNzC0Stride_;
    }
 
    __aicore__ inline void SetDstNzNStride(const uint16_t dstNzNStride_)
    {
        dstNzNStride = dstNzNStride_;
    }
 
    __aicore__ inline void SetDstNzMatrixStride(const uint16_t dstNzMatrixStride_)
    {
        dstNzMatrixStride = dstNzMatrixStride_;
    }
 
    uint16_t dnNum = 0;
    uint16_t nValue = 0;
    uint32_t dValue = 0;
    uint64_t srcDnMatrixStride = 0;
    uint64_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint32_t dstNzMatrixStride = 0;
};

struct DataCopyAttrParams {
    __aicore__ DataCopyAttrParams()
    {
        fixBufPos = FixBufPos::QUANT_PRE;
    }

    __aicore__ DataCopyAttrParams(const FixBufPos fixBufPosValue)
    {
        fixBufPos = fixBufPosValue;
    }

    __aicore__ inline void SetFixBufPos(const FixBufPos fixBufPos_)
    {
        fixBufPos = fixBufPos_;
    }

    FixBufPos fixBufPos = FixBufPos::QUANT_PRE;
};
#endif
} // namespace AscendC

/* **************************************************************************************************
 * DataCopy(Layout) API Level2                                              *
 * ************************************************************************************************* */
namespace AscendC {

struct DataCopyTrait {};
constexpr DataCopyTrait DEFAULT_DATA_COPY_TRAIT;

}
#endif // ASCENDC_MODULE_STRUCT_DATA_COPY_H