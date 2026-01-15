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
 * \file #inlcude "broadcast_common_utils.h"
 * \brief
 */
#ifndef IMPL_PAD_BROADCAST_BROADCAST_COMMON_UTILS_H
#define IMPL_PAD_BROADCAST_BROADCAST_COMMON_UTILS_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"

namespace AscendC {

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)

struct BroadcastTiling {
    uint32_t oriRank;
    uint32_t rank;
    uint32_t dstSize;
    uint32_t srcSize;
    uint32_t loopNum = 0;
    uint32_t oriSrcShape[9];
    uint32_t oriDstShape[9];
    uint32_t dstShape[9];
    uint32_t dstStride[9];
    uint32_t srcStride[10];
};
namespace BroadcastInternal {
constexpr uint32_t U16_MAX = 65536;

/*
    ExtractSignedTypeBySize is to calculate gather
    index offset according to the type size.
    Because b64 would be converted into b32 and
    with b8's situation, the index should be calculated
    using b16, they should be handled specially.
*/
template <uint32_t size = sizeof(int8_t)> struct ExtractSignedTypeBySize {
    using T = int16_t;
};

template <> struct ExtractSignedTypeBySize<sizeof(int16_t)> {
    using T = int16_t;
};

template <> struct ExtractSignedTypeBySize<sizeof(int32_t)> {
    using T = int32_t;
};

template <> struct ExtractSignedTypeBySize<sizeof(int64_t)> {
    using T = int32_t;
};

/*
    ExtractUnsignedTypeBySize is used for broadcast
    according to the type size.
    Because b64 would be converted into b32, it
    should be handled specially.
*/
template <uint32_t size = sizeof(uint8_t)> struct ExtractUnsignedTypeBySize {
    using T = uint8_t;
};

template <> struct ExtractUnsignedTypeBySize<sizeof(uint16_t)> {
    using T = uint16_t;
};

template <> struct ExtractUnsignedTypeBySize<sizeof(uint32_t)> {
    using T = uint32_t;
};

template <> struct ExtractUnsignedTypeBySize<sizeof(uint64_t)> {
    using T = uint32_t;
};

/*
    ExtractIndexTypeBySize is used for broadcast
    indexub according to the type size.
    Because b8 would be treated as b16 to read from srcub, it
    should be handled specially.
*/
template <uint32_t size = sizeof(uint8_t)> struct ExtractIndexTypeBySize {
    using T = uint16_t;
};

template <> struct ExtractIndexTypeBySize<sizeof(uint16_t)> {
    using T = uint16_t;
};

template <> struct ExtractIndexTypeBySize<sizeof(uint32_t)> {
    using T = uint32_t;
};

template <> struct ExtractIndexTypeBySize<sizeof(uint64_t)> {
    using T = uint32_t;
};

__aicore__ inline void DstShapeCheck(const uint32_t *dstShape, uint32_t dim)
{
    for (uint16_t i = 0; i < dim; ++i) {
        ASCENDC_ASSERT((dstShape[i] <= U16_MAX), { KERNEL_LOG(KERNEL_ERROR, "shape should be less than uint16 max"); });
    }
}

__aicore__ inline void ShapeCheck(uint32_t *tillingShape, const uint32_t *shape, uint32_t rank)
{
    for (uint16_t i = 0; i < rank; ++i) {
        ASCENDC_ASSERT(
            (shape[i] == tillingShape[i]), { KERNEL_LOG(KERNEL_ERROR, "Tilling shape should be equal to shape!"); });
    }
}
}
#endif

constexpr uint32_t ONE_VOR_BLOCK_DIM = 8;
constexpr uint32_t ELEMENT_NUM_FOR_UINT16 = 16;
constexpr int32_t FLOAT_ELEMENT_NUM = 2;
constexpr uint32_t REPEAT_STRIDE_NUM = 8;
constexpr uint32_t MAX_REPEAT_NUM = 255;

template <typename T, bool isReuseSource = false>
__aicore__ inline void TwoDimBroadCastDimAlign(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<T> &zeroTemp, const uint32_t firstDim, const uint32_t blockDim)
{
    int32_t dtypeCount = 1;
    if constexpr (sizeof(T) == sizeof(float)) {
        dtypeCount = FLOAT_ELEMENT_NUM;
    }
    uint32_t orCounts = firstDim / ONE_VOR_BLOCK_DIM;
    constexpr uint32_t oneBlockElementNum = ONE_BLK_SIZE / sizeof(T);
    uint8_t repeateTimes = blockDim / oneBlockElementNum;
    SetMaskNorm();
    SetVectorMask<uint16_t, MaskMode::NORMAL>(ONE_VOR_BLOCK_DIM * ELEMENT_NUM_FOR_UINT16);
    uint8_t dstBlkStride = blockDim * dtypeCount / ELEMENT_NUM_FOR_UINT16;
    BinaryRepeatParams binaryParams(dstBlkStride, 0, 0, 1, 1, 0);
    uint32_t transTmpBufferOffset = 0;
    for (uint32_t i = 0; i < orCounts; i++) {
        Or<uint16_t, false>(dstLocal[transTmpBufferOffset].template ReinterpretCast<uint16_t>(),
            srcLocal.template ReinterpretCast<uint16_t>(),
            zeroTemp.template ReinterpretCast<uint16_t>(),
            MASK_PLACEHOLDER,
            repeateTimes,
            binaryParams);
        transTmpBufferOffset += ONE_VOR_BLOCK_DIM * blockDim;
    }
    uint32_t orCountsTail = firstDim - orCounts * ONE_VOR_BLOCK_DIM;
    if (orCountsTail > 0) {
        SetMaskNorm();
        SetVectorMask<uint16_t, MaskMode::NORMAL>(orCountsTail * ELEMENT_NUM_FOR_UINT16);
        Or<uint16_t, false>(dstLocal[transTmpBufferOffset].template ReinterpretCast<uint16_t>(),
                            srcLocal.template ReinterpretCast<uint16_t>(),
                            zeroTemp.template ReinterpretCast<uint16_t>(),
                            MASK_PLACEHOLDER,
                            repeateTimes,
                            binaryParams);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void LoopBroadCast(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<T> &zeroTemp, const uint32_t firstDim, const uint32_t blockDim)
{
    int32_t dtypeCount = 1;
    if constexpr (sizeof(T) == sizeof(float)) {
        dtypeCount = FLOAT_ELEMENT_NUM;
    }
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(firstDim * dtypeCount);
    BinaryRepeatParams binaryParams(1, 1, 0, REPEAT_STRIDE_NUM, REPEAT_STRIDE_NUM, 0);
    uint32_t temBufferOffset = 0;
    for (uint32_t i = 0; i < blockDim; i++) {
        Or<uint16_t, false>(dstLocal[temBufferOffset].template ReinterpretCast<uint16_t>(),
            srcLocal.template ReinterpretCast<uint16_t>(),
            zeroTemp.template ReinterpretCast<uint16_t>(),
            MASK_PLACEHOLDER,
            1,
            binaryParams);
        temBufferOffset += firstDim;
    }
    PipeBarrier<PIPE_V>();
}

}  // namespace AscendC

#endif  // IMPL_PAD_BROADCAST_BROADCAST_COMMON_UTILS_H