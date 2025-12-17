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
 * \file kernel_struct_binary.h
 * \brief
 */
#ifndef ASCENDC_MODULE_STRUCT_BINARY_H
#define ASCENDC_MODULE_STRUCT_BINARY_H
#include "utils/kernel_utils_constants.h"

namespace AscendC {
struct BinaryRepeatParams {
    __aicore__ BinaryRepeatParams() {}

    __aicore__ BinaryRepeatParams(const uint8_t dstBlkStrideIn, const uint8_t src0BlkStrideIn,
        const uint8_t src1BlkStrideIn, const uint8_t dstRepStrideIn, const uint8_t src0RepStrideIn,
        const uint8_t src1RepStrideIn)
        : dstBlkStride(dstBlkStrideIn),
          src0BlkStride(src0BlkStrideIn),
          src1BlkStride(src1BlkStrideIn),
          dstRepStride(dstRepStrideIn),
          src0RepStride(src0RepStrideIn),
          src1RepStride(src1RepStrideIn)
    {}

    uint32_t blockNumber = DEFAULT_BLK_NUM;
    uint8_t dstBlkStride = DEFAULT_BLK_STRIDE;
    uint8_t src0BlkStride = DEFAULT_BLK_STRIDE;
    uint8_t src1BlkStride = DEFAULT_BLK_STRIDE;
    uint8_t dstRepStride = DEFAULT_REPEAT_STRIDE;
    uint8_t src0RepStride = DEFAULT_REPEAT_STRIDE;
    uint8_t src1RepStride = DEFAULT_REPEAT_STRIDE;
    bool repeatStrideMode = false;
    bool strideSizeMode = false;
};
} // namespace AscendC
#endif // ASCENDC_MODULE_STRUCT_BINARY_H
