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
 * \file confusion_transpose_tiling.h
 * \brief
 */
#ifndef LIB_TRANSPOSE_CONFUSION_TRANSPOSE_TILING_H
#define LIB_TRANSPOSE_CONFUSION_TRANSPOSE_TILING_H
#include "graph/tensor.h"
#include "confusion_transpose_tilingdata.h"
namespace AscendC {
constexpr uint32_t TWO_TIMES = 2;
#ifndef __NPU_DEVICE__

#ifndef ASCC_PARAM_BLOCK_CUBE
#define ASCC_PARAM_BLOCK_CUBE
constexpr uint32_t BLOCK_CUBE = 16;
#endif

#ifndef ASCC_PARAM_ONE_BLK_SIZE
#define ASCC_PARAM_ONE_BLK_SIZE
constexpr uint32_t ONE_BLK_SIZE = 32;
#endif

#ifndef ASCC_PARAM_CUBE_MAX_SIZE
#define ASCC_PARAM_CUBE_MAX_SIZE
constexpr int32_t CUBE_MAX_SIZE = 256;
#endif

#else // #ifdef __NPU_DEVICE__

#ifndef ASCC_PARAM_BLOCK_CUBE
#define ASCC_PARAM_BLOCK_CUBE
const int32_t BLOCK_CUBE = 16;
#endif

#ifndef ASCC_PARAM_ONE_BLK_SIZE
#define ASCC_PARAM_ONE_BLK_SIZE
const uint16_t ONE_BLK_SIZE = 32;
#endif

#ifndef ASCC_PARAM_CUBE_MAX_SIZE
#define ASCC_PARAM_CUBE_MAX_SIZE
const int32_t CUBE_MAX_SIZE = 256;
#endif

#endif // __NPU_DEVICE__
/*!
 * \brief calculate max and min tmp buffer size for Transpose interface.
   tmp buffer size is a input for GetConfusionTransposeTilingInfo
 *
 * \param [in] srcShape input shape
 * \param [in] typeSize data type size: sizeof(TYPE)
 * \param [in] transposeTypeIn transpose type.
 * \param [out] maxValue max size of tmp buffer
 * \param [out] minValue min size of tmp buffer
 */
[[deprecated("NOTICE: GetConfusionTransposeMaxMinTmpSize has been deprecated and will be removed in the next version. "
             "please do not use it!")]]
void GetConfusionTransposeMaxMinTmpSize(const ge::Shape &srcShape, const uint32_t typeSize,
    const uint32_t transposeTypeIn, uint32_t &maxValue, uint32_t &minValue);

void GetTransposeMaxMinTmpSize(const ge::Shape &srcShape, const uint32_t typeSize,
    const uint32_t transposeTypeIn, uint32_t &maxValue, uint32_t &minValue);

/*!
 * \brief calculate tiling params for Transpose interface
 *
 * \note stackBufferSize should be greater than min tmpSize from GetTransposeMaxMinTmpSize
 *
 * \param [in] srcShape input shape
 * \param [in] stackBufferSize input stack buffer size in uint of Byte, used as tmp buffer size for tiling
 * \param [in] typeSize data type size: sizeof(TYPE)
 * \param [in] transposeTypeIn transpose type.
 * \param [out] tiling Transpose tiling
 */
[[deprecated("NOTICE: GetConfusionTransposeTilingInfo has been deprecated and will be removed in the next version. "
             "please do not use it!")]]
void GetConfusionTransposeTilingInfo(const ge::Shape &srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
    const uint32_t transposeTypeIn, optiling::ConfusionTransposeTiling &tiling);

void GetTransposeTilingInfo(const ge::Shape &srcShape, const uint32_t stackBufferSize, const uint32_t typeSize,
    const uint32_t transposeTypeIn, optiling::ConfusionTransposeTiling &tiling);
[[deprecated("NOTICE: GetConfusionTransposeOnlyTilingInfo has been deprecated and will be removed in the next version. "
             "please do not use it!")]]
void GetConfusionTransposeOnlyTilingInfo(const ge::Shape &srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, optiling::ConfusionTransposeTiling &tiling);
}
#endif // LIB_TRANSPOSE_CONFUSION_TRANSPOSE_TILING_H
