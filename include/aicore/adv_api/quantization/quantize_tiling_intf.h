/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantize_tiling_intf.h
 * \brief
 */
#ifndef AICORE_ADV_API_QUANTIZATION_QUANTIZE_TILING_INTF_H
#define AICORE_ADV_API_QUANTIZATION_QUANTIZE_TILING_INTF_H

#include "quantize_tiling.h"
namespace AscendC {
[[deprecated(
    __FILE__ " is deprecated, please use quantize_tiling.h instead!")]] typedef void QuantizeTilingDeprecatedHeader;
using LibQuantizeTilingInterface = QuantizeTilingDeprecatedHeader;
} // namespace AscendC
#endif // LIB_QUANTIZATION_ASCEND_QUANT_TILING_INTF_H