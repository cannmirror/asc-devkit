/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file tan_tiling_intf.h
 * \brief
 */
#ifndef AICORE_ADV_API_MATH_TAN_TILING_INTF_H
#define AICORE_ADV_API_MATH_TAN_TILING_INTF_H

#include "tan_tiling.h"

namespace AscendC {
[[deprecated(__FILE__ " is deprecated, please use tan_tiling.h instead!")]] typedef void TanTilingDeprecatedHeader;
using LibTanTilingInterface = TanTilingDeprecatedHeader;
} // namespace AscendC
#endif // AICORE_ADV_API_MATH_TAN_TILING_INTF_H