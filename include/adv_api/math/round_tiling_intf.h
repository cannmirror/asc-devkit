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
 * \file round_tiling.h
 * \brief
 */

#if !defined(_ASCENDC_INCLUDE_INTERNAL_HEADERS_)
#define _ASCENDC_INCLUDE_INTERNAL_HEADERS_
#define UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ROUND_TILING_INTF_H
#endif

#ifndef LIB_MATH_ROUND_INTF_H
#define LIB_MATH_ROUND_INTF_H

#include "round_tiling.h"
namespace AscendC {
[[deprecated(__FILE__ " is deprecated, please use round_tiling.h instead!")]] typedef void using_deprecated_header_h;
using LIB_MATH_ROUND = using_deprecated_header_h;
using LibMathRoundTilingInterface = using_deprecated_header_h;
}
#endif // LIB_MATH_ROUND_INTF

#if defined(UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ROUND_TILING_INTF_H)
#undef _ASCENDC_INCLUDE_INTERNAL_HEADERS_
#undef UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ROUND_TILING_INTF_H
#endif