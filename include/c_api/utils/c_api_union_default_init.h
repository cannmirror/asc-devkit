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
 * \file c_api_union_default_init.h
 * \brief
 */

#ifndef INCLUDE_C_API_UTILS_C_API_UNION_DEFAULT_INIT_H
#define INCLUDE_C_API_UTILS_C_API_UNION_DEFAULT_INIT_H

#include <cstdint>

constexpr asc_binary_config CAPI_DEFAULT_BINARY_CFG{};
constexpr asc_unary_config CAPI_DEFAULT_UNARY_CFG{};
constexpr asc_block_reduce_config CAPI_BLOCK_DEFAULT_REDUCE_CFG{};
constexpr asc_repeat_reduce_config CAPI_REPEAT_DEFAULT_REDUCE_CFG{};
constexpr asc_duplicate_config CAPI_DEFAULT_DUPLICATE_CFG{};

#endif
