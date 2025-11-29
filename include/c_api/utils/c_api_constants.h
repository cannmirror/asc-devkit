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
 * \file c_api_constants.h
 * \brief
 */

#ifndef INCLUDE_C_API_UTILS_C_API_CONSTANTS_H
#define INCLUDE_C_API_UTILS_C_API_CONSTANTS_H

#include <cstdint>
#include <type_traits>

#ifndef __aicore__
#define __aicore__ [aicore]
#endif // __aicore__

/*
 * VALUE_INDEX = 0;
 * INDEX_VALUE = 1;
 * ONLY_VALUE = 2;
 * ONLY_INDEX = 3;
*/
using order_t = Order_t;

template<pipe_t PIPE>
using PipeType = std::integral_constant<pipe_t, PIPE>;

#define PIPE_TYPE_S PipeType<PIPE_S>{}
#define PIPE_TYPE_V PipeType<PIPE_V>{}
#define PIPE_TYPE_M PipeType<PIPE_M>{}
#define PIPE_TYPE_MTE1 PipeType<PIPE_MTE1>{}
#define PIPE_TYPE_MTE2 PipeType<PIPE_MTE2>{}
#define PIPE_TYPE_MTE3 PipeType<PIPE_MTE3>{}
#define PIPE_TYPE_ALL PipeType<PIPE_ALL>{}
#define PIPE_TYPE_FIX PipeType<PIPE_FIX>{}

constexpr uint16_t CAPI_DATABLOCK_NUM = 8;
constexpr uint16_t CAPI_ONE_DATABLOCK_SIZE = 32;

constexpr uint8_t CAPI_DEFAULT_REPEAT = 1;
constexpr uint8_t CAPI_DEFAULT_BLOCK_STRIDE = 1;
constexpr uint8_t CAPI_DEFAULT_REPEAT_STRIDE = 8;
constexpr uint64_t CAPI_DEFAULT_BINARY_CONFIG_VALUE = 0x0100080808010101;
constexpr uint64_t CAPI_DEFAULT_UNARY_CONFIG_VALUE = 0x0100800800010001;
constexpr uint64_t CAPI_DEFAULT_REDUCE_CONFIG_VALUE = 0x0100000800010001;
constexpr uint64_t CAPI_DEFAULT_DUPLICATE_CONFIG_VALUE = 0x0100800800010001;

#endif