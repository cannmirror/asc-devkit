/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INCLUDE_C_API_UTILS_C_API_UNION_H
#define INCLUDE_C_API_UTILS_C_API_UNION_H

#include <cstdint>
#include "c_api_constants.h"

union asc_copy_config {
    uint64_t config;
    struct {
        uint64_t sid : 4;
        uint64_t n_burst : 12;
        uint64_t burst_len : 16;
        uint64_t src_gap : 16;
        uint64_t dst_gap : 16;
    };
};

union asc_binary_config {
    uint64_t config = CAPI_DEFAULT_BINARY_CONFIG_VALUE;
    struct {
        uint64_t dst_block_stride : 8;
        uint64_t src0_block_stride : 8;
        uint64_t src1_block_stride : 8;
        uint64_t dst_repeat_stride : 8;
        uint64_t src0_repeat_stride : 8;
        uint64_t src1_repeat_stride : 8;
        uint64_t reserved : 8;
        uint64_t repeat : 8;
    };
};

union asc_unary_config {
    uint64_t config = CAPI_DEFAULT_UNARY_CONFIG_VALUE;
    struct {
        uint64_t dst_block_stride : 16;
        uint64_t src_block_stride : 16;
        uint64_t dst_repeat_stride : 12;
        uint64_t src_repeat_stride : 12;
        uint64_t repeat : 8;
    };
};

union asc_block_reduce_config {
    uint64_t config = CAPI_DEFAULT_REDUCE_CONFIG_VALUE;
    struct {
        uint64_t dst_repeat_stride : 16;
        uint64_t src_block_stride : 16;
        uint64_t src_repeat_stride : 16;
        uint64_t reserved : 8;
        uint64_t repeat : 8;
    };
};

union asc_repeat_reduce_config {
    uint64_t config = CAPI_DEFAULT_REDUCE_CONFIG_VALUE;
    struct {
        uint64_t dst_repeat_stride : 16;
        uint64_t src_block_stride : 16;
        uint64_t src_repeat_stride : 16;
        uint64_t reserved : 8;
        uint64_t repeat : 8;
    };
};

union asc_duplicate_config {
    uint64_t config = CAPI_DEFAULT_DUPLICATE_CONFIG_VALUE;
    struct {
        uint64_t dst_block_stride : 16;
        uint64_t src_block_stride : 16;
        uint64_t dst_repeat_stride : 12;
        uint64_t src_repeat_stride : 12;
        uint64_t repeat : 8;
    };
};

union asc_brcb_config {
    uint64_t config;
    struct {
        uint64_t dst_block_stride : 16;
        uint64_t reserved1 : 16;
        uint64_t dst_repeat_stride : 12;
        uint64_t reserved2 : 12;
        uint64_t repeat : 8;
    };
};

#endif