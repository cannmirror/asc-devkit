/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef C_API_INSTR_VECTOR_COMPUTE_GET_MAX_MIN_CNT_ASC_2201_GET_MAX_MIN_CNT_IMPL_H
#define C_API_INSTR_VECTOR_COMPUTE_GET_MAX_MIN_CNT_ASC_2201_GET_MAX_MIN_CNT_IMPL_H

namespace CApiInternal {

constexpr uint8_t offset = 32;

__aicore__ inline void get_reduce_max_min_cnt_impl(half& val, uint32_t& index)
{
    int64_t max_min_cnt = get_max_min_cnt();
    union {
        half h;
        uint16_t u;
    } u162half = {.u = static_cast<uint16_t>(0xffff & max_min_cnt)};
    val = u162half.h;
    index = 0xffffffff & (max_min_cnt >> offset);
}

__aicore__ inline void get_reduce_max_min_cnt_impl(float& val, uint32_t& index)
{
    int64_t max_min_cnt = get_max_min_cnt();
    union {
        float f;
        uint32_t u;
    } u322float = {.u = static_cast<uint32_t>(0xffffffff & max_min_cnt)};
    val = u322float.f;
    index = 0xffffffff & (max_min_cnt >> offset);
}

} // namespace CApiInternal

#endif