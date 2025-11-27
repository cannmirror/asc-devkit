/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef C_API_INSTR_VECTOR_COMPUTE_GET_VMS4_SR_ASC_2201_GET_VMS4_SR_IMPL_H
#define C_API_INSTR_VECTOR_COMPUTE_GET_VMS4_SR_ASC_2201_GET_VMS4_SR_IMPL_H

namespace CApiInternal {

__aicore__ inline void get_vms4_sr_impl(uint16_t sorted_num[4])
{
    uint8_t offset_step = 16;
    uint64_t vms4_sr = (uint64_t)get_vms4_sr();
    sorted_num[0] = 0xffff & vms4_sr;
    sorted_num[1] = 0xffff & (vms4_sr >> offset_step); // >> 16
    sorted_num[2] = 0xffff & (vms4_sr >> (offset_step * 2)); // >> 32
    sorted_num[3] = 0xffff & (vms4_sr >> (offset_step * 3)); // >> 48
}

} // namespace CApiInternal

#endif