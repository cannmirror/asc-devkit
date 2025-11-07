/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef C_API_INSTR_VECTOR_COMPUTE_GET_VMS4_SR_ASC_2201_GET_VMS4_SR_IMPL_H
#define C_API_INSTR_VECTOR_COMPUTE_GET_VMS4_SR_ASC_2201_GET_VMS4_SR_IMPL_H

namespace CApiInternal {

__aicore__ inline void asc_GetVms4Sr(uint16_t sortedNum[4])
{
    uint64_t vms4_sr = (uint64_t)get_vms4_sr();
    sortedNum[0] = 0xffff & vms4_sr;
    sortedNum[1] = 0xffff & (vms4_sr >> 16);
    sortedNum[2] = 0xffff & (vms4_sr >> 32);
    sortedNum[3] = 0xffff & (vms4_sr >> 48);
}

} // namespace CApiInternal

__aicore__ inline void asc_GetVms4Sr(uint16_t sortedNum[4])
{
    CApiInternal::asc_GetVms4Sr(sortedNum);
}

#endif