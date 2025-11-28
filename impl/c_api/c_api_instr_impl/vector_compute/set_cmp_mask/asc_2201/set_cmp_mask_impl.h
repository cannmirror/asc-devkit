/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef IMPL_C_API_INSTR_VECTOR_COMPUTE_SET_CMP_MASK_ASC_2201_SET_CMP_MASK_IMPL_H
#define IMPL_C_API_INSTR_VECTOR_COMPUTE_SET_CMP_MASK_ASC_2201_SET_CMP_MASK_IMPL_H

namespace CApiInternal {

__aicore__ inline void set_cmp_mask_impl(__ubuf__ void *sel_mask)
{
    set_cmpmask(sel_mask);
}

} // namespace CApiInternal

#endif