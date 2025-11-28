/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
 
#ifndef IMPL_C_API_INSTR_SIMD_ATOMIC_DCCI_ASC_2201_DCCI_IMPL_H
#define IMPL_C_API_INSTR_SIMD_ATOMIC_DCCI_ASC_2201_DCCI_IMPL_H
 
namespace CApiInternal {
 
__aicore__ inline void data_cache_clean_and_invalid_impl(__gm__ void* dst, uint64_t entire)
{
    dcci(dst, entire);
}
 
} // namespace CApiInternal
 
#endif