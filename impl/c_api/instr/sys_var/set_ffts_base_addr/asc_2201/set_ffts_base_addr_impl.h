/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef C_API_INSTR_SYS_VAR_SET_FFTS_BASE_ADDR_ASC_2201_SET_FFTS_BASE_ADDR_IMPL_H
#define C_API_INSTR_SYS_VAR_SET_FFTS_BASE_ADDR_ASC_2201_SET_FFTS_BASE_ADDR_IMPL_H

namespace CApiInternal {

__aicore__ inline void asc_SetFftsBaseAddr(uint64_t config)
{
    set_ffts_base_addr(config);
}

} // namespace CApiInternal

#endif