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
 * \file simt_common_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULES_SIMT_C_COMMON_INTERFACE_H
#define ASCENDC_MODULES_SIMT_C_COMMON_INTERFACE_H

#include "simt_utils.h"

__simt_callee__ inline int32_t asc_get_block_idx();

__simt_callee__ inline int32_t asc_get_block_num();

#include "impl/simt_api/simt_common_intf_impl.h"
#endif
