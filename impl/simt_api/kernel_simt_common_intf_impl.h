/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file kernel_simt_common_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_SIMT_COMMON_INTERFACE_IMPL_H
#define ASCENDC_MODULE_SIMT_COMMON_INTERFACE_IMPL_H

#include "simt_api/dav_c310/kernel_simt_impl.h"

namespace AscendC {
namespace Simt {

__aicore__ inline int32_t GetWarpSize()
{
    return GetWarpSizeImpl();
}

template <int32_t dim>
__aicore__ inline int32_t GetThreadNum()
{
    return GetThreadNumImpl<dim>();
}

template <int32_t dim>
__aicore__ inline int32_t GetThreadIdx()
{
    return GetThreadIdxImpl<dim>();
}

__aicore__ inline int32_t GetBlockIdx()
{
    return GetBlockIdxImpl();
}

__aicore__ inline int32_t GetBlockNum()
{
    return GetBlockNumImpl();
}
}  // namespace Simt
}  // namespace AscendC

#endif  // ASCENDC_MODULE_SIMT_COMMON_INTERFACE_IMPL_H
