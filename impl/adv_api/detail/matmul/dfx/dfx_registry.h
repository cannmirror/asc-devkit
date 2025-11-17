/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file dfx_registry.h
 * \brief
 */


#ifndef MATMUL_DFX_REGISTRY_H
#define MATMUL_DFX_REGISTRY_H

#include "dfx_proxy.h"

namespace AscendC {
namespace Impl {
namespace Detail {
    MATMUL_DFX_PROXY_REGISTER(InputL1Cache, ClearAL1Cache, ClearBL1Cache);
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _DFX_REGISTRY_H_
