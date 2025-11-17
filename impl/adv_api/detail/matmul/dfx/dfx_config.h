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
 * \file dfx_config.h
 * \brief
 */

#ifndef MATMUL_DFX_CONFIG_H
#define MATMUL_DFX_CONFIG_H

#include "handlers/dfx_chain_handler.h"
#include "dfx_func_info.h"

namespace AscendC {
namespace Impl {
namespace Detail {
struct DfxConfig {
    static constexpr bool ENABLE = false;
    using EnabledHandlers = DfxChainHandler <>;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _DFX_CONFIG_H_
