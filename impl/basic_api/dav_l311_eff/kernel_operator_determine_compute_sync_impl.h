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

/*!
 * \file kernel_operator_determine_compute_sync_impl.h
 * \brief
 */
#include "kernel_operator_common_intf.h"
#ifndef ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_IMPL_H

namespace AscendC {
__aicore__ inline void InitDetermineComputeWorkspaceCalc(GlobalTensor<int32_t> &gmWorkspace,
    LocalTensor<int32_t> &ubWorkspace)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported InitDetermineComputeWorkspace on this version"); });
}

__aicore__ inline void WaitPreBlockCalc(const GlobalTensor<int32_t> &gmWorkspace, LocalTensor<int32_t> &ubWorkspace)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported WaitPreBlock on this version"); });
}

__aicore__ inline void NotifyNextBlockCalc(GlobalTensor<int32_t> &gmWorkspace, LocalTensor<int32_t> &ubWorkspace)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported NotifyNextBlock on this version"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_IMPL_H