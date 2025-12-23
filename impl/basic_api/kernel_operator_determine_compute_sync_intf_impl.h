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
 * \file kernel_operator_determine_compute_sync_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_INTF_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_INTF_IMPL_H

#if __NPU_ARCH__ == 1001
#include "dav_c100/kernel_operator_determine_compute_sync_impl.h"
#elif __NPU_ARCH__ == 2002
#include "dav_m200/kernel_operator_determine_compute_sync_impl.h"
#elif __NPU_ARCH__ == 2201
#include "dav_c220/kernel_operator_determine_compute_sync_impl.h"
#elif __NPU_ARCH__ == 3002
#include "dav_m300/kernel_operator_determine_compute_sync_impl.h"
#elif __NPU_ARCH__ == 3102
#include "dav_m310/kernel_operator_determine_compute_sync_impl.h"
#elif __NPU_ARCH__ == 3101
#include "dav_c310/kernel_operator_determine_compute_sync_impl.h"
#elif (__NPU_ARCH__ == 5102)
#include "dav_m510/kernel_operator_determine_compute_sync_impl.h"
#elif __NPU_ARCH__ == 3003
#include "dav_l300/kernel_operator_determine_compute_sync_impl.h"
#elif __NPU_ARCH__ == 3113
#include "dav_l311/kernel_operator_determine_compute_sync_impl.h"
#endif

namespace AscendC {
/*
 * @ingroup WaitPreBlock
 * @brief wait previous compute finish
 * @param [in] global memory workspace
 */
__aicore__ inline void InitDetermineComputeWorkspace(GlobalTensor<int32_t> &gmWorkspace,
    LocalTensor<int32_t> &ubWorkspace)
{
    InitDetermineComputeWorkspaceCalc(gmWorkspace, ubWorkspace);
}
/*
 * @ingroup WaitPreBlock
 * @brief wait previous compute finish
 * @param [in] global memory workspace
 */
__aicore__ inline void WaitPreBlock(GlobalTensor<int32_t> &gmWorkspace, LocalTensor<int32_t> &ubWorkspace)
{
    WaitPreBlockCalc(gmWorkspace, ubWorkspace);
}

/*
 * @ingroup NotifyNextBlock
 * @brief wait previous compute finish
 * @param [in] global memory workspace
 */
__aicore__ inline void NotifyNextBlock(GlobalTensor<int32_t> &gmWorkspace, LocalTensor<int32_t> &ubWorkspace)
{
    NotifyNextBlockCalc(gmWorkspace, ubWorkspace);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_INTF_IMPL_H
