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
 * \file kernel_micro_common_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_MICRO_COMMON_INTERFACE_IMPL_H
#define ASCENDC_KERNEL_MICRO_COMMON_INTERFACE_IMPL_H

#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102) || \
    (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3113) || defined(__ASC_NPU_HOST__)
#include "micro_api/kernel_micro_maskreg_intf.h"
#if __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_common_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_common_impl.h"
#elif __NPU_ARCH__ == 5102
#include "micro_api/dav_m510/kernel_micro_common_impl.h"
#else
#include "micro_api/dav_c310/kernel_micro_common_impl.h"
#endif
#endif

#endif // ASCENDC_KERNEL_MICRO_COMMON_INTERFACE_IMPL_H