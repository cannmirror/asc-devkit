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
 * \file simt_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_SIMT_C_INTF_IMPL_H
#define ASCENDC_MODULE_SIMT_C_INTF_IMPL_H

#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
#include "simt_atomic_intf_impl.h"
#include "simt_common_intf_impl.h"
#include "simt_transcendental_intf_impl.h"
#include "simt_warp_level_intf_impl.h"
#include "simt_bessel_intf_impl.h"
#endif

#endif  // ASCENDC_MODULE_SIMT_INTF_IMPL_H
