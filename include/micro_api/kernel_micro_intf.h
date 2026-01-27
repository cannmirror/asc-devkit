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
 * \file kernel_micro_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_INTERFACE_H
#define ASCENDC_MODULE_MICRO_INTERFACE_H

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 3003) || \
    ((__NPU_ARCH__ == 3113))) || defined(__ASC_NPU_HOST__)
#include "kernel_micro_common_intf.h"
#include "kernel_micro_maskreg_intf.h"
#include "kernel_micro_addrreg_intf.h"
#include "kernel_micro_vec_duplicate_intf.h"
#include "kernel_micro_vec_cmpsel_intf.h"
#include "kernel_micro_vec_binary_intf.h"
#include "kernel_micro_vec_binary_scalar_intf.h"
#include "kernel_micro_copy_intf.h"
#include "kernel_micro_datacopy_intf.h"
#include "kernel_micro_gather_mask_intf.h"
#include "kernel_micro_pack_intf.h"
#include "kernel_micro_vec_arange_intf.h"
#include "kernel_micro_vec_reduce_intf.h"
#include "kernel_micro_vec_ternary_scalar_intf.h"
#include "kernel_micro_vec_unary_intf.h"
#include "kernel_micro_vec_vconv_intf.h"
#include "kernel_micro_vec_fused_intf.h"
#include "kernel_micro_histograms_intf.h"
#endif
#endif // ASCENDC_MODULE_MICRO_INTERFACE_H