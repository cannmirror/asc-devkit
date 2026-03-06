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
 * \file kernel_reg_compute_impl_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_REG_COMPUTE_IMPL_INTF_H
#define ASCENDC_MODULE_REG_COMPUTE_IMPL_INTF_H

#include "kernel_reg_compute_addrreg_impl.h"
#include "kernel_reg_compute_common_impl.h"
#include "kernel_reg_compute_copy_impl.h"
#include "kernel_reg_compute_datacopy_impl.h"
#include "kernel_reg_compute_gather_mask_impl.h"
#include "kernel_reg_compute_maskreg_impl.h"
#include "kernel_reg_compute_pack_impl.h"
#include "kernel_reg_compute_vec_binary_impl.h"
#include "kernel_reg_compute_vec_binary_scalar_impl.h"
#include "kernel_reg_compute_vec_cmpsel_impl.h"
#include "kernel_reg_compute_vec_createvecindex_impl.h"
#include "kernel_reg_compute_vec_duplicate_impl.h"
#include "kernel_reg_compute_vec_reduce_impl.h"
#include "kernel_reg_compute_vec_ternary_scalar_impl.h"
#include "kernel_reg_compute_vec_unary_impl.h"
#include "kernel_reg_compute_vec_vconv_impl.h"
#include "kernel_reg_compute_vec_fused_impl.h"
#include "kernel_reg_compute_histograms_impl.h"

#endif // ASCENDC_MODULE_REG_COMPUTE_IMPL_INTF_H