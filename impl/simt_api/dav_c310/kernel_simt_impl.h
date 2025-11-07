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
 * \file kernel_simt_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_SIMT_IMPL_H
#define ASCENDC_MODULE_SIMT_IMPL_H

#include "kernel_tensor.h"
#include "kernel_tpipe.h"
#include "kernel_utils.h"
#include "kernel_common.h"
#include "kernel_operator_data_copy_intf.h"
#include "kernel_operator_vec_binary_scalar_intf.h"
#if __NPU_ARCH__ == 5102
    #include "dav_m510/kernel_operator_common_impl.h"
#else 
    #include "dav_c310/kernel_operator_common_impl.h"
#endif

#ifdef ASCENDC_CPU_DEBUG
#include "kernel_simt_cpu.h"
#endif

#include "kernel_simt_atomic_impl.h"
#include "kernel_simt_cast_impl.h"
#include "kernel_simt_cmp_impl.h"
#include "kernel_simt_common_impl.h"
#include "kernel_simt_print_impl.h"
#include "kernel_simt_constant.h"
#include "kernel_simt_math_impl.h"
#include "kernel_simt_transcendental_impl.h"
#include "kernel_simt_warp_level_impl.h"
#include "kernel_simt_bessel_impl.h"

#endif  // ASCENDC_MODULE_SIMT_IMPL_H
