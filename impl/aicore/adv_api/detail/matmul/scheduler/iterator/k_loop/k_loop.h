/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file k_loop.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_H
#define AICORE_ADV_API_DETAIL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_H

#include "k_loop_mdl_reorder.h"
#include "k_loop_mdl.h"
#include "k_loop_norm.h"
#include "k_loop_intrablock.h"
#include "k_loop_mdl_partial_output.h"
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "k_loop_mdl_mx.h"
#endif

#endif // _K_LOOP_H_