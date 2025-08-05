/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file reduce_device.h
 * \brief
 */

#ifndef ATVC_REDUCE_DEVICE_H
#define ATVC_REDUCE_DEVICE_H

#include "common/atvc_opdef.h"
#include "common/const_def.h"
#include "reduce/common/reduce_common.h"
#include "kernel_operator.h"
#ifndef __ASCC_HOST__
#include "common/kernel_utils.h"
#include "reduce/reduce_sum.h"
#include "reduce/reduce_op_template.h"
#include "register/op_def_registry.h"
#endif

#endif // ATVC_REDUCE_DEVICE_H