/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATVC_ATVC_H
#define ATVC_ATVC_H

#include "common/atvc_opdef.h"
#include "common/atvc_op_check.h"
#include "common/const_def.h"
#include "elewise/common/elewise_common.h"
#include "elewise/elewise_host.h"

#include "reduce/common/reduce_common.h"
#include "reduce/reduce_host.h"

#include "broadcast/common/broadcast_common.h"
#include "broadcast/broadcast_host.h"

#include "kernel_operator.h"
#ifndef __ASCC_HOST__
#ifndef __NPU_HOST__
#include "common/kernel_utils.h"
#include "elewise/elewise_op_template.h"
#include "reduce/reduce_sum.h"
#include "reduce/reduce_op_template.h"
#include "broadcast/broadcast_compute.h"
#include "broadcast/broadcast_op_template.h"
#endif
#endif

#endif