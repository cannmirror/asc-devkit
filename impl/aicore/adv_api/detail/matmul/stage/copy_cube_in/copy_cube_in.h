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
 * \file copy_cube_in.h
 * \brief
 */

#ifndef AICORE_ADV_API_DETAIL_MATMUL_STAGE_COPY_CUBE_IN_COPY_CUBE_IN_H
#define AICORE_ADV_API_DETAIL_MATMUL_STAGE_COPY_CUBE_IN_COPY_CUBE_IN_H

#include "base/copy_cube_in_atop.h"
#include "base/copy_cube_in_from_l1.h"
#include "base/copy_cube_in_ubtol1_singleshape.h"
#include "base/copy_cube_in_mdl.h"
#include "base/copy_cube_in_mdl_fullload.h"
#include "base/copy_cube_in_norm.h"
#if __CCE_AICORE__ == 220
#include "base/copy_cube_in_mdl_sparse.h"
#endif
#if __CCE_AICORE__ <= 200
#include "base/copy_cube_in_using_ub.h"
#include "batch/batch_copy_cube_in_using_ub.h"
#endif
#include "batch/batch_copy_cube_in.h"
#include "batch/batch_copy_cube_in_from_l1.h"
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "base/copy_cube_in_mdl_mx.h"
#include "base/copy_cube_in_norm_mx.h"
#endif

#endif // AICORE_ADV_API_DETAIL_MATMUL_STAGE_COPY_CUBE_IN_COPY_CUBE_IN_H
