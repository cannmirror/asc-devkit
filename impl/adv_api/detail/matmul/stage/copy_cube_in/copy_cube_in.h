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
* \file copy_cube_in.h
* \brief
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BASE_COPY_CUBE_IN_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BASE_COPY_CUBE_IN_H

#include "base/copy_cube_in_atop.h"
#include "base/copy_cube_in_from_l1.h"
#include "base/copy_cube_in_ubtol1_singleshape.h"
#include "base/copy_cube_in_mdl.h"
#include "base/copy_cube_in_mdl_fullload.h"
#include "base/copy_cube_in_norm.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
#include "base/copy_cube_in_mdl_sparse.h"
#endif
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 1001 || __NPU_ARCH__ == 2002)
#include "base/copy_cube_in_using_ub.h"
#include "batch/batch_copy_cube_in_using_ub.h"
#endif
#include "batch/batch_copy_cube_in.h"
#include "batch/batch_copy_cube_in_from_l1.h"

#endif // IMPL_MATMUL_STAGE_COPY_CUBE_IN_BASE_COPY_CUBE_IN_H
