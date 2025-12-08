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
* \file cube_in_buffer.h
* \brief
*/
#ifndef IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_H
#define IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_H

#include "cube_in_buffer_normal.h"
#include "cube_in_buffer_single_buffer.h"
#include "cube_in_buffer_single_global_buffer.h"
#include "cube_in_buffer_double_buffer.h"
#include "cube_in_buffer_double_global_buffer.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
#include "cube_in_buffer_double_buffer_sparse.h"
#include "cube_in_buffer_n_buffer.h"
#include "cube_in_buffer_bmm_db.h"
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3101
#include "cube_in_buffer_n_buffer.h"
#include "cube_in_buffer_bmm_db.h"
#endif

#if (__NPU_ARCH__ == 5102)
#include "cube_in_buffer_bmm_db.h"
#endif

#endif // _CUBE_IN_BUFFER_H_
