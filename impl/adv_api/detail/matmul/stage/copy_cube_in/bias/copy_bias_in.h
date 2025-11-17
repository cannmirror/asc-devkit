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
* \file copy_bias_in.h
* \brief
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BIAS_COPY_BIAS_IN_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BIAS_COPY_BIAS_IN_H

#if __CCE_AICORE__ <= 200 && (__NPU_ARCH__ != 5102)
#include "copy_bias_in_v200.h"
#else
#include "copy_bias_in_v220.h"
#include "copy_bias_in_batch.h"
#endif
#include "load_bias_to_c2.h"
#endif // _COPY_BIAS_IN_H_
