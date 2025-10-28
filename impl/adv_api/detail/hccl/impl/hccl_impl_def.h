/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_impl_def.h
 * \brief
 */
#ifndef IMPL_HCCL_HCCL_IMPL_DEF_H
#define IMPL_HCCL_HCCL_IMPL_DEF_H

#include "../common/hccl_base.h"

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
#include "hccl_v220_impl.h"
#endif

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "hccl_v310_impl.h"
#endif

#endif