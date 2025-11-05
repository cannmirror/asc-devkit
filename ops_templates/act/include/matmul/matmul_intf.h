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
 * \file matmul_intf.h
 * \brief
 */
#ifndef MATMUL_MATMUL_INTF_H
#define MATMUL_MATMUL_INTF_H

#include "lib/matmul/matmul.h"
#ifdef __CCE_AICORE__
#include "block/block_mmad_multi_block_bias.h"
#include "block/block_mmad_multi_block.h"
#include "block/block_mmad_multi_block_bias_with_layout.h"
#include "block/block_mmad_multi_block_with_layout.h"
#include "block/block_quant_mmad_multi_block.h"

#include "block/block_mmad_l0c_output_with_layout.h"

#include "block/block_mmad_l1_input_bias.h"
#include "block/block_mmad_l1_input.h"
#include "block/block_mmad_l1_input_bias_with_layout.h"
#include "block/block_mmad_l1_input_with_layout.h"

#include "block/block_mmad_naive_pipeline_with_layout.h"

#include "block/block_mmad_multi_block_on_kaxis_with_layout.h"

#include "block/sparse_block_mmad_multi_block_on_kaxis_with_layout.h"

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "block/block_mmad_pingpong_without_que.h"
#include "block/block_mmad_iterbatch.h"
#include "block/block_mmad_streamk.h"
#include "block/block_batch_matmul_to_mul.h"
#endif

#endif
#endif
