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
 * \file kernel_api.h
 * \brief
 */
#ifndef LIB_KERNEL_API_H
#define LIB_KERNEL_API_H

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
#include "hccl/hccl.h"
#endif // __NPU_ARCH__ == 3101

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
#include "activation/softmax.h"
#include "index/arithprogression.h"
#include "activation/sigmoid.h"
#include "math/power.h"
#include "math/log.h"
#include "math/sin.h"
#include "math/cos.h"
#include "math/asin.h"
#include "math/acos.h"
#include "math/asinh.h"
#include "math/acosh.h"
#include "math/atan.h"
#include "math/cosh.h"
#include "math/erf.h"
#include "math/erfc.h"
#include "math/floor.h"
#include "math/tanh.h"
#include "math/exp.h"
#include "reduce/sum.h"
#include "select/selectwithbytesmask.h"
#include "math/sinh.h"
#include "math/tan.h"
#include "math/round.h"
#include "math/ceil.h"
#include "sort/topk.h"
#include "math/sign.h"
#include "pad/broadcast.h"
#include "utils/init_global_memory.h"
#include "quantization/ascend_quant.h"
#include "quantization/ascend_antiquant.h"
#include "quantization/ascend_dequant.h"
#include "transpose/confusion_transpose.h"
#include "activation/softmaxflash.h"
#include "activation/softmaxflashv2.h"
#include "filter/dropout.h"
#include "activation/gelu.h"
#include "activation/silu.h"
#include "activation/swiglu.h"
#include "activation/geglu.h"
#include "activation/reglu.h"
#include "normalization/rmsnorm.h"
#include "math/cumsum.h"
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 1001 || __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201 || \
    __NPU_ARCH__ == 3002)
#include "filter/dropout.h"
#include "activation/sigmoid.h"
#include "activation/softmax.h"
#include "activation/simplesoftmax.h"
#include "activation/softmaxflashv2.h"
#include "activation/softmaxflashv3.h"
#include "activation/softmaxgrad.h"
#include "math/xor.h"
#include "math/floor.h"
#include "sort/sort.h"
#endif

#include "../utils/std/tuple.h"
#include "../utils/std/type_traits.h"
#include "../utils/std/utility.h"
#include "../utils/std/algorithm.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 1001 || __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "index/arithprogression.h"
#include "normalization/layernormgrad.h"
#include "normalization/layernormgradbeta.h"
#include "pad/pad.h"
#include "hccl/hccl.h"
#include "math/frac.h"
#include "math/power.h"
#include "math/log.h"
#include "math/sin.h"
#include "math/cos.h"
#include "math/asin.h"
#include "math/acos.h"
#include "math/asinh.h"
#include "math/acosh.h"
#include "math/atan.h"
#include "math/cosh.h"
#include "math/erf.h"
#include "math/erfc.h"
#include "math/clamp.h"
#include "normalization/rmsnorm.h"
#include "normalization/batchnorm.h"
#include "math/tanh.h"
#include "math/atanh.h"
#include "normalization/deepnorm.h"
#include "math/exp.h"
#include "normalization/layernorm.h"
#include "normalization/welfordfinalize.h"
#include "normalization/normalize.h"
#include "reduce/sum.h"
#include "activation/silu.h"
#include "activation/gelu.h"
#include "quantization/ascend_quant.h"
#include "quantization/ascend_dequant.h"
#include "quantization/ascend_antiquant.h"
#include "activation/logsoftmax.h"
#include "activation/softmaxflash.h"
#include "transpose/confusion_transpose.h"
#include "select/selectwithbytesmask.h"
#include "math/sinh.h"
#include "activation/swiglu.h"
#include "activation/reglu.h"
#include "math/tan.h"
#include "math/round.h"
#include "math/trunc.h"
#include "activation/swish.h"
#include "sort/topk.h"
#include "activation/geglu.h"
#include "math/lgamma.h"
#include "math/digamma.h"
#include "math/sign.h"
#include "reduce/mean.h"
#include "math/axpy.h"
#include "math/ceil.h"
#include "pad/broadcast.h"
#include "reduce/reduce_xor_sum.h"
#include "reduce/reduce.h"
#include "transpose/transdata.h"
#include "math/cumsum.h"
#include "math/fmod.h"
#include "normalization/groupnorm.h"
#include "utils/init_global_memory.h"
#endif // __NPU_ARCH__ == 1001 || __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201

#endif // LIB_KERNEL_API_H
