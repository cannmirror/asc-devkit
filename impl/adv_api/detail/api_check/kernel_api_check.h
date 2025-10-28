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
 * \file kernel_api_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_API_CHECK_H_
#define IMPL_API_CHECK_KERNEL_API_CHECK_H_

#ifdef ASCENDC_CPU_DEBUG

#include "kernel_check/activation/geglu/geglu_check.h"
#include "kernel_check/activation/gelu/faster_gelu_check.h"
#include "kernel_check/activation/gelu/faster_geluv2_check.h"
#include "kernel_check/activation/gelu/gelu_check.h"
#include "kernel_check/activation/reglu/reglu_check.h"
#include "kernel_check/activation/sigmoid/sigmoid_check.h"
#include "kernel_check/activation/silu/silu_check.h"
#include "kernel_check/activation/swiglu/swiglu_check.h"
#include "kernel_check/activation/swish/swish_check.h"
#include "kernel_check/activation/softmax/adjust_softmax_res/adjust_softmax_res_check.h"
#include "kernel_check/activation/softmax/log_softmax/log_softmax_check.h"
#include "kernel_check/activation/softmax/simple_softmax/simple_softmax_check.h"
#include "kernel_check/activation/softmax/softmax/softmax_check.h"
#include "kernel_check/activation/softmax/softmax_flash/softmax_flash_check.h"
#include "kernel_check/activation/softmax/softmax_flashv2/softmax_flashv2_check.h"
#include "kernel_check/activation/softmax/softmax_flashv3/softmax_flashv3_check.h"
#include "kernel_check/activation/softmax/softmax_grad/softmax_grad_check.h"
#include "kernel_check/activation/softmax/softmax_grad/softmax_grad_check.h"
#include "kernel_check/activation/softmax/softmax_grad_front/softmax_grad_front_check.h"
#include "kernel_check/filter/dropout/dropout_check.h"
#include "kernel_check/index/arithprogression/arithprogression_check.h"
#include "kernel_check/normalization/batchnorm/batchnorm_check.h"
#include "kernel_check/normalization/deepnorm/deepnorm_check.h"
#include "kernel_check/normalization/groupnorm/groupnorm_check.h"
#include "kernel_check/normalization/layernorm/layernorm_check.h"
#include "kernel_check/normalization/layernormgrad/layernormgrad_check.h"
#include "kernel_check/normalization/layernormgradbeta/layernormgradbeta_check.h"
#include "kernel_check/normalization/normalize/normalize_check.h"
#include "kernel_check/normalization/rmsnorm/rmsnorm_check.h"
#include "kernel_check/normalization/welfordfinalize/welfordfinalize_check.h"
#include "kernel_check/normalization/welforupdate/welfordupdate_check.h"
#include "kernel_check/pad/broadcast/broadcast_check.h"
#include "kernel_check/pad/pad/pad_check.h"
#include "kernel_check/pad/pad/unpad_check.h"
#include "kernel_check/select/selectwithbytesmask/selectwithbytesmaskcheck.h"
#include "kernel_check/sort/topk/topk_check.h"
#include "kernel_check/sort/concat/concat_check.h"
#include "kernel_check/sort/extract/extract_check.h"
#include "kernel_check/sort/sort/sort_check.h"
#include "kernel_check/sort/mrgsort/mrgsort_check.h"
#include "kernel_check/transpose/confusion_transpose/confusion_transpose_check.h"
#include "kernel_check/utils/init_global_memory/init_global_memory_check.h"
#include "kernel_check/quantization/antiquant/antiquant_check.h"
#include "kernel_check/quantization/dequant/dequant_check.h"
#include "kernel_check/quantization/quant/quant_check.h"
#include "kernel_check/math/logical_and/logical_and_check.h"
#include "kernel_check/math/logical_ands/logical_ands_check.h"
#include "kernel_check/math/logical_or/logical_or_check.h"
#include "kernel_check/math/logical_ors/logical_ors_check.h"
#include "kernel_check/math/logical_not/logical_not_check.h"
#include "kernel_check/math/bitwise_and/bitwise_and_check.h"
#include "kernel_check/math/fma/fma_check.h"
#include "kernel_check/math/rint/rint_check.h"
#include "kernel_check/math/isnan/is_nan_check.h"
#include "kernel_check/math/isinf/is_inf_check.h"
#include "kernel_check/math/sincos/sincos_check.h"
#include "kernel_check/math/acos/acos_check.h"
#include "kernel_check/math/acosh/acosh_check.h"
#include "kernel_check/math/asin/asin_check.h"
#include "kernel_check/math/asinh/asinh_check.h"
#include "kernel_check/math/atan/atan_check.h"
#include "kernel_check/math/atanh/atanh_check.h"
#include "kernel_check/math/axpy/axpy_check.h"
#include "kernel_check/math/ceil/ceil_check.h"
#include "kernel_check/math/clamp/clamp_check.h"
#include "kernel_check/math/cos/cos_check.h"
#include "kernel_check/math/cosh/cosh_check.h"
#include "kernel_check/math/cumsum/cumsum_check.h"
#include "kernel_check/math/digamma/digamma_check.h"
#include "kernel_check/math/erf/erf_check.h"
#include "kernel_check/math/erfc/erfc_check.h"
#include "kernel_check/math/exp/exp_check.h"
#include "kernel_check/math/floor/floor_check.h"
#include "kernel_check/math/fmod/fmod_check.h"
#include "kernel_check/math/frac/frac_check.h"
#include "kernel_check/math/lgamma/lgamma_check.h"
#include "kernel_check/math/log/log_check.h"
#include "kernel_check/math/power/power_check.h"
#include "kernel_check/math/round/round_check.h"
#include "kernel_check/math/sign/sign_check.h"
#include "kernel_check/math/sin/sin_check.h"
#include "kernel_check/math/sinh/sinh_check.h"
#include "kernel_check/math/tan/tan_check.h"
#include "kernel_check/math/tanh/tanh_check.h"
#include "kernel_check/math/trunc/trunc_check.h"
#include "kernel_check/math/xor/xor_check.h"
#include "kernel_check/math/where/where_check.h"
#include "kernel_check/reduce/mean/mean_check.h"
#include "kernel_check/math/bitwise_not/bitwise_not_check.h"
#include "kernel_check/math/bitwise_or/bitwise_or_check.h"
#include "kernel_check/math/bitwise_xor/bitwise_xor_check.h"
#include "kernel_check/math/logical_xor/logical_xor_check.h"
#include "kernel_check/reduce/reduce_all/reduce_all_check.h"
#include "kernel_check/reduce/reduce_any/reduce_any_check.h"
#include "kernel_check/reduce/reduce_max/reduce_max_check.h"
#include "kernel_check/reduce/reduce_min/reduce_min_check.h"
#include "kernel_check/reduce/reduce_mean/reduce_mean_check.h"
#include "kernel_check/reduce/reduce_sum/reduce_sum_check.h"
#include "kernel_check/reduce/reduce_xor_sum/reduce_xor_sum_check.h"
#include "kernel_check/reduce/reduce_prod/reduce_prod_check.h"
#include "kernel_check/reduce/sum/sum_check.h"
#include "kernel_check/transpose/transdata/transdata_check.h"

#define EXPAND(...) __VA_ARGS__
#define MERGE_ARGS(first, args) (first, EXPAND args)
#define CHECK_FUNC_HIGHLEVEL_API(API, templateArgs, inputArgs)\
    HighLevelApiCheck::CheckFunc##API<EXPAND templateArgs> MERGE_ARGS(#API, inputArgs)
#else
#define CHECK_FUNC_HIGHLEVEL_API(...)
#endif

#endif // IMPL_API_CHECK_KERNEL_API_CHECK_H_
