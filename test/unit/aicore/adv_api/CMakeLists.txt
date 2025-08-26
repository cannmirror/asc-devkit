# Copyright (c) 2024 Huawei Technologies Co., Ltd. This file is a part of the CANN Open Software. Licensed under CANN
# Open Software License Agreement Version 1.0 (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF
# ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the License.
# ===============================================================================

# get the current absolute dir
get_filename_component(ASCENDC_ADV_API_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

set(PRODUCT_TYPE_LIST
    ascend610
    ascend310p
    ascend910B1_AIC
    ascend910B1_AIV
    ascend910_9599_AIC
    ascend910_9599_AIV
    ascend310B1
)

set(ASCENDC_ADV_API_INCLUDE ${ASCENDC_INCLUDE_DIR}/aicore/adv_api)
set(ASCENDC_ADV_API_IMPL ${ASCENDC_IMPL_DIR}/aicore/adv_api)
set(ASCENDC_UTILS_INCLUDE ${ASCENDC_INCLUDE_DIR}/aicore/utils)
set(ASCENDC_UTILS_IMPL ${ASCENDC_IMPL_DIR}/aicore/utils)

add_compile_definitions(__CCE_KT_TEST__=1 ASCENDC_CPU_DEBUG=1 ASCENDC_DUMP=0 LOG_CPP)

set(ASCENDC_SYS_DIR ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux)

set(ASCENDC_DEP_HEADER_DIR ${ASCENDC_SYS_DIR}/include ${ASCENDC_SYS_DIR}/include/base
                           ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/include
)

set(ASCEND_DEP_LIB_DIR ${ASCENDC_SYS_DIR}/lib64/ ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/)

set(CMAKE_CXX_STANDARD 17)

set(ASCENDC_TEST_HEADER_FILES
    ${ASCENDC_UNIT_TESTS_DIR}/common
    ${ASCENDC_ADV_API_TESTS_DIR}/matmul/utils
    ${ASCENDC_ADV_API_INCLUDE}
    ${ASCENDC_IMPL_DIR}/aicore
    ${ASCENDC_UTILS_INCLUDE}
    ${ASCENDC_UTILS_IMPL}
    ${KERNEL_TILING_DIR}
    ${ASCENDC_SYS_DIR}/tikcpp/tikcfw
    ${ASCENDC_SYS_DIR}/tikcpp/tikcfw/interface
    ${ASCENDC_SYS_DIR}/tikcpp/tikcfw/impl
)

# ascend610 test cases
file(
  GLOB
  ASCENDC_TEST_ascend610_CASE_SRC_FILES
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/sigmoid/test_operator_vec_sigmoid.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/pad/broadcast/test_operator_broadcast_v200.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/erf/test_operator_erf.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/erfc/test_operator_erfc.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/round/test_operator_round.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquant/test_ascend_quant.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquant/test_ascend_quant_per_channel.cpp
)

# ascend310p test cases
file(
  GLOB
  ASCENDC_TEST_ascend310p_CASE_SRC_FILES
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/sigmoid/test_operator_vec_sigmoid.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/pad/broadcast/test_operator_broadcast_v200.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/erf/test_operator_erf.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/erfc/test_operator_erfc.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/round/test_operator_round.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquant/test_ascend_quant.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquant/test_ascend_quant_per_channel.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/dequant/test_operator_dequant_v200.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/exp/test_operator_exp.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/test_matmul_config.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/test_operator_matmul_v200.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_mdl_310p.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_norm_310p.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/bias/test_copy_bias_in_310p.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/power/test_operator_power_v200.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_xor_sum/test_operator_reduce_xor_sum.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/welfordupdate/test_operator_welfordupdate.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/welfordfinalize/test_operator_welfordfinalize.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/normalize/test_operator_normalize.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/layernormV2/test_operator_layernormV2.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/bias_scheduler/test_bias_scheduler_v200.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/batch_scheduler/test_batch_scheduler_v200.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/test_scheduler_norm.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/acos/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/acosh/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/asin/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/asinh/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/atan/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/atanh/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/axpy/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/ceil/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/clamp/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/cos/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/cosh/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/cumsum/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/digamma/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/erf/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/erfc/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/floor/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/fmod/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/frac/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/lgamma/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/log/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/power/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/round/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/sign/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/sin/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/sinh/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/tan/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/tanh/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/trunc/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/xor/*.cpp
)

# ascend910B1 aiv test cases
file(
  GLOB
  ASCENDC_TEST_ascend910B1_AIV_CASE_SRC_FILES
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/geglu/test_operator_geglu.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/gelu/test_operator_fast_gelu.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/gelu/test_operator_fast_gelu_v2.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/reglu/test_operator_reglu.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/sigmoid/test_operator_vec_sigmoid.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/silu/test_operator_silu.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/softmax/test_operator_softmax_v220.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/softmax/test_operator_softmaxflashv3_v220.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/swiglu/test_operator_swiglu.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/swish/test_operator_swish.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/index/arithprogression/test_operator_arithprogression.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/pad/broadcast/test_operator_broadcast_v220.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/filter/dropout/test_operator_dropout.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/pad/pad/test_operator_pad.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/select/selectwithbytesmask/test_operator_selectwithbytesmask.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/transpose/confusion_transpose/test_operator_confusion_transpose.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/acos/test_operator_acos.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/acosh/test_operator_acosh.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/asin/test_operator_asin.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/asinh/test_operator_asinh.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/atan/test_operator_atan.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/atanh/test_operator_atanh.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/axpy/test_operator_axpy.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/ceil/test_operator_ceil.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/clamp/test_operator_clamp.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/cos/test_operator_cos.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/cosh/test_operator_cosh.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/cumsum/test_operator_cumsum.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/digamma/test_operator_digamma.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/erf/test_operator_erf.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/erfc/test_operator_erfc.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/exp/test_operator_exp.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/floor/test_operator_floor.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/frac/test_operator_frac.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/lgamma/test_operator_lgamma.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/log/test_operator_log.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/power/test_operator_power.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/round/test_operator_round.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/sign/test_operator_sign.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/sin/test_operator_sin.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/sinh/test_operator_sinh.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/tan/test_operator_tan.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/tanh/test_operator_tanh.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/trunc/test_operator_trunc.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/fmod/test_operator_fmod.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/xor/test_operator_xor.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/welfordfinalize/test_operator_welfordfinalize.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/batchnorm/test_operator_batchnorm.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/deepnorm/test_operator_deepnorm.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/layernorm/test_operator_layernorm.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/groupnorm/test_operator_groupnorm.cpp
  # ${ASCENDC_ADV_API_TESTS_DIR}/normalization/layernorm/test_operator_layernormgrad.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/layernorm/test_operator_layernormgradbeta.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/normalize/test_operator_normalize.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/layernormV2/test_operator_layernormV2.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/rmsnorm/test_operator_rmsnorm.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/welfordupdate/test_operator_welfordupdate.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant_scalar.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant_weight_scalar.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant_weight.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/dequant/test_operator_dequant.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/quant/test_operator_quant.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/quant/test_operator_quant_per_channel.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/mean/test_operator_mean.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/sum/test_operator_sum.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_prod/test_operator_reduce_prod.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_max/test_operator_reduce_max.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_min/test_operator_reduce_min.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_any/test_operator_reduce_any.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_all/test_operator_reduce_all.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_sum/test_operator_reduce_sum.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_mean/test_operator_reduce_mean.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_xor_sum/test_operator_reduce_xor_sum.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/sort/topk/test_operator_topk.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/utils/init_global_memory/test_operator_init_global_memory.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/std/sequence/test_sequence.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/std/tuple/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/std/type_traits/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/transpose/transdata/*cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/geglu/kernel_geglu_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/gelu/kernel_gelu_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/gelu/kernel_faster_gelu_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/gelu/kernel_faster_geluv2_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/reglu/kernel_reglu_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/sigmoid/kernel_sigmoid_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/silu/kernel_silu_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/swiglu/kernel_swiglu_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/swish/kernel_swish_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/softmax/adjust_softmax_res/kernel_adjust_softmax_res_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/softmax/log_softmax/kernel_log_softmax_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/softmax/simple_softmax/kernel_simple_softmax_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/softmax/softmax/kernel_softmax_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/softmax/softmax_flash/kernel_softmax_flash_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/softmax/softmax_flashv2/kernel_softmax_flashv2_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/softmax/softmax_flashv3/kernel_softmax_flashv3_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/softmax/softmax_grad/kernel_softmax_grad_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/activation/softmax/softmax_grad_front/kernel_softmax_grad_front_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/filter/droupout/kernel_droupout_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/index/arithprogression/kernel_arithprogression_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/normalization/batchnorm/kernel_batchnorm_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/normalization/deepnorm/kernel_deepnorm_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/normalization/groupnorm/kernel_groupnorm_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/normalization/layernorm/kernel_layernorm_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/normalization/layernormgrad/kernel_layernormgrad_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/normalization/layernormgradbeta/kernel_layernormgradbeta_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/normalization/normalize/kernel_normalize_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/normalization/rmsnorm/kernel_rmsnorm_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/normalization/welfordfinalize/kernel_welfordfinalize_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/normalization/welfordupdate/kernel_welfordupdate_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/pad/broadcast/kernel_broadcast_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/pad/pad/kernel_pad_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/pad/unpad/kernel_unpad_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/quantization/antiquant/kernel_antiquant_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/quantization/dequant/kernel_dequant_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/quantization/quant/kernel_quant_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/select/selectwithbytesmask/kernel_selectwithbytesmask_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/sort/topk/kernel_topk_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/transpose/confusion_transpose/kernel_confusion_transpose_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/transpose/transdata/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/utils/init_global_memory/kernel_init_global_memory_check.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/acos/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/acosh/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/asin/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/asinh/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/atan/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/atanh/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/axpy/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/ceil/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/clamp/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/cos/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/cosh/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/cumsum/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/digamma/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/erf/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/erfc/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/exp/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/floor/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/fmod/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/frac/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/lgamma/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/log/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/power/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/round/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/sign/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/sin/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/sinh/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/tan/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/tanh/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/trunc/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/math/xor/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/reduce/mean/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/reduce/sum/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/reduce/reduce_xor_sum/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/reduce/reduce_all/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/reduce/reduce_any/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/reduce/reduce_max/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/reduce/reduce_min/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/reduce/reduce_sum/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/reduce/reduce_prod/*.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/api_check/reduce/reduce_mean/*.cpp
)

# ascend910B1 aic test cases
file(
  GLOB
  ASCENDC_TEST_ascend910B1_AIC_CASE_SRC_FILES
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/test_operator_matmul_v220.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/test_operator_matmul_v220_batch.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/test_matmul_l0db.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/test_matmul_config.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/test_matmul_channel_split.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_normal.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_single_buffer.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_double_buffer.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_single_global_buffer.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_double_global_buffer.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_n_buffer.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/test_matmul_l0c_buffer.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/test_matmul_shape_info.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/test_matmul_shape_info_left.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/hccl/test_hccl.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_norm.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_mdl.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_params.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_k_loop_norm.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_k_loop_mdl.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_k_loop_mdl_reorder.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_k_loop_n_buffer.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_m_loop_norm.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_n_loop_norm.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_m_loop_basic.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_n_loop_basic.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_m_loop_mdl.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_n_loop_mdl.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_m_loop_mdl_outer_product.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_n_loop_mdl_outer_product.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_m_loop_n_buffer.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_batch_loop.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_batch_loop_single.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_batch_m_loop.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_batch_n_loop.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/iterator/test_matmul_iterator.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/load_buffer/test_matmul_load_to_l0a.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/load_buffer/test_matmul_load_to_l0b.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/load_buffer/test_matmul_mmad_compute.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/load_buffer/test_matmul_tbufpool_l0.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/bias/test_bias_c1_buffer.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/bias/test_bias_c2_buffer.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/bias/test_copy_bias_in.cpp
  # ${ASCENDC_ADV_API_TESTS_DIR}/matmul/bias/test_copy_bias_in_batch.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/test_scheduler_mdl.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/test_scheduler_special_mdl.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/test_scheduler_norm.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/test_scheduler_intrablock.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/test_scheduler_norm_outer_product.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/test_scheduler_mdl_outer_product.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/test_scheduler_n_buffer.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/test_scheduler_mdl_fullload.cpp
  # ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/bias_scheduler/test_bias_scheduler_batch.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/bias_scheduler/test_bias_scheduler_v220.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/batch_scheduler/test_batch_scheduler.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/batch_scheduler/test_batch_scheduler_single.cpp
)

# ascend910_9599 test cases
file(GLOB ASCENDC_TEST_ascend910_9599_AIC_CASE_SRC_FILES
     ${ASCENDC_ADV_API_TESTS_DIR}/matmul/test_operator_matmul_mx_c310.cpp
     ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/test_scheduler_mdl_mx.cpp
)

file(
  GLOB
  ASCENDC_TEST_ascend910_9599_AIV_CASE_SRC_FILES
  # activate
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/gelu/test_operator_fast_gelu.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/gelu/test_operator_fast_gelu_v2.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/sigmoid/test_operator_vec_sigmoid.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/softmax/test_operator_softmax_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/gelu/test_operator_gelu.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/geglu/test_operator_geglu.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/reglu/test_operator_reglu.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/swish/test_operator_swish.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/silu/test_operator_silu.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/swiglu/test_operator_swiglu.cpp
  # filter
  ${ASCENDC_ADV_API_TESTS_DIR}/filter/dropout/test_operator_dropout.cpp
  # index
  ${ASCENDC_ADV_API_TESTS_DIR}/index/arithprogression/test_operator_arithprogression.cpp
  # math
  ${ASCENDC_ADV_API_TESTS_DIR}/math/isfinite/test_operator_isfinite.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/atanh/test_operator_atan_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/cos/test_operator_cos_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/cumsum/test_operator_cumsum.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/erf/test_operator_erf.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/fmod/test_operator_fmod.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/hypot/test_operator_hypot.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/log/test_operator_log.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/philox/test_operator_philox_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/power/test_operator_power_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/sin/test_operator_sin_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/tanh/test_operator_tanh_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/cosh/test_operator_cosh.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/sinh/test_operator_sinh.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/sign/test_operator_sign.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/exp/test_operator_exphighprecision.cpp
  # normalization
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/welfordfinalize/test_operator_welfordfinalize.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/normalize/test_operator_normalize_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/layernormV2/test_operator_layernormV2_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/normalization/welfordupdate/test_operator_welfordupdate.cpp
  # quantization
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/dequantize/test_dequantize.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquantize/test_antiquantize.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant_scalar.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant_weight_scalar.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant_weight_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/dequant/test_ascend_quant_per_group.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/dequant/test_ascend_quant_per_group_transpose.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/dequant/test_ascend_quant_per_token.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/dequant/test_ascend_quant_per_token_transpose.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/dequant/test_ascend_dequant.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/dequant/test_ascend_dequant_per_group.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/dequant/test_ascend_dequant_per_token.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/quant/test_ascend_quant_per_group_scalar.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/quant/test_ascend_quant_per_group.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/quant/test_ascend_quant_per_token_scalar.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/quant/test_ascend_quant_per_token.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/quantize/test_quantize_per_tensor.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/quantize/test_quantize_per_channel.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/quantize/test_quantize_per_token.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/quantization/quantize/test_quantize_per_group.cpp
  # reduce
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_max/test_operator_reduce_max_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_min/test_operator_reduce_min_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_sum/test_operator_reduce_sum_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/mean/test_operator_mean.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/sum/test_operator_sum.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/reduce/reduce_xor_sum/test_operator_reduce_xor_sum.cpp
  # pad
  ${ASCENDC_ADV_API_TESTS_DIR}/pad/broadcast/test_operator_broadcast_c310.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/pad/pad/test_operator_pad.cpp
  # select
  ${ASCENDC_ADV_API_TESTS_DIR}/select/selectwithbytesmask/test_operator_selectwithbytesmask.cpp
  # sort
  ${ASCENDC_ADV_API_TESTS_DIR}/sort/sort/test_operator_radix_sort.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/sort/topk/test_operator_radix_topk.cpp
  # transpose
  ${ASCENDC_ADV_API_TESTS_DIR}/transpose/confusion_transpose/test_operator_confusion_transpose_c310.cpp
  # hccl
  ${ASCENDC_ADV_API_TESTS_DIR}/hccl/test_hccl_910_9599.cpp
  # cmath
  ${ASCENDC_ADV_API_TESTS_DIR}/std/cmath/test_abs.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/std/cmath/test_sqrt.cpp
  # matmul
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/kfc/test_matmul_server_c310.cpp
)

# ascend310B1 test cases
file(
  GLOB
  ASCENDC_TEST_ascend310B1_CASE_SRC_FILES
  ${ASCENDC_ADV_API_TESTS_DIR}/activation/softmax/test_operator_softmax_v300.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/test_operator_matmul_v300.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/xor/test_operator_xor.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/math/floor/test_operator_floor.cpp
  ${ASCENDC_ADV_API_TESTS_DIR}/matmul/scheduler/test_scheduler_norm.cpp
)

foreach (product_type ${PRODUCT_TYPE_LIST})
  add_executable(
    ascendc_utest_${product_type}
    ${ASCENDC_ADV_API_TESTS_DIR}/main.cpp
    $<$<STREQUAL:${product_type},ascend610>:${ASCENDC_UNIT_TESTS_DIR}/common/tik_pv_wrapper.cpp>
    $<$<STREQUAL:${product_type},ascend310p>:${ASCENDC_UNIT_TESTS_DIR}/common/tik_pv_wrapper.cpp>
    $<$<STREQUAL:${product_type},ascend910B1_AIC>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
    $<$<STREQUAL:${product_type},ascend910B1_AIV>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
    $<$<STREQUAL:${product_type},ascend910_9599_AIC>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
    $<$<STREQUAL:${product_type},ascend910_9599_AIV>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
    $<$<STREQUAL:${product_type},ascend310B1>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
    ${ASCENDC_TEST_${product_type}_CASE_SRC_FILES}
  )

  add_dependencies(ascendc_utest_${product_type} gen_kernel_tiling)

  if (${product_type} STREQUAL "ascend610")
    target_compile_definitions(
      ascendc_utest_${product_type}
      PRIVATE
      UT_TEST
      __CCE_AICORE__=200
      __NPU_ARCH__=2002
      __DAV_M200__
      __UT_CHIP_VER__=610
    )
    target_link_directories(
      ascendc_utest_${product_type} PRIVATE ${ASCEND_DEP_LIB_DIR}
      ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend610 ${ASCENDC_SYS_DIR}/simulator/Ascend610/lib/
    )
  elseif (${product_type} STREQUAL "ascend310p")
    target_compile_definitions(
      ascendc_utest_${product_type}
      PRIVATE
      UT_TEST
      __CCE_AICORE__=200
      __NPU_ARCH__=2002
      __DAV_M200__
      __UT_CHIP_VER__=710
    )
    target_link_directories(
      ascendc_utest_${product_type} PRIVATE ${ASCEND_DEP_LIB_DIR}
      ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend610/ ${ASCENDC_SYS_DIR}/simulator/Ascend610/lib/
    )
  elseif (${product_type} STREQUAL "ascend910B1_AIC")
    target_compile_definitions(
      ascendc_utest_${product_type}
      PRIVATE
      UT_TEST
      __CCE_AICORE__=220
      __NPU_ARCH__=2201
      __DAV_C220__
      __DAV_C220_CUBE__
    )
    target_link_directories(
      ascendc_utest_${product_type} PRIVATE ${ASCEND_DEP_LIB_DIR}
      ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910B1/ ${ASCENDC_SYS_DIR}/simulator/Ascend910B1/lib/
    )
  elseif (${product_type} STREQUAL "ascend910B1_AIV")
    target_compile_definitions(
      ascendc_utest_${product_type}
      PRIVATE
      UT_TEST
      __CCE_AICORE__=220
      __NPU_ARCH__=2201
      __DAV_C220__
      __DAV_C220_VEC__
    )
    target_link_directories(
      ascendc_utest_${product_type} PRIVATE ${ASCEND_DEP_LIB_DIR}
      ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910B1 ${ASCENDC_SYS_DIR}/simulator/Ascend910B1/lib/
    )
  elseif (${product_type} STREQUAL "ascend910_9599_AIC")
    target_compile_definitions(
      ascendc_utest_${product_type} PRIVATE UT_TEST __CCE_AICORE__=310 __NPU_ARCH__=3101 __DAV_C310__ __DAV_C310_CUBE__
    )
    target_link_directories(
      ascendc_utest_${product_type} PRIVATE ${ASCEND_DEP_LIB_DIR}
      ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910_9599 ${ASCENDC_SYS_DIR}/simulator/Ascend910_9599/lib/
    )
  elseif (${product_type} STREQUAL "ascend910_9599_AIV")
    target_compile_definitions(
      ascendc_utest_${product_type} PRIVATE UT_TEST __CCE_AICORE__=310 __NPU_ARCH__=3101 __DAV_C310__ __DAV_C310_VEC__
    )
    target_link_directories(
      ascendc_utest_${product_type} PRIVATE ${ASCEND_DEP_LIB_DIR}
      ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910_9599 ${ASCENDC_SYS_DIR}/simulator/Ascend910_9599/lib/
    )
  elseif (${product_type} STREQUAL "ascend310B1")
    target_compile_definitions(
      ascendc_utest_${product_type}
      PRIVATE
      UT_TEST
      __CCE_AICORE__=300
      __NPU_ARCH__=3002
      __DAV_M300__
      __UT_CHIP_VER__=300
    )
    target_link_directories(
      ascendc_utest_${product_type} PRIVATE ${ASCEND_DEP_LIB_DIR}
      ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend310B1/ ${ASCENDC_SYS_DIR}/simulator/Ascend310B1/lib/
    )
  endif ()

  target_compile_definitions(ascendc_utest_${product_type} PRIVATE __CCE_KT_TEST__=1 ASCENDC_CPU_DEBUG=1 ASCENDC_OOM=1)

  target_include_directories(
    ascendc_utest_${product_type}
    PRIVATE ${ASCENDC_TEST_HEADER_FILES}
            ${ASCENDC_DEP_HEADER_DIR}
            ${ASCENDC_ADV_API_IMPL}
            ${ASCENDC_ADV_API_INCLUDE}/
            ${ASCENDC_ADV_API_INCLUDE}/activation/
            ${ASCENDC_ADV_API_INCLUDE}/filter/
            ${ASCENDC_ADV_API_INCLUDE}/index/
            ${ASCENDC_ADV_API_INCLUDE}/math/
            ${ASCENDC_ADV_API_INCLUDE}/matmul/
            ${ASCENDC_ADV_API_INCLUDE}/normalization/
            ${ASCENDC_ADV_API_INCLUDE}/pad/
            ${ASCENDC_ADV_API_INCLUDE}/quantization/
            ${ASCENDC_ADV_API_INCLUDE}/reduce/
            ${ASCENDC_ADV_API_INCLUDE}/select/
            ${ASCENDC_ADV_API_INCLUDE}/sort/
            ${ASCENDC_ADV_API_INCLUDE}/transpose/
            ${ASCENDC_ADV_API_INCLUDE}/conv_backprop/
            ${ASCENDC_ADV_API_INCLUDE}/lib/conv/conv3d/
  )

  target_compile_options(ascendc_utest_${product_type} PRIVATE -fno-access-control)

  target_link_libraries(
    ascendc_utest_${product_type}
    PRIVATE $<BUILD_INTERFACE:intf_llt_pub>
            -Wl,--no-as-needed
            tikicpulib_npuchk
            tikicpulib_cceprint
            tikicpulib_stubreg
            tikcpp_debug
            $<$<STREQUAL:${product_type},ascend310p>:_pvmodel>
            $<$<STREQUAL:${product_type},ascend610>:_pvmodel>
            $<$<STREQUAL:${product_type},ascend910B1_AIV>:pem_davinci>
            $<$<STREQUAL:${product_type},ascend910B1_AIC>:pem_davinci>
            $<$<STREQUAL:${product_type},ascend910_9599_AIC>:pem_davinci>
            $<$<STREQUAL:${product_type},ascend910_9599_AIV>:pem_davinci>
            $<$<STREQUAL:${product_type},ascend310B1>:pem_davinci>
            ascendalog
            c_sec
            mmpa
            error_manager
            -Wl,--as-needed
  )

  run_llt_test(TARGET ascendc_utest_${product_type} TASK_NUM 1)
endforeach ()

# ###############################tiling ut#############################
file(GLOB ASCENDC_TILING_TEST_SRC_FILES ${ASCENDC_ADV_API_TESTS_DIR}/tiling/*.cpp
     ${ASCENDC_ADV_API_TESTS_DIR}/tiling/conv/*.cpp ${ASCENDC_ADV_API_TESTS_DIR}/tiling/conv_backprop/*.cpp
)

file(
  GLOB
  ASCENDC_TILING_SRC_FILES
  ${ASCENDC_ADV_API_IMPL}/tiling/activation/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/filter/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/index/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/math/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/matmul/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/hccl/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/normalization/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/pad/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/quantization/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/reduce/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/select/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/sort/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/transpose/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/conv/*.cpp
  ${ASCENDC_ADV_API_IMPL}/tiling/conv_backprop/*.cpp
)

# ascendc_tiling_utest
foreach (product_type ${PRODUCT_TYPE_LIST})
  add_executable(
    ascendc_tiling_utest_${product_type}
    ${ASCENDC_TILING_SRC_FILES} ${ASCENDC_TILING_TEST_SRC_FILES} ${ASCENDC_UNIT_TESTS_DIR}/common/platform_stub.cpp
    ${ASCENDC_ADV_API_TESTS_DIR}/tiling/main.cpp
  )

  if (${product_type} STREQUAL "ascend610")
    target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE UT_TEST __CCE_AICORE__=200 __NPU_ARCH__=2002)
  elseif (${product_type} STREQUAL "ascend310p")
    target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE UT_TEST __CCE_AICORE__=200 __NPU_ARCH__=2002)
  elseif (${product_type} STREQUAL "ascend910B1_AIC")
    target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE UT_TEST __CCE_AICORE__=220 __NPU_ARCH__=2201)
  elseif (${product_type} STREQUAL "ascend910B1_AIV")
    target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE UT_TEST __CCE_AICORE__=220 __NPU_ARCH__=2201)
  elseif (${product_type} STREQUAL "ascend910_9599_AIC")
    target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE UT_TEST __CCE_AICORE__=310 __NPU_ARCH__=3101 __DAV_C310__)
  elseif (${product_type} STREQUAL "ascend910_9599_AIV")
    target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE UT_TEST __CCE_AICORE__=310 __NPU_ARCH__=3101 __DAV_C310__)
  elseif (${product_type} STREQUAL "ascend310B1")
    target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE UT_TEST __CCE_AICORE__=300 __NPU_ARCH__=3002)
  elseif (${product_type} STREQUAL "ascend610Lite")
    target_compile_definitions(ascendc_utest_${product_type} PRIVATE UT_TEST __CCE_AICORE__=310 __NPU_ARCH__=3102)
  endif ()

  target_include_directories(
    ascendc_tiling_utest_${product_type}
    PRIVATE ${ASCENDC_DEP_HEADER_DIR}
            ${ASCENDC_TEST_HEADER_FILES}
            ${ASCENDC_ADV_API_IMPL}/
            ${ASCENDC_ADV_API_INCLUDE}/
            ${ASCENDC_ADV_API_INCLUDE}/activation/
            ${ASCENDC_ADV_API_INCLUDE}/filter/
            ${ASCENDC_ADV_API_INCLUDE}/index/
            ${ASCENDC_ADV_API_INCLUDE}/math/
            ${ASCENDC_ADV_API_INCLUDE}/matmul/
            ${ASCENDC_ADV_API_INCLUDE}/normalization/
            ${ASCENDC_ADV_API_INCLUDE}/pad/
            ${ASCENDC_ADV_API_INCLUDE}/quantization/
            ${ASCENDC_ADV_API_INCLUDE}/reduce/
            ${ASCENDC_ADV_API_INCLUDE}/select/
            ${ASCENDC_ADV_API_INCLUDE}/sort/
            ${ASCENDC_ADV_API_INCLUDE}/transpose/
            ${ASCENDC_ADV_API_INCLUDE}/conv_backprop/
            ${ASCENDC_ADV_API_INCLUDE}/lib/conv/conv3d/
  )

  target_link_directories(ascendc_tiling_utest_${product_type} PRIVATE ${ASCEND_DEP_LIB_DIR})

  target_link_libraries(
    ascendc_tiling_utest_${product_type}
    PRIVATE $<BUILD_INTERFACE:intf_llt_pub>
            platform
            -Wl,--no-as-needed
            ascend_protobuf
            ascendalog
            c_sec
            mmpa
            graph
            graph_base
            register
            error_manager
            -Wl,--as-needed
  )

  run_llt_test(TARGET ascendc_tiling_utest_${product_type} TASK_NUM 1)
endforeach ()
