# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
get_filename_component(ASCENDC_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

set(ASCENDC_ADV_API_INCLUDE ${ASCENDC_INCLUDE_DIR}/adv_api)
set(ASCENDC_ADV_API_IMPL ${ASCENDC_IMPL_DIR}/adv_api)
set(ASCENDC_UTILS_INCLUDE ${ASCENDC_INCLUDE_DIR}/utils)
set(ASCENDC_UTILS_IMPL ${ASCENDC_IMPL_DIR}/utils)

set(PRODUCT_TYPE_LIST_INIT
    ascend610 ascend310p ascend910B1_AIC ascend910B1_AIV ascend910_9599_AIC ascend910_9599_AIV ascend310B1)
set(PRODUCT_TYPE_LIST ${PRODUCT_TYPE_LIST_INIT}
    CACHE STRING "Supported product types")

add_compile_definitions(__CCE_KT_TEST__=1 ASCENDC_CPU_DEBUG=1 ASCENDC_DUMP=0)

set(ASCENDC_SYS_DIR ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux)

set(ASCENDC_DEP_HEADER_DIR
    ${ASCENDC_SYS_DIR}/include
    ${ASCENDC_SYS_DIR}/include/base
    ${ASCENDC_SYS_DIR}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/include
)

set(ASCEND_DEP_LIB_DIR
    ${ASCENDC_SYS_DIR}/lib64/
    ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/
)

set(CMAKE_CXX_STANDARD 17)

set(ASCENDC_TEST_HEADER_FILES
    ${ASCENDC_UNIT_TESTS_DIR}/common
    ${ASCENDC_TESTS_DIR}/matmul/utils
    ${ASCENDC_DIR}
    ${KERNEL_TILING_DIR}
    ${ASCENDC_DIR}/include
    ${ASCENDC_DIR}/include/adv_api
    ${ASCENDC_DIR}/include/basic_api
    ${ASCENDC_DIR}/impl
    ${ASCENDC_DIR}/impl/basic_api
)

# ascend610 test cases
file(GLOB ASCENDC_TEST_ascend610_CASE_SRC_FILES
    ${ASCENDC_TESTS_DIR}/activation/sigmoid/test_operator_vec_sigmoid.cpp
    ${ASCENDC_TESTS_DIR}/pad/broadcast/test_operator_broadcast_v200.cpp
    ${ASCENDC_TESTS_DIR}/math/erf/test_operator_erf.cpp
    ${ASCENDC_TESTS_DIR}/math/erfc/test_operator_erfc.cpp
    ${ASCENDC_TESTS_DIR}/math/round/test_operator_round.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquant/test_ascend_quant.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquant/test_ascend_quant_per_channel.cpp
)

# ascend310p test cases
file(GLOB ASCENDC_TEST_ASCEND310P_CASE_SRC_PART_FILES
    ${ASCENDC_TESTS_DIR}/activation/sigmoid/test_operator_vec_sigmoid.cpp
    ${ASCENDC_TESTS_DIR}/pad/broadcast/test_operator_broadcast_v200.cpp
    ${ASCENDC_TESTS_DIR}/math/erf/test_operator_erf.cpp
    ${ASCENDC_TESTS_DIR}/math/erfc/test_operator_erfc.cpp
    ${ASCENDC_TESTS_DIR}/math/round/test_operator_round.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquant/test_ascend_quant.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquant/test_ascend_quant_per_channel.cpp
    ${ASCENDC_TESTS_DIR}/quantization/dequant/test_operator_dequant_v200.cpp
    ${ASCENDC_TESTS_DIR}/math/exp/test_operator_exp.cpp
    ${ASCENDC_TESTS_DIR}/matmul/test_matmul_config.cpp
    ${ASCENDC_TESTS_DIR}/matmul/test_operator_matmul_v200.cpp
    ${ASCENDC_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_mdl_310p.cpp
    ${ASCENDC_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_norm_310p.cpp
    #${ASCENDC_TESTS_DIR}/matmul/bias/test_copy_bias_in_310p.cpp
    ${ASCENDC_TESTS_DIR}/math/power/test_operator_power_v200.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_xor_sum/test_operator_reduce_xor_sum.cpp
    ${ASCENDC_TESTS_DIR}/normalization/welfordupdate/test_operator_welfordupdate.cpp
    ${ASCENDC_TESTS_DIR}/normalization/welfordfinalize/test_operator_welfordfinalize.cpp
    ${ASCENDC_TESTS_DIR}/normalization/normalize/test_operator_normalize.cpp
    ${ASCENDC_TESTS_DIR}/normalization/layernormV2/test_operator_layernormV2.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/bias_scheduler/test_bias_scheduler_v200.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/batch_scheduler/test_batch_scheduler_v200.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/test_scheduler_norm.cpp
)

file(GLOB_RECURSE ASCENDC_TEST_ASCEND310P_CASE_SRC_FILES_API_CHECK ${ASCENDC_TESTS_DIR}/api_check/math/*.cpp)

set(ASCENDC_TEST_ascend310p_CASE_SRC_FILES ${ASCENDC_TEST_ASCEND310P_CASE_SRC_PART_FILES} ${ASCENDC_TEST_ASCEND310P_CASE_SRC_FILES_API_CHECK})

# ascend910B1 aiv test cases
file(GLOB ASCENDC_TEST_ASCEND910B1_AIV_CASE_SRC_PART_FILES
    ${ASCENDC_TESTS_DIR}/activation/geglu/test_operator_geglu.cpp
    ${ASCENDC_TESTS_DIR}/activation/gelu/test_operator_fast_gelu.cpp
    ${ASCENDC_TESTS_DIR}/activation/gelu/test_operator_fast_gelu_v2.cpp
    ${ASCENDC_TESTS_DIR}/activation/reglu/test_operator_reglu.cpp
    ${ASCENDC_TESTS_DIR}/activation/sigmoid/test_operator_vec_sigmoid.cpp
    ${ASCENDC_TESTS_DIR}/activation/silu/test_operator_silu.cpp
    ${ASCENDC_TESTS_DIR}/activation/softmax/test_operator_softmax_v220.cpp
    ${ASCENDC_TESTS_DIR}/activation/softmax/test_operator_softmaxflashv3_v220.cpp
    ${ASCENDC_TESTS_DIR}/activation/swiglu/test_operator_swiglu.cpp
    ${ASCENDC_TESTS_DIR}/activation/swish/test_operator_swish.cpp
    ${ASCENDC_TESTS_DIR}/index/arithprogression/test_operator_arithprogression.cpp
    ${ASCENDC_TESTS_DIR}/pad/broadcast/test_operator_broadcast_v220.cpp
    ${ASCENDC_TESTS_DIR}/filter/dropout/test_operator_dropout.cpp
    ${ASCENDC_TESTS_DIR}/pad/pad/test_operator_pad.cpp
    ${ASCENDC_TESTS_DIR}/select/selectwithbytesmask/test_operator_selectwithbytesmask.cpp
    ${ASCENDC_TESTS_DIR}/transpose/confusion_transpose/test_operator_confusion_transpose.cpp
    ${ASCENDC_TESTS_DIR}/math/acos/test_operator_acos.cpp
    ${ASCENDC_TESTS_DIR}/math/acosh/test_operator_acosh.cpp
    ${ASCENDC_TESTS_DIR}/math/asin/test_operator_asin.cpp
    ${ASCENDC_TESTS_DIR}/math/asinh/test_operator_asinh.cpp
    ${ASCENDC_TESTS_DIR}/math/atan/test_operator_atan.cpp
    ${ASCENDC_TESTS_DIR}/math/atanh/test_operator_atanh.cpp
    ${ASCENDC_TESTS_DIR}/math/axpy/test_operator_axpy.cpp
    ${ASCENDC_TESTS_DIR}/math/ceil/test_operator_ceil.cpp
    ${ASCENDC_TESTS_DIR}/math/clamp/test_operator_clamp.cpp
    ${ASCENDC_TESTS_DIR}/math/cos/test_operator_cos.cpp
    ${ASCENDC_TESTS_DIR}/math/cosh/test_operator_cosh.cpp
    ${ASCENDC_TESTS_DIR}/math/cumsum/test_operator_cumsum.cpp
    ${ASCENDC_TESTS_DIR}/math/digamma/test_operator_digamma.cpp
    ${ASCENDC_TESTS_DIR}/math/erf/test_operator_erf.cpp
    ${ASCENDC_TESTS_DIR}/math/erfc/test_operator_erfc.cpp
    ${ASCENDC_TESTS_DIR}/math/exp/test_operator_exp.cpp
    ${ASCENDC_TESTS_DIR}/math/floor/test_operator_floor.cpp
    ${ASCENDC_TESTS_DIR}/math/frac/test_operator_frac.cpp
    # ${ASCENDC_TESTS_DIR}/math/lgamma/test_operator_lgamma.cpp
    ${ASCENDC_TESTS_DIR}/math/log/test_operator_log.cpp
    ${ASCENDC_TESTS_DIR}/math/power/test_operator_power.cpp
    ${ASCENDC_TESTS_DIR}/math/round/test_operator_round.cpp
    ${ASCENDC_TESTS_DIR}/math/sign/test_operator_sign.cpp
    ${ASCENDC_TESTS_DIR}/math/sin/test_operator_sin.cpp
    ${ASCENDC_TESTS_DIR}/math/sinh/test_operator_sinh.cpp
    ${ASCENDC_TESTS_DIR}/math/tan/test_operator_tan.cpp
    ${ASCENDC_TESTS_DIR}/math/tanh/test_operator_tanh.cpp
    ${ASCENDC_TESTS_DIR}/math/trunc/test_operator_trunc.cpp
    ${ASCENDC_TESTS_DIR}/math/fmod/test_operator_fmod.cpp
    ${ASCENDC_TESTS_DIR}/math/xor/test_operator_xor.cpp
    ${ASCENDC_TESTS_DIR}/normalization/welfordfinalize/test_operator_welfordfinalize.cpp
    ${ASCENDC_TESTS_DIR}/normalization/batchnorm/test_operator_batchnorm.cpp
    ${ASCENDC_TESTS_DIR}/normalization/deepnorm/test_operator_deepnorm.cpp
    ${ASCENDC_TESTS_DIR}/normalization/layernorm/test_operator_layernorm.cpp
    ${ASCENDC_TESTS_DIR}/normalization/groupnorm/test_operator_groupnorm.cpp
    # ${ASCENDC_TESTS_DIR}/normalization/layernorm/test_operator_layernormgrad.cpp
    ${ASCENDC_TESTS_DIR}/normalization/layernorm/test_operator_layernormgradbeta.cpp
    ${ASCENDC_TESTS_DIR}/normalization/normalize/test_operator_normalize.cpp
    ${ASCENDC_TESTS_DIR}/normalization/layernormV2/test_operator_layernormV2.cpp
    ${ASCENDC_TESTS_DIR}/normalization/rmsnorm/test_operator_rmsnorm.cpp
    ${ASCENDC_TESTS_DIR}/normalization/welfordupdate/test_operator_welfordupdate.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant_scalar.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant_weight_scalar.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant_weight.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant.cpp
    ${ASCENDC_TESTS_DIR}/quantization/dequant/test_operator_dequant.cpp
    ${ASCENDC_TESTS_DIR}/quantization/quant/test_operator_quant.cpp
    ${ASCENDC_TESTS_DIR}/quantization/quant/test_operator_quant_per_channel.cpp
    ${ASCENDC_TESTS_DIR}/reduce/mean/test_operator_mean.cpp
    ${ASCENDC_TESTS_DIR}/reduce/sum/test_operator_sum.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_prod/test_operator_reduce_prod.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_max/test_operator_reduce_max.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_min/test_operator_reduce_min.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_any/test_operator_reduce_any.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_all/test_operator_reduce_all.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_sum/test_operator_reduce_sum.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_mean/test_operator_reduce_mean.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_xor_sum/test_operator_reduce_xor_sum.cpp
    ${ASCENDC_TESTS_DIR}/sort/topk/test_operator_topk.cpp
    ${ASCENDC_TESTS_DIR}/utils/init_global_memory/test_operator_init_global_memory.cpp
    ${ASCENDC_TESTS_DIR}/std/sequence/test_sequence.cpp
    ${ASCENDC_TESTS_DIR}/std/tuple/*.cpp
    ${ASCENDC_TESTS_DIR}/std/type_traits/*.cpp
    ${ASCENDC_TESTS_DIR}/transpose/transdata/*cpp
)

file(GLOB_RECURSE ASCENDC_TEST_ASCEND910B1_AIV_CASE_SRC_FILES_API_CHECK ${ASCENDC_TESTS_DIR}/api_check/*.cpp)

set(ASCENDC_TEST_ascend910B1_AIV_CASE_SRC_FILES ${ASCENDC_TEST_ASCEND910B1_AIV_CASE_SRC_PART_FILES} ${ASCENDC_TEST_ASCEND910B1_AIV_CASE_SRC_FILES_API_CHECK})

# ascend910B1 aic test cases
file(GLOB ASCENDC_TEST_ascend910B1_AIC_CASE_SRC_FILES
    ${ASCENDC_TESTS_DIR}/matmul/test_operator_matmul_v220.cpp
    ${ASCENDC_TESTS_DIR}/matmul/test_operator_matmul_v220_batch.cpp
    ${ASCENDC_TESTS_DIR}/matmul/test_matmul_l0db.cpp
    ${ASCENDC_TESTS_DIR}/matmul/test_matmul_config.cpp
    ${ASCENDC_TESTS_DIR}/matmul/test_matmul_channel_split.cpp
    ${ASCENDC_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_normal.cpp
    ${ASCENDC_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_single_buffer.cpp
    ${ASCENDC_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_double_buffer.cpp
    ${ASCENDC_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_single_global_buffer.cpp
    ${ASCENDC_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_double_global_buffer.cpp
    ${ASCENDC_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_n_buffer.cpp
    ${ASCENDC_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_bmm_db.cpp
    ${ASCENDC_TESTS_DIR}/matmul/test_matmul_l0c_buffer.cpp
    ${ASCENDC_TESTS_DIR}/matmul/test_matmul_shape_info.cpp
    ${ASCENDC_TESTS_DIR}/matmul/test_matmul_shape_info_left.cpp
    ${ASCENDC_TESTS_DIR}/hccl/test_hccl.cpp
    ${ASCENDC_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_norm.cpp
    ${ASCENDC_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_mdl.cpp
    ${ASCENDC_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_params.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_k_loop_norm.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_k_loop_mdl.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_k_loop_mdl_reorder.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_k_loop_n_buffer.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_m_loop_norm.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_n_loop_norm.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_m_loop_basic.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_n_loop_basic.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_m_loop_mdl.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_n_loop_mdl.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_m_loop_mdl_outer_product.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_n_loop_mdl_outer_product.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_m_loop_n_buffer.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_batch_loop.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_batch_loop_single.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_batch_m_loop.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_batch_n_loop.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_matmul_iterator.cpp
    ${ASCENDC_TESTS_DIR}/matmul/load_buffer/test_matmul_load_to_l0a.cpp
    ${ASCENDC_TESTS_DIR}/matmul/load_buffer/test_matmul_load_to_l0b.cpp
    ${ASCENDC_TESTS_DIR}/matmul/load_buffer/test_matmul_mmad_compute.cpp
    ${ASCENDC_TESTS_DIR}/matmul/load_buffer/test_matmul_tbufpool_l0.cpp
    ${ASCENDC_TESTS_DIR}/matmul/bias/test_bias_c1_buffer.cpp
    ${ASCENDC_TESTS_DIR}/matmul/bias/test_bias_c2_buffer.cpp
    ${ASCENDC_TESTS_DIR}/matmul/bias/test_copy_bias_in.cpp
    # ${ASCENDC_TESTS_DIR}/matmul/bias/test_copy_bias_in_batch.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/test_scheduler_mdl.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/test_scheduler_special_mdl.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/test_scheduler_norm.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/test_scheduler_intrablock.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/test_scheduler_norm_outer_product.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/test_scheduler_mdl_outer_product.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/test_scheduler_n_buffer.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/test_scheduler_mdl_fullload.cpp
    #${ASCENDC_TESTS_DIR}/matmul/scheduler/bias_scheduler/test_bias_scheduler_batch.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/bias_scheduler/test_bias_scheduler_v220.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/batch_scheduler/test_batch_scheduler.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/batch_scheduler/test_batch_scheduler_single.cpp
)

# ascend910_9599 test cases
file(GLOB ASCENDC_TEST_ascend910_9599_AIC_CASE_SRC_FILES
    ${ASCENDC_TESTS_DIR}/matmul/resource/cube_out_buffer/test_cube_out_buffer_co1_shared.cpp
    ${ASCENDC_TESTS_DIR}/matmul/param/test_matmul_cross_core_sync.cpp
    ${ASCENDC_TESTS_DIR}/matmul/kfc/test_matmul_server_c310.cpp
    ${ASCENDC_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_ub_to_l1_ss.cpp
    ${ASCENDC_TESTS_DIR}/matmul/tiling/test_matmul_constant_tiling_impl.cpp
    ${ASCENDC_TESTS_DIR}/matmul/test_operator_matmul_mx_c310.cpp
    ${ASCENDC_TESTS_DIR}/matmul/test_mx_matmul_impl.cpp
    ${ASCENDC_TESTS_DIR}/matmul/utils/test_mx_matmul_utils.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/test_scheduler_norm_mx.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/test_scheduler_mdl_mx.cpp
    ${ASCENDC_TESTS_DIR}/matmul/iterator/test_k_loop_mdl_mx.cpp
    ${ASCENDC_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_norm_mx.cpp
    ${ASCENDC_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_mdl_mx.cpp
    ${ASCENDC_TESTS_DIR}/matmul/copy_cube_in/test_copy_cube_in_params_mx.cpp
    ${ASCENDC_TESTS_DIR}/matmul/copy_cube_in/copy_tile_to_cube/test_copy_tile_to_cube_mx.cpp
    ${ASCENDC_TESTS_DIR}/matmul/param/test_matmul_shape_tiling.cpp
    ${ASCENDC_TESTS_DIR}/matmul/cube_in_buffer/test_cube_in_buffer_utils.cpp
    ${ASCENDC_TESTS_DIR}/matmul/copy_cube_out/test_copy_cube_out_fixpipe.cpp
    ${ASCENDC_TESTS_DIR}/matmul/copy_cube_out/test_quant_processor_fixpipe.cpp
    ${ASCENDC_TESTS_DIR}/matmul/load_buffer/test_matmul_load_to_l0aV2.cpp
    ${ASCENDC_TESTS_DIR}/matmul/load_buffer/test_matmul_load_to_l0bV2.cpp
    ${ASCENDC_TESTS_DIR}/matmul/load_buffer/test_matmul_load_to_l0a_gemv_mx.cpp
    ${ASCENDC_TESTS_DIR}/matmul/lib/test_matmul_client.cpp
)

file(GLOB ASCENDC_TEST_ascend910_9599_AIV_CASE_SRC_FILES
    #activate
    ${ASCENDC_TESTS_DIR}/activation/gelu/test_operator_fast_gelu.cpp
    ${ASCENDC_TESTS_DIR}/activation/gelu/test_operator_fast_gelu_v2.cpp
    ${ASCENDC_TESTS_DIR}/activation/sigmoid/test_operator_vec_sigmoid.cpp
    ${ASCENDC_TESTS_DIR}/activation/softmax/test_operator_softmax_c310.cpp
    ${ASCENDC_TESTS_DIR}/api_check/activation/softmax/kernel_softmax_flashv3_check.cpp
    ${ASCENDC_TESTS_DIR}/activation/gelu/test_operator_gelu.cpp
    ${ASCENDC_TESTS_DIR}/activation/geglu/test_operator_geglu.cpp
    ${ASCENDC_TESTS_DIR}/activation/reglu/test_operator_reglu.cpp
    ${ASCENDC_TESTS_DIR}/activation/swish/test_operator_swish.cpp
    ${ASCENDC_TESTS_DIR}/activation/silu/test_operator_silu.cpp
    ${ASCENDC_TESTS_DIR}/activation/swiglu/test_operator_swiglu.cpp
    #filter
    ${ASCENDC_TESTS_DIR}/filter/dropout/test_operator_dropout.cpp
    #index
    ${ASCENDC_TESTS_DIR}/index/arithprogression/test_operator_arithprogression.cpp
    #math
    ${ASCENDC_TESTS_DIR}/math/clamp/test_operator_clamp.cpp
    ${ASCENDC_TESTS_DIR}/math/bitwise_and/test_operator_bitwise_and.cpp
    ${ASCENDC_TESTS_DIR}/math/logical_and/test_operator_logical_and.cpp
    ${ASCENDC_TESTS_DIR}/math/logical_ands/test_operator_logical_ands.cpp
    ${ASCENDC_TESTS_DIR}/math/logical_or/test_operator_logical_or.cpp
    ${ASCENDC_TESTS_DIR}/math/logical_ors/test_operator_logical_ors.cpp
    ${ASCENDC_TESTS_DIR}/math/logical_not/test_operator_logical_not.cpp
    ${ASCENDC_TESTS_DIR}/math/isnan/test_operator_isnan.cpp
    ${ASCENDC_TESTS_DIR}/math/isinf/test_operator_isinf.cpp
    ${ASCENDC_TESTS_DIR}/math/fma/test_operator_fma.cpp
    ${ASCENDC_TESTS_DIR}/math/rint/test_operator_rint.cpp
    ${ASCENDC_TESTS_DIR}/math/sincos/test_operator_sincos.cpp
    ${ASCENDC_TESTS_DIR}/math/isfinite/test_operator_isfinite.cpp
    ${ASCENDC_TESTS_DIR}/math/atanh/test_operator_atan_c310.cpp
    ${ASCENDC_TESTS_DIR}/math/cos/test_operator_cos_c310.cpp
    ${ASCENDC_TESTS_DIR}/math/cumsum/test_operator_cumsum.cpp
    ${ASCENDC_TESTS_DIR}/math/erf/test_operator_erf.cpp
    ${ASCENDC_TESTS_DIR}/math/fmod/test_operator_fmod.cpp
    ${ASCENDC_TESTS_DIR}/math/hypot/test_operator_hypot.cpp
    ${ASCENDC_TESTS_DIR}/math/log/test_operator_log.cpp
    ${ASCENDC_TESTS_DIR}/math/philox/test_operator_philox_c310.cpp
    ${ASCENDC_TESTS_DIR}/math/power/test_operator_power_c310.cpp
    ${ASCENDC_TESTS_DIR}/math/sin/test_operator_sin_c310.cpp
    ${ASCENDC_TESTS_DIR}/math/tanh/test_operator_tanh_c310.cpp
    ${ASCENDC_TESTS_DIR}/math/cosh/test_operator_cosh.cpp
    ${ASCENDC_TESTS_DIR}/math/sinh/test_operator_sinh.cpp
    ${ASCENDC_TESTS_DIR}/math/sign/test_operator_sign.cpp
    ${ASCENDC_TESTS_DIR}/math/exp/test_operator_exphighprecision.cpp
    ${ASCENDC_TESTS_DIR}/math/where/test_operator_where.cpp
    ${ASCENDC_TESTS_DIR}/api_check/math/where/kernel_where_check.cpp
    ${ASCENDC_TESTS_DIR}/math/bitwise_not/test_operator_bitwise_not.cpp
    ${ASCENDC_TESTS_DIR}/math/bitwise_or/test_operator_bitwise_or.cpp
    ${ASCENDC_TESTS_DIR}/math/bitwise_xor/test_operator_bitwise_xor.cpp
    ${ASCENDC_TESTS_DIR}/math/logical_xor/test_operator_logical_xor.cpp
    # normalization
    ${ASCENDC_TESTS_DIR}/normalization/welfordfinalize/test_operator_welfordfinalize.cpp
    ${ASCENDC_TESTS_DIR}/normalization/normalize/test_operator_normalize_c310.cpp
    ${ASCENDC_TESTS_DIR}/normalization/layernormV2/test_operator_layernormV2_c310.cpp
    ${ASCENDC_TESTS_DIR}/normalization/welfordupdate/test_operator_welfordupdate.cpp
    ${ASCENDC_TESTS_DIR}/normalization/groupnorm/test_operator_groupnorm.cpp
    # quantization
    ${ASCENDC_TESTS_DIR}/quantization/dequantize/test_dequantize.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquantize/test_antiquantize.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant_scalar.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant_weight_scalar.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant_weight_c310.cpp
    ${ASCENDC_TESTS_DIR}/quantization/antiquant/test_ascend_antiquant.cpp
    ${ASCENDC_TESTS_DIR}/quantization/dequant/test_ascend_quant_per_group.cpp
    ${ASCENDC_TESTS_DIR}/quantization/dequant/test_ascend_quant_per_group_transpose.cpp
    ${ASCENDC_TESTS_DIR}/quantization/dequant/test_ascend_quant_per_token.cpp
    ${ASCENDC_TESTS_DIR}/quantization/dequant/test_ascend_quant_per_token_transpose.cpp
    ${ASCENDC_TESTS_DIR}/quantization/dequant/test_ascend_dequant.cpp
    ${ASCENDC_TESTS_DIR}/quantization/dequant/test_ascend_dequant_per_group.cpp
    ${ASCENDC_TESTS_DIR}/quantization/dequant/test_ascend_dequant_per_token.cpp
    ${ASCENDC_TESTS_DIR}/quantization/quant/test_ascend_quant_per_group_scalar.cpp
    ${ASCENDC_TESTS_DIR}/quantization/quant/test_ascend_quant_per_group.cpp
    ${ASCENDC_TESTS_DIR}/quantization/quant/test_ascend_quant_per_token_scalar.cpp
    ${ASCENDC_TESTS_DIR}/quantization/quant/test_ascend_quant_per_token.cpp
    ${ASCENDC_TESTS_DIR}/quantization/quantize/test_quantize_per_tensor.cpp
    ${ASCENDC_TESTS_DIR}/quantization/quantize/test_quantize_per_channel.cpp
    ${ASCENDC_TESTS_DIR}/quantization/quantize/test_quantize_per_token.cpp
    ${ASCENDC_TESTS_DIR}/quantization/quantize/test_quantize_per_group.cpp
    # reduce
    ${ASCENDC_TESTS_DIR}/reduce/reduce_max/test_operator_reduce_max_c310.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_min/test_operator_reduce_min_c310.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_sum/test_operator_reduce_sum_c310.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_all/test_operator_reduce_all_c310.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_any/test_operator_reduce_any_c310.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_prod/test_operator_reduce_prod_c310.cpp
    ${ASCENDC_TESTS_DIR}/api_check/reduce/reduce_all/kernel_reduce_all_check.cpp
    ${ASCENDC_TESTS_DIR}/api_check/reduce/reduce_any/kernel_reduce_any_check.cpp
    ${ASCENDC_TESTS_DIR}/api_check/reduce/reduce_prod/kernel_reduce_prod_check.cpp
    ${ASCENDC_TESTS_DIR}/reduce/mean/test_operator_mean.cpp
    ${ASCENDC_TESTS_DIR}/reduce/sum/test_operator_sum.cpp
    ${ASCENDC_TESTS_DIR}/reduce/reduce_xor_sum/test_operator_reduce_xor_sum.cpp
    # pad
    ${ASCENDC_TESTS_DIR}/pad/broadcast/test_operator_broadcast_c310.cpp
    ${ASCENDC_TESTS_DIR}/api_check/pad/broadcast/kernel_broadcast_check.cpp
    ${ASCENDC_TESTS_DIR}/pad/pad/test_operator_pad.cpp
    # select
    ${ASCENDC_TESTS_DIR}/select/selectwithbytesmask/test_operator_selectwithbytesmask.cpp
    # sort
    ${ASCENDC_TESTS_DIR}/sort/sort/test_operator_radix_sort.cpp
    ${ASCENDC_TESTS_DIR}/sort/topk/test_operator_radix_topk.cpp
    # transpose
    ${ASCENDC_TESTS_DIR}/transpose/confusion_transpose/test_operator_confusion_transpose_c310.cpp
    # hccl
    ${ASCENDC_TESTS_DIR}/hccl/test_hccl_910_9599.cpp
    # cmath
    ${ASCENDC_TESTS_DIR}/std/cmath/test_abs.cpp
    ${ASCENDC_TESTS_DIR}/std/cmath/test_sqrt.cpp
    # matmul
)

# ascend310B1 test cases
file(GLOB ASCENDC_TEST_ascend310B1_CASE_SRC_FILES
    ${ASCENDC_TESTS_DIR}/activation/softmax/test_operator_softmax_v300.cpp
    ${ASCENDC_TESTS_DIR}/matmul/test_operator_matmul_v300.cpp
    ${ASCENDC_TESTS_DIR}/math/xor/test_operator_xor.cpp
    ${ASCENDC_TESTS_DIR}/math/floor/test_operator_floor.cpp
    ${ASCENDC_TESTS_DIR}/matmul/scheduler/test_scheduler_norm.cpp
)

foreach(product_type ${PRODUCT_TYPE_LIST})
    add_executable(ascendc_utest_${product_type}
        ${ASCENDC_TESTS_DIR}/main_global.cpp
        $<$<STREQUAL:${product_type},ascend610>:${ASCENDC_UNIT_TESTS_DIR}/common/tik_pv_wrapper.cpp>
        $<$<STREQUAL:${product_type},ascend310p>:${ASCENDC_UNIT_TESTS_DIR}/common/tik_pv_wrapper.cpp>
        $<$<STREQUAL:${product_type},ascend910B1_AIC>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend910B1_AIV>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend910B1_ACT>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend910B1_ATVC>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend910_9599_AIC>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend310B1>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
        ${ASCENDC_TEST_${product_type}_CASE_SRC_FILES}
    )

    add_dependencies(ascendc_utest_${product_type} gen_kernel_tiling)

    # add soc version flags
    if(${product_type} STREQUAL "ascend610")
        target_compile_definitions(ascendc_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=200
            __NPU_ARCH__=2002
            __DAV_M200__
            __UT_CHIP_VER__=610
        )
        target_link_directories(ascendc_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend610
            ${ASCENDC_SYS_DIR}/simulator/Ascend610/lib/
        )
    elseif(${product_type} STREQUAL "ascend310p")
        target_compile_definitions(ascendc_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=200
            __NPU_ARCH__=2002
            __DAV_M200__
            __UT_CHIP_VER__=710
        )
        target_link_directories(ascendc_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend610/
            ${ASCENDC_SYS_DIR}/simulator/Ascend610/lib/
        )
    elseif(${product_type} STREQUAL "ascend910B1_AIC")
        target_compile_definitions(ascendc_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=220
            __NPU_ARCH__=2201
            __DAV_C220__
            __DAV_C220_CUBE__
        )
        target_link_directories(ascendc_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910B1/
            ${ASCENDC_SYS_DIR}/simulator/Ascend910B1/lib/
        )
    elseif(${product_type} STREQUAL "ascend910B1_AIV")
        target_compile_definitions(ascendc_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=220
            __NPU_ARCH__=2201
            __DAV_C220__
            __DAV_C220_VEC__
        )
        target_link_directories(ascendc_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910B1
            ${ASCENDC_SYS_DIR}/simulator/Ascend910B1/lib/
        )
    elseif(${product_type} STREQUAL "ascend910B1_ACT")
        target_compile_definitions(ascendc_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=220
            __NPU_ARCH__=2201
            __DAV_C220__
            __DAV_C220_CUBE__
        )
        target_link_directories(ascendc_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910B1/
            ${ASCENDC_SYS_DIR}/simulator/Ascend910B1/lib/
        )
    elseif(${product_type} STREQUAL "ascend910_9599_AIC")
        target_compile_definitions(ascendc_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __NPU_ARCH__=3101
            __DAV_C310__
            __DAV_C310_CUBE__
        )
        target_link_directories(ascendc_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910_9599
            ${ASCENDC_SYS_DIR}/simulator/Ascend910_9599/lib/
        )
    elseif(${product_type} STREQUAL "ascend910_9599_AIV")
        target_compile_definitions(ascendc_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __NPU_ARCH__=3101
            __DAV_C310__
            __DAV_C310_VEC__
        )
        target_link_directories(ascendc_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910_9599
            ${ASCENDC_SYS_DIR}/simulator/Ascend910_9599/lib/
        )
    elseif(${product_type} STREQUAL "ascend310B1")
        target_compile_definitions(ascendc_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=300
            __NPU_ARCH__=3002
            __DAV_M300__
            __UT_CHIP_VER__=300
        )
        target_link_directories(ascendc_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend310B1/
            ${ASCENDC_SYS_DIR}/simulator/Ascend310B1/lib/
        )
    elseif(${product_type} STREQUAL "ascend910B1_ATVC")
        target_compile_definitions(ascendc_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=220
            __NPU_ARCH__=2201
            __DAV_C220__
            __DAV_C220_VEC__
        )
        target_link_directories(ascendc_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910B1
            ${ASCENDC_SYS_DIR}/simulator/Ascend910B1/lib/
        )
    endif()

    target_compile_definitions(ascendc_utest_${product_type} PRIVATE
        __CCE_KT_TEST__=1
        ASCENDC_CPU_DEBUG=1
        ASCENDC_OOM=1
    )

    target_include_directories(ascendc_utest_${product_type} PRIVATE
        ${ASCENDC_TEST_HEADER_FILES}
        ${ASCENDC_DEP_HEADER_DIR}
        ${ASCENDC_TEST_${product_type}_HEADER_FILES}
    )

    target_compile_options(ascendc_utest_${product_type} PRIVATE
        -fno-access-control
    )

    target_link_libraries(ascendc_utest_${product_type} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub>
        -Wl,--no-as-needed
        tikicpulib_npuchk
        tikicpulib_cceprint
        tikicpulib_stubreg
        tikcpp_debug
        $<$<STREQUAL:${product_type},ascend310p>:_pvmodel>
        $<$<STREQUAL:${product_type},ascend610>:_pvmodel>
        $<$<STREQUAL:${product_type},ascend910B1_AIC>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend910B1_AIV>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend910B1_ACT>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend910B1_ATVC>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend910_9599_AIC>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend310B1>:pem_davinci>
        unified_dlog
        c_sec
        mmpa
        error_manager
        -Wl,--as-needed
    )

    run_llt_test(
        TARGET ascendc_utest_${product_type}
        TASK_NUM 1
    )
endforeach()


############################################tiling ut###################################
file(GLOB ASCENDC_TILING_TEST_SRC_FILES
    ${ASCENDC_TESTS_DIR}/tiling/*.cpp
    ${ASCENDC_TESTS_DIR}/tiling/conv/*.cpp
    ${ASCENDC_TESTS_DIR}/tiling/conv_backprop/*.cpp
)

file(GLOB ASCENDC_TILING_SRC_FILES
    ${ASCENDC_ADV_API_IMPL}/tiling/activation/*.cpp
    ${ASCENDC_ADV_API_IMPL}/tiling/filter/*.cpp
    ${ASCENDC_ADV_API_IMPL}/tiling/index/*.cpp
    ${ASCENDC_ADV_API_IMPL}/tiling/math/*.cpp
    ${ASCENDC_ADV_API_IMPL}/tiling/matmul/*cpp
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
    ${ASCENDC_ADV_API_IMPL}/detail/host_log.cpp
)

# ascendc_tiling_utest
foreach(product_type ${PRODUCT_TYPE_LIST})
    add_executable(ascendc_tiling_utest_${product_type}
        ${ASCENDC_TILING_SRC_FILES}
        ${ASCENDC_TILING_TEST_SRC_FILES}
        ${ASCENDC_TILING_TEST_${product_type}_CASE_SRC_FILES}
        ${ASCENDC_UNIT_TESTS_DIR}/common/platform_stub.cpp
        ${ASCENDC_TESTS_DIR}/tiling/main.cpp
    )

    if(${product_type} STREQUAL "ascend610")
        target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=200
            __NPU_ARCH__=2002
        )
    elseif(${product_type} STREQUAL "ascend310p")
        target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=200
            __NPU_ARCH__=2002
        )
    elseif(${product_type} STREQUAL "ascend910B1_AIC")
        target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=220
            __NPU_ARCH__=2201
        )
    elseif(${product_type} STREQUAL "ascend910B1_AIV")
        target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=220
            __NPU_ARCH__=2201
        )
    elseif(${product_type} STREQUAL "ascend910B1_ATVC")
        target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=220
            __NPU_ARCH__=2201
        )
    elseif(${product_type} STREQUAL "ascend910_9599_AIC")
        target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __NPU_ARCH__=3101
            __DAV_C310__
        )
    elseif(${product_type} STREQUAL "ascend910_9599_AIV")
        target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __NPU_ARCH__=3101
            __DAV_C310__
        )
    elseif(${product_type} STREQUAL "ascend310B1")
        target_compile_definitions(ascendc_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=300
            __NPU_ARCH__=3002
        )
    elseif(${product_type} STREQUAL "ascend610Lite")
        target_compile_definitions(ascendc_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __NPU_ARCH__=3102
        )
    endif()

    target_include_directories(ascendc_tiling_utest_${product_type} PRIVATE
        ${ASCENDC_DEP_HEADER_DIR}
        ${ASCENDC_TEST_HEADER_FILES}
        ${ASCENDC_TEST_${product_type}_HEADER_FILES}
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
        ${ASCENDC_ADV_API_INCLUDE}/conv/
        ${ASCENDC_ADV_API_INCLUDE}/conv_backprop/
        ${ASCENDC_UNIT_TESTS_DIR}/common/
        ${ASCENDC_ADV_API_INCLUDE}/conv/conv3d/
    )

    target_link_directories(ascendc_tiling_utest_${product_type} PRIVATE
        ${ASCEND_DEP_LIB_DIR}
    )

    target_link_libraries(ascendc_tiling_utest_${product_type} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub>
        platform
        -Wl,--no-as-needed
        ascend_protobuf
        unified_dlog
        c_sec
        mmpa
        graph
        graph_base
        register
        error_manager
        dl
        -Wl,--as-needed
    )

    run_llt_test(
        TARGET ascendc_tiling_utest_${product_type}
        TASK_NUM 1
    )
endforeach()