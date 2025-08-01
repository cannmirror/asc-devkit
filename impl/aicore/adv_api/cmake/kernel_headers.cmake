# Copyright (c) 2024 Huawei Technologies Co., Ltd. This file is a part of the
# CANN Open Software. Licensed under CANN Open Software License Agreement
# Version 1.0 (the "License"). Please refer to the License for details. You may
# not use this file except in compliance with the License. THIS SOFTWARE IS
# PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
# FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
# repository for the full text of the License.
# ===============================================================================

set(ASCENDC_API_PATH @INSTALL_LIBRARY_DIR@)
set(ASCENDC_INSTALL_BASE_PATH ${CMAKE_INSTALL_PREFIX}/${ASCENDC_API_PATH})

# arithprogression
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/arithprogression
)
file(
  CREATE_LINK
  ../index/kernel_operator_arithprogression_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/arithprogression/kernel_operator_arithprogression_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../index/arithprogression.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/arithprogression/arithprogression.h
  SYMBOLIC)

# ascend_antiquant
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/ascend_antiquant
)
file(
  CREATE_LINK
  ../quantization/kernel_operator_ascend_antiquant_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/ascend_antiquant/kernel_operator_ascend_antiquant_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../quantization/ascend_antiquant.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/ascend_antiquant/ascend_antiquant.h
  SYMBOLIC)

# ascend_dequant
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/ascend_dequant)
file(
  CREATE_LINK
  ../quantization/kernel_operator_ascend_dequant_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/ascend_dequant/kernel_operator_ascend_dequant_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../quantization/ascend_dequant.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/ascend_dequant/ascend_dequant.h
  SYMBOLIC)

# ascend_quant
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/ascend_quant)
file(
  CREATE_LINK
  ../quantization/kernel_operator_ascend_quant_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/ascend_quant/kernel_operator_ascend_quant_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../quantization/ascend_quant.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/ascend_quant/ascend_quant.h
  SYMBOLIC)

# batchnorm
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/batchnorm)
file(
  CREATE_LINK
  ../normalization/kernel_operator_batchnorm_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/batchnorm/kernel_operator_batchnorm_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../normalization/batchnorm.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/batchnorm/batchnorm.h
  SYMBOLIC)

# broadcast
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/broadcast)
file(
  CREATE_LINK
  ../pad/broadcast.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/broadcast/broadcast.h
  SYMBOLIC)

# deepnorm
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/deepnorm)
file(
  CREATE_LINK
  ../normalization/kernel_operator_deepnorm_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/deepnorm/kernel_operator_deepnorm_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../normalization/deepnorm.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/deepnorm/deepnorm.h
  SYMBOLIC)

# dropout
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/dropout)
file(
  CREATE_LINK
  ../filter/kernel_operator_dropout_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/dropout/kernel_operator_dropout_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../filter/dropout.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/dropout/dropout.h
  SYMBOLIC)

# gelu
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/gelu)
file(
  CREATE_LINK
  ../activation/kernel_operator_gelu_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/gelu/kernel_operator_gelu_intf.h
  SYMBOLIC)
file(CREATE_LINK ../activation/gelu.h
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/gelu/gelu.h
     SYMBOLIC)

# layernorm
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/layernorm)
file(
  CREATE_LINK
  ../normalization/kernel_operator_layernorm_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/layernorm/kernel_operator_layernorm_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../normalization/layernorm.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/layernorm/layernorm.h
  SYMBOLIC)

# welfordfinalize
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/welfordfinalize
)
file(
  CREATE_LINK
  ../normalization/kernel_operator_welfordfinalize_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/welfordfinalize/kernel_operator_welfordfinalize_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../normalization/welfordfinalize.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/welfordfinalize/welfordfinalize.h
  SYMBOLIC)

# normalize
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/normalize)
file(
  CREATE_LINK
  ../normalization/normalize.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/normalize/normalize.h
  SYMBOLIC)

# layernormgrad
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/layernormgrad)
file(
  CREATE_LINK
  ../normalization/kernel_operator_layernormgrad_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/layernormgrad/kernel_operator_layernormgrad_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../normalization/layernormgrad.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/layernormgrad/layernormgrad.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../normalization/kernel_operator_layernormgradbeta_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/layernormgrad/kernel_operator_layernormgradbeta_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../normalization/layernormgradbeta.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/layernormgrad/layernormgradbeta.h
  SYMBOLIC)

# matmul
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/matrix)
file(
  CREATE_LINK ../matmul
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/matrix/matmul
  SYMBOLIC)
file(
  CREATE_LINK matmul/matmul_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/matmul_intf.h
  SYMBOLIC)

# mean
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/mean)
file(
  CREATE_LINK
  ../reduce/kernel_operator_mean_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/mean/kernel_operator_mean_intf.h
  SYMBOLIC)
file(CREATE_LINK ../reduce/mean.h
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/mean/mean.h
     SYMBOLIC)

# reglu
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/reglu)
file(
  CREATE_LINK
  ../activation/kernel_operator_reglu_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/reglu/kernel_operator_reglu_intf.h
  SYMBOLIC)
file(
  CREATE_LINK ../activation/reglu.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/reglu/reglu.h
  SYMBOLIC)

# rmsnorm
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/rmsnorm)
file(
  CREATE_LINK
  ../normalization/kernel_operator_rmsnorm_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/rmsnorm/kernel_operator_rmsnorm_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../normalization/rmsnorm.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/rmsnorm/rmsnorm.h
  SYMBOLIC)

# sigmoid
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/sigmoid)
file(
  CREATE_LINK
  ../activation/kernel_operator_sigmoid_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/sigmoid/kernel_operator_sigmoid_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../activation/sigmoid.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/sigmoid/sigmoid.h
  SYMBOLIC)

# silu
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/silu)
file(
  CREATE_LINK
  ../activation/kernel_operator_silu_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/silu/kernel_operator_silu_intf.h
  SYMBOLIC)
file(CREATE_LINK ../activation/silu.h
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/silu/silu.h
     SYMBOLIC)

# softmax
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax)
file(
  CREATE_LINK
  ../activation/kernel_operator_softmax_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax/kernel_operator_softmax_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../activation/softmax.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax/softmax.h
  SYMBOLIC)

file(
  CREATE_LINK
  ../activation/kernel_operator_logsoftmax_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax/kernel_operator_logsoftmax_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../activation/logsoftmax.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax/logsoftmax.h
  SYMBOLIC)

file(
  CREATE_LINK
  ../activation/kernel_operator_simple_softmax_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax/kernel_operator_simple_softmax_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../activation/simplesoftmax.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax/simplesoftmax.h
  SYMBOLIC)

file(
  CREATE_LINK
  ../activation/kernel_operator_softmax_flashv2_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax/kernel_operator_softmax_flashv2_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../activation/kernel_operator_softmax_flash_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax/kernel_operator_softmax_flash_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../activation/softmaxflashv2.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax/softmaxflashv2.h
  SYMBOLIC)

file(
  CREATE_LINK
  ../activation/kernel_operator_softmax_grad_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax/kernel_operator_softmax_grad_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../activation/softmaxgrad.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax/softmaxgrad.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../activation/softmaxflashv3.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/softmax/softmaxflashv3.h
  SYMBOLIC)

# sum
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/sum)
file(
  CREATE_LINK
  ../reduce/kernel_operator_sum_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/sum/kernel_operator_sum_intf.h
  SYMBOLIC)
file(CREATE_LINK ../reduce/sum.h
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/sum/sum.h
     SYMBOLIC)

# swiglu
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/swiglu)
file(
  CREATE_LINK
  ../activation/kernel_operator_swiglu_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/swiglu/kernel_operator_swiglu_intf.h
  SYMBOLIC)
file(
  CREATE_LINK ../activation/swiglu.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/swiglu/swiglu.h
  SYMBOLIC)

# swish
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/swish)
file(
  CREATE_LINK
  ../activation/kernel_operator_swish_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/swish/kernel_operator_swish_intf.h
  SYMBOLIC)
file(
  CREATE_LINK ../activation/swish.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/swish/swish.h
  SYMBOLIC)

# topk
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/topk)
file(
  CREATE_LINK
  ../sort/kernel_operator_topk_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/topk/kernel_operator_topk_intf.h
  SYMBOLIC)
file(CREATE_LINK ../sort/topk.h
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/topk/topk.h
     SYMBOLIC)

# xor
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/xor)
file(
  CREATE_LINK
  ../math/kernel_operator_xor_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/xor/kernel_operator_xor_intf.h
  SYMBOLIC)
file(CREATE_LINK ../math/xor.h
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/xor/xor.h
     SYMBOLIC)

# geglu
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/math)
file(
  CREATE_LINK
  ../activation/kernel_operator_geglu_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/math/kernel_operator_geglu_intf.h
  SYMBOLIC)
file(CREATE_LINK ../activation/geglu.h
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/math/geglu.h
     SYMBOLIC)
file(
  CREATE_LINK
  ../activation/geglu_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/math/geglu_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ../activation/geglu_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/math/geglu_tiling.h
  SYMBOLIC)

# initglobalmemory
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/init_global_memory
)
file(
  CREATE_LINK
  ../utils/init_global_memory.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib/init_global_memory/init_global_memory.h
  SYMBOLIC)
