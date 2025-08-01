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
set(ASCENDC_DIR @ASCENDC_DIR@)
set(ASCENDC_INSTALL_BASE_PATH ${CMAKE_INSTALL_PREFIX}/${ASCENDC_API_PATH})

file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling)

file(
  CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/activation
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/activation
  SYMBOLIC)
file(CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/filter
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/filter
     SYMBOLIC)
file(CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/index
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/index
     SYMBOLIC)
file(CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/math
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/math
     SYMBOLIC)
file(CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/matmul
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/matmul
     SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/normalization
  SYMBOLIC)
file(CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/pad
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/pad
     SYMBOLIC)
file(
  CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/quantization
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/quantization
  SYMBOLIC)
file(CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/pad
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/pad
     SYMBOLIC)
file(
  CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/quantization
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/quantization
  SYMBOLIC)
file(CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/reduce
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/reduce
     SYMBOLIC)
file(CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/select
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/select
     SYMBOLIC)
file(CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/sort
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sort
     SYMBOLIC)
file(CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/transpose
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/transpose
     SYMBOLIC)

# arithprogression
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/arithprogression
)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/index/arithprogression_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/arithprogression/arithprogression_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/index/arithprogression_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/arithprogression/arithprogression_tiling_intf.h
  SYMBOLIC)

# ascend_antiquant
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_antiquant
)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/quantization/ascend_antiquant_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_antiquant/ascend_antiquant_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/quantization/ascend_antiquant_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_antiquant/ascend_antiquant_tiling_intf.h
  SYMBOLIC)

# ascend_dequant
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_dequant
)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/quantization/ascend_dequant_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_dequant/ascend_dequant_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/quantization/ascend_dequant_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_dequant/ascend_dequant_tiling_intf.h
  SYMBOLIC)

# ascend_quant
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_quant
)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/quantization/ascend_quant_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_quant/ascend_quant_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/quantization/ascend_quant_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_quant/ascend_quant_tiling_intf.h
  SYMBOLIC)

# batchnorm
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/batchnorm)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/batchnorm_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/batchnorm/batchnorm_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/batchnorm_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/batchnorm/batchnorm_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/batchnorm_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/batchnorm/batchnorm_tilingdata.h
  SYMBOLIC)

# broadcast
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/broadcast)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/pad/broadcast_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/broadcast/broadcast_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/pad/broadcast_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/broadcast/broadcast_tiling.h
  SYMBOLIC)

# deepnorm
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/deepnorm)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/deepnorm_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/deepnorm/deepnorm_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/deepnorm_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/deepnorm/deepnorm_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/deepnorm_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/deepnorm/deepnorm_tilingdata.h
  SYMBOLIC)

# dropout
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/dropout)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/filter/dropout_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/dropout/dropout_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/filter/dropout_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/dropout_tiling_intf.h
  SYMBOLIC)

# gelu
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/gelu)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/gelu_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/gelu/gelu_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/gelu_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/gelu/gelu_tiling_intf.h
  SYMBOLIC)

# layernorm
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernorm)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/layernorm_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernorm/layernorm_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/layernorm_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernorm/layernorm_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/layernorm_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernorm/layernorm_tilingdata.h
  SYMBOLIC)

# welfordfinalize
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/welfordfinalize
)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/welfordfinalize_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/welfordfinalize/welfordfinalize_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/welfordfinalize_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/welfordfinalize/welfordfinalize_tiling_intf.h
  SYMBOLIC)
# normalize
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/normalize)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/normalize_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/normalize/normalize_tiling.h
  SYMBOLIC)

# layernormgrad
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad
)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/layernorm_grad_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad/layernorm_grad_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/layernorm_grad_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad/layernorm_grad_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/layernorm_grad_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad/layernorm_grad_tilingdata.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/layernorm_grad_beta_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad/layernorm_grad_beta_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/layernorm_grad_beta_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad/layernorm_grad_beta_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/layernorm_grad_beta_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad/layernorm_grad_beta_tilingdata.h
  SYMBOLIC)

# matmul
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/matrix)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/matmul/bmm_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/matrix/bmm_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/matmul/matmul_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/matrix/matmul_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/matmul/matmul_tiling_base.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/matrix/matmul_tiling_base.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/matmul/matmul_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/matrix/matmul_tilingdata.h
  SYMBOLIC)

# hccl
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/hccl)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/hccl/hccl_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/hccl/hccl_tilingdata.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/hccl/hccl_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/hccl/hccl_tiling.h
  SYMBOLIC)

# mean
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/mean)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/reduce/mean_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/mean/mean_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/reduce/mean_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/mean/mean_tiling.h
  SYMBOLIC)

# reglu
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/reglu)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/reglu_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/reglu/reglu_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/reglu_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/reglu/reglu_tiling_intf.h
  SYMBOLIC)

# rmsnorm
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/rmsnorm)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/rmsnorm_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/rmsnorm/rmsnorm_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/rmsnorm_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/rmsnorm/rmsnorm_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/normalization/rmsnorm_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/rmsnorm/rmsnorm_tilingdata.h
  SYMBOLIC)

# sigmoid
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sigmoid)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/sigmoid_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sigmoid/sigmoid_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/sigmoid_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sigmoid/sigmoid_tiling_intf.h
  SYMBOLIC)

# silu
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/silu)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/silu_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/silu/silu_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/silu_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/silu/silu_tiling.h
  SYMBOLIC)

# reduce_xor_sum
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/reduce_xor_sum
)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/reduce/reduce_xor_sum_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/reduce_xor_sum/reduce_xor_sum_tiling.h
  SYMBOLIC)

# softmax
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/softmax_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax/softmax_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/softmax_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax/softmax_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/softmax_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax/softmax_tilingdata.h
  SYMBOLIC)

file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/logsoftmax_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax/logsoftmax_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/logsoftmax_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax/logsoftmax_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/logsoftmax_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax/logsoftmax_tilingdata.h
  SYMBOLIC)

# sum
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sum)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/reduce/sum_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sum/sum_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/reduce/sum_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sum/sum_tiling.h
  SYMBOLIC)

# swiglu
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/swiglu)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/swiglu_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/swiglu/swiglu_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/swiglu_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/swiglu/swiglu_tiling.h
  SYMBOLIC)

# swish
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/swish)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/swish_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/swish/swish_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/activation/swish_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/swish/swish_tiling.h
  SYMBOLIC)

# topk
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/topk)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/sort/topk_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/topk/topk_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/sort/topk_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/topk/topk_tilingdata.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/sort/topk_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/topk/topk_tiling_intf.h
  SYMBOLIC)

# xor
file(MAKE_DIRECTORY
     ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/xor)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/math/xor_tiling_intf.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/xor/xor_tiling_intf.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/math/xor_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/xor/xor_tiling.h
  SYMBOLIC)

# conv3d
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/conv/conv3d)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/conv/conv3d/conv3d_tiling_base.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/conv/conv3d/conv3d_tiling_base.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/conv/conv3d/conv3d_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/conv/conv3d/conv3d_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/conv/conv3d/conv3d_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/conv/conv3d/conv3d_tilingdata.h
  SYMBOLIC)

# conv3d_backprop
file(
  MAKE_DIRECTORY
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/conv_backprop
)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/conv_backprop/conv3d_bp_tiling_base.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/conv_backprop/conv3d_bp_tiling_base.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/conv_backprop/conv3d_bp_filter_tiling_base.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/conv_backprop/conv3d_bp_filter_tiling_base.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/conv_backprop/conv3d_bp_filter_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/conv_backprop/conv3d_bp_filter_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/conv_backprop/conv3d_bp_filter_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/conv_backprop/conv3d_bp_filter_tilingdata.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/conv_backprop/conv3d_bp_input_tiling.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/conv_backprop/conv3d_bp_input_tiling.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/conv_backprop/conv3d_bp_input_tiling_base.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/conv_backprop/conv3d_bp_input_tiling_base.h
  SYMBOLIC)
file(
  CREATE_LINK
  ${ASCENDC_DIR}/include/aicore/adv_api/conv_backprop/conv3d_bp_input_tilingdata.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/conv_backprop/conv3d_bp_input_tilingdata.h
  SYMBOLIC)

file(
  CREATE_LINK ${ASCENDC_DIR}/include/aicore/adv_api/tiling_api.h
  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/tiling_api.h
  SYMBOLIC)
