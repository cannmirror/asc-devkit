# Copyright (c) 2024 Huawei Technologies Co., Ltd. This file is a part of the
# CANN Open Software. Licensed under CANN Open Software License Agreement
# Version 1.0 (the "License"). Please refer to the License for details. You may
# not use this file except in compliance with the License. THIS SOFTWARE IS
# PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
# FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
# repository for the full text of the License.
# ===============================================================================

add_library(tiling_api_headers INTERFACE)
target_include_directories(
  tiling_api_headers
  INTERFACE
    $<BUILD_INTERFACE:${ASCENDC_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${ASCENDC_DIR}/impl/aicore/adv_api>
    $<BUILD_INTERFACE:${ASCENDC_DIR}/include/aicore/adv_api>
    $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include>>
    $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/base>>
    $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:${ASCENDC_DIR}/../framework/tikcfw>>
    $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:${ASCENDC_DIR}/../framework/tikcfw/tiling>>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/ascendc>
    $<INSTALL_INTERFACE:include/ascendc/include/>
    $<INSTALL_INTERFACE:include/ascendc/include/highlevel_api>
    $<INSTALL_INTERFACE:include/ascendc/include/highlevel_api/lib>
    $<INSTALL_INTERFACE:include/ascendc/include/highlevel_api/impl>
    $<INSTALL_INTERFACE:include/ascendc/include/highlevel_api/tiling>)

target_link_libraries(tiling_api_headers
                      INTERFACE $<BUILD_INTERFACE:tikcfw_headers>)

add_library(
  tiling_api STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/activation/gelu_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/activation/geglu_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/activation/sigmoid_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/activation/swiglu_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/activation/softmax_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/activation/logsoftmax_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/activation/reglu_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv/conv3d_tiling_algorithm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv/conv3d_tiling_base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv/conv3d_tiling_util.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv/conv3d_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_backprop/conv3d_bp_filter_tiling_base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_backprop/conv3d_bp_filter_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_backprop/conv3d_bp_input_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_backprop/conv3d_bp_input_tiling_base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/filter/dropout_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hccl/hccl_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/index/arithprogression_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/clamp_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/acos_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/acosh_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/asin_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/asinh_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/atan_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/atanh_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/cos_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/cosh_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/erf_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/erfc_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/exp_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/frac_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/lgamma_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/digamma_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/log_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/sin_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/sinh_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/power_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/round_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/tan_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/tanh_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/trunc_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/axpy_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/hypot_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/ceil_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/floor_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/xor_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/cumsum_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/sign_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math/fmod_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul/bmm_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul/matmul_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul/matmul_tiling_base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul/matmul_tiling_algorithm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul/math_util.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/rmsnorm_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/batchnorm_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/deepnorm_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/layernorm_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/normalize_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/layernorm_grad_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/layernorm_grad_beta_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/groupnorm_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/welfordfinalize_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pad/pad_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pad/broadcast_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantization/ascend_dequant_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantization/ascend_quant_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantization/ascend_antiquant_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantization/quantize_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantization/dequantize_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantization/antiquantize_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce/mean_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce/reduce_xor_sum_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce/sum_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce/reduce_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/select/selectwithbytesmask_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sort/sort_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sort/topk_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transpose/confusion_transpose_tiling_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transpose/transdata_tiling.cpp
  $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<TARGET_OBJECTS:${ASCENDC_ADV_API_OBJ}>>)

target_compile_options(
  tiling_api
  PRIVATE -fvisibility=hidden
          -Wextra
          -Wfloat-equal
          -D_FORTIFY_SOURCE=2
          -O2
          $<$<STREQUAL:${PRODUCT_SIDE},host>:-DLOG_CPP>
          $<$<STREQUAL:${PRODUCT_SIDE},device>:-fPIC>
          $<$<STREQUAL:${PRODUCT_SIDE},device>:-fstack-protector-strong>
          $<$<STREQUAL:${PRODUCT_SIDE},device>:-fstack-protector-all>
          $<$<STREQUAL:${PRODUCT_SIDE},device>:-std=c++11>
          $<$<STREQUAL:${PRODUCT_SIDE},device>:-fvisibility-inlines-hidden>
          $<$<STREQUAL:${PRODUCT_SIDE},device>:-fvisibility=hidden>)

target_compile_definitions(
  tiling_api PRIVATE $<$<STREQUAL:${PRODUCT_SIDE},device>:_FORTIFY_SOURCE=2>)

target_link_libraries(
  tiling_api
  PRIVATE $<BUILD_INTERFACE:intf_pub>
          $<BUILD_INTERFACE:slog_headers>
          -Wl,--no-as-needed
          c_sec
          $<$<STREQUAL:${PRODUCT_SIDE},host>:alog>
          $<$<STREQUAL:${PRODUCT_SIDE},device>:slog>
          -Wl,--as-needed
          $<$<STREQUAL:${PRODUCT_SIDE},host>:platform>
          $<$<STREQUAL:${PRODUCT_SIDE},host>:exe_graph>
          $<$<STREQUAL:${PRODUCT_SIDE},host>:register>
  PUBLIC $<BUILD_INTERFACE:kernel_tiling_headers>
         $<$<STREQUAL:${PRODUCT_SIDE},host>:metadef_headers> tiling_api_headers)

if(BUILD_OPEN_PROJECT)
  set(ASCENDC_API_ADV_INSTALL_DIR ${INSTALL_LIBRARY_DIR}/lib64)
  set(ASCENDC_API_ADV_MULT_INSTALL_DIR
      ${INSTALL_LIBRARY_DIR}/devlib/${CMAKE_SYSTEM_PROCESSOR})
  set(ASCENDC_TILING_INCLUDE_INSTALL_DIR ${INSTALL_LIBRARY_DIR}/include)
else()
  set(ASCENDC_API_ADV_INSTALL_DIR ${INSTALL_LIBRARY_DIR}/tiling/)
  set(ASCENDC_API_ADV_MULT_INSTALL_DIR
      ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
  set(ASCENDC_TILING_INCLUDE_INSTALL_DIR ${INSTALL_LIBRARY_DIR}/ascendc/include)
endif()

install(TARGETS tiling_api ARCHIVE DESTINATION ${ASCENDC_API_ADV_INSTALL_DIR}
                                   OPTIONAL)

install(TARGETS tiling_api
        ARCHIVE DESTINATION ${ASCENDC_API_ADV_MULT_INSTALL_DIR} OPTIONAL)

configure_file(${ASCENDC_ADV_API_CMAKE_DIR}/tiling_headers.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/tiling_headers.cmake @ONLY)
install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/tiling_headers.cmake)

if(NOT BUILD_OPEN_PROJECT)
  install_package(
    PACKAGE
    tiling_api
    TARGETS
    tiling_api
    tiling_api_headers
    DIRECTORY
    ${ASCENDC_DIR}/include/aicore/adv_api
    DESTINATION
    ${INSTALL_INCLUDE_DIR}/ascendc/include/highlevel_api/lib
    DIRECTORY
    ${ASCENDC_DIR}/impl/aicore/adv_api/detail
    DESTINATION
    ${INSTALL_INCLUDE_DIR}/ascendc/include/highlevel_api/impl
    FILES
    ${ASCENDC_DIR}/include/aicore/adv_api/tiling_api.h
    DESTINATION
    ${INSTALL_INCLUDE_DIR}/ascendc/include/highlevel_api/tiling)

  install(SCRIPT ${ASCENDC_DIR}/cmake/tiling_directory.cmake)
endif()

if(BUILD_OPEN_PROJECT)
  set(install_script_dir ${CMAKE_CURRENT_BINARY_DIR}/install_scripts/)

  add_custom_target(
    generate_install_script ALL
    COMMAND rm -rf ${install_script_dir}
    COMMAND
      cp -rf
      ${ASCEND_CANN_PACKAGE_PATH}/tools/ascend_project/open_install_scripts
      ${install_script_dir}
    COMMAND chmod -R u+w ${install_script_dir}
    COMMAND echo "base_package=toolkit" > ${install_script_dir}/version.info
    COMMAND echo "backup_dir=${CMAKE_PROJECT_NAME}" >>
            ${install_script_dir}/version.info
    COMMAND echo "Version=${CANN_VERSION}" >>
            ${install_script_dir}/version.info)

  install(
    DIRECTORY ${install_script_dir}
    DESTINATION .
    FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ)

  set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
  set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
  set(CPACK_PACKAGE_DESCRIPTION "CPack ascendc-api-adv project")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack ascendc-api-adv project")
  set(CPACK_PACKAGE_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CPACK_PACKAGE_FILE_NAME
      "CANN-ascendc_api_adv-${CANN_VERSION}-linux.${CMAKE_SYSTEM_PROCESSOR}.run"
  )
  set(CPACK_GENERATOR External)
  set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
  set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
  set(CPACK_EXTERNAL_PACKAGE_SCRIPT
      ${ASCEND_CANN_PACKAGE_PATH}/tools/op_project_templates/ascendc/customize/cmake/makeself.cmake
  )
  set(CPACK_EXTERNAL_BUILT_PACKAGES
      ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME}
  )
  include(CPack)
endif()
