# Copyright (c) 2025 Huawei Technologies Co., Ltd. This file is a part of the CANN Open Software. Licensed under CANN
# Open Software License Agreement Version 1.0 (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF
# ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the License.
# ===============================================================================

# get the current absolute dir
get_filename_component(ASCENDC_ATVC_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

set(PRODUCT_TYPE_LIST
    ascend910B1_ATVC
)

add_compile_definitions(
  __CCE_KT_TEST__=1
  ASCENDC_CPU_DEBUG=1
  ASCENDC_DUMP=0 LOG_CPP
)

set(ASCENDC_SYS_DIR
  ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux
)

set(ASCENDC_DEP_HEADER_DIR
  ${ASCENDC_SYS_DIR}/include
  ${ASCENDC_SYS_DIR}/include/base
  ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/include
)

set(ASCEND_DEP_LIB_DIR
  ${ASCENDC_SYS_DIR}/lib64/
  ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/
)

set(CMAKE_CXX_STANDARD 17)

set(ASCENDC_ADV_API_INCLUDE ${ASCENDC_INCLUDE_DIR}/aicore/adv_api)
set(ASCENDC_UTILS_INCLUDE ${ASCENDC_INCLUDE_DIR}/aicore/utils)
set(ASCENDC_AICORE_IMPL ${ASCENDC_IMPL_DIR}/aicore)
set(ASCENDC_ADV_API_IMPL ${ASCENDC_IMPL_DIR}/aicore/adv_api)
set(ASCENDC_UTILS_IMPL ${ASCENDC_IMPL_DIR}/aicore/utils)

set(ASCENDC_ATVC_INCLUDE
  ${ASCENDC_DIR}/ops_templates/atvc/include
)

set(ASCENDC_TEST_HEADER_FILES
  ${ASCENDC_UNIT_TESTS_DIR}/common
  ${ASCENDC_ADV_API_INCLUDE}
  ${ASCENDC_UTILS_INCLUDE}
  ${ASCENDC_AICORE_IMPL}
  ${ASCENDC_ADV_API_IMPL}
  ${ASCENDC_UTILS_IMPL}
  ${ASCENDC_ATVC_INCLUDE}
  ${ASCENDC_SYS_DIR}/tikcpp/tikcfw
  ${ASCENDC_SYS_DIR}/tikcpp/tikcfw/interface
  ${ASCENDC_SYS_DIR}/tikcpp/tikcfw/impl
)

# testcases
set(ASCENDC_TEST_ascend910B1_ATVC_CASE_SRC_FILES
  ${ASCENDC_ATVC_TESTS_DIR}/test_add.cpp
)

foreach (product_type ${PRODUCT_TYPE_LIST})
  add_executable(
    ascendc_utest_${product_type}
    ${ASCENDC_ATVC_TESTS_DIR}/main.cpp
    $<$<STREQUAL:${product_type},ascend910B1_ATVC>:${ASCENDC_UNIT_TESTS_DIR}/common/k3_pvwrap.cpp>
    ${ASCENDC_TEST_${product_type}_CASE_SRC_FILES}
  )

  add_dependencies(ascendc_utest_${product_type} gen_kernel_tiling)

  if (${product_type} STREQUAL "ascend910B1_ATVC")
    target_compile_definitions(
      ascendc_utest_${product_type}
      PRIVATE UT_TEST
              __CCE_AICORE__=220
              __NPU_ARCH__=2201
              __DAV_C220__
              __DAV_C220_VEC__
    )
    target_link_directories(
      ascendc_utest_${product_type}
      PRIVATE ${ASCEND_DEP_LIB_DIR}
              ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910B1
              ${ASCENDC_SYS_DIR}/simulator/Ascend910B1/lib/
    )
  endif ()

  target_compile_definitions(
    ascendc_utest_${product_type}
    PRIVATE __CCE_KT_TEST__=1
            ASCENDC_CPU_DEBUG=1
            ASCENDC_OOM=1
  )

  target_include_directories(
    ascendc_utest_${product_type}
    PRIVATE ${ASCENDC_TEST_HEADER_FILES}
            ${ASCENDC_DEP_HEADER_DIR}
  )

  target_compile_options(
    ascendc_utest_${product_type}
    PRIVATE -fno-access-control
  )

  target_link_libraries(
    ascendc_utest_${product_type}
    PRIVATE $<BUILD_INTERFACE:intf_llt_pub>
            -Wl,--no-as-needed
            tikicpulib_npuchk
            tikicpulib_cceprint
            tikicpulib_stubreg
            tikcpp_debug
            $<$<STREQUAL:${product_type},ascend910B1_ATVC>:pem_davinci>
            ascendalog
            c_sec
            mmpa
            error_manager
            -Wl,--as-needed
  )

  run_llt_test(TARGET ascendc_utest_${product_type} TASK_NUM 1)
endforeach ()

# ###############################tiling ut#############################
# testcases
set(ASCENDC_TILING_TEST_ascend910B1_ATVC_CASE_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/tiling/test_atvc_tiling.cpp
)

# ascendc_tiling_utest
foreach (product_type ${PRODUCT_TYPE_LIST})
  add_executable(
    ascendc_tiling_utest_${product_type}
    ${ASCENDC_TILING_TEST_${product_type}_CASE_SRC_FILES}
    ${ASCENDC_UNIT_TESTS_DIR}/common/platform_stub.cpp
    ${ASCENDC_ATVC_TESTS_DIR}/tiling/main.cpp
  )

  if (${product_type} STREQUAL "ascend910B1_ATVC")
    target_compile_definitions(
      ascendc_tiling_utest_${product_type}
      PRIVATE UT_TEST __CCE_AICORE__=220)
  endif ()

  target_include_directories(
    ascendc_tiling_utest_${product_type}
    PRIVATE ${ASCENDC_DEP_HEADER_DIR}
            ${ASCENDC_TEST_HEADER_FILES}
  )

  target_link_directories(
    ascendc_tiling_utest_${product_type}
    PRIVATE ${ASCEND_DEP_LIB_DIR}
  )

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
