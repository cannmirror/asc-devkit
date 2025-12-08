# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
project(ascendc_devkit_ut)
cmake_minimum_required(VERSION 3.16.0)

set(PRODUCT_TYPE_LIST
    ascend910 ascend610 ascend310p ascend910B1_AIC ascend910B1_AIV ascend310B1 ascend610Lite ascend910_9599_AIV ascend910_9599_AIV_BASIC
    ascend910_9599_AIV_FRAMEWORK ascend910_9599_AIV_HIGH_LEVEL ascend910_9599_AIV_MICRO ascend910_9599_AIV_SIMT ascend910_9599_AIC
    mc62cm12aa_AIC mc62cm12aa_BASIC mc62cm12aa_FRAMEWORK mc62cm12aa_MICRO mc62cm12aa_SIMT)

set(TIKCPP_UT_PRODUCT_TYPE_LIST ascend910 ascend610 ascend310p ascend910B1_AIC ascend910B1_AIV ascend310B1 ascend610Lite
    ascend910_9599_AIV_BASIC ascend910_9599_AIV_FRAMEWORK ascend910_9599_AIV_MICRO ascend910_9599_AIV_SIMT ascend910_9599_AIC
    mc62cm12aa_AIC mc62cm12aa_BASIC mc62cm12aa_FRAMEWORK mc62cm12aa_MICRO mc62cm12aa_SIMT)
# set(TIKCPP_UT_PRODUCT_TYPE_LIST ascend910)

add_compile_definitions(ASCENDC_CPU_DEBUG=1 ASCENDC_DUMP=0)

message(STATUS "tikcpp_ut_test compile start")
set(CMAKE_CXX_STANDARD 17)

set(LOCAL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ASCENDC_SYS_DIR ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux)

############# gen tiling head file
set(tiling_script ${LOCAL_DIR}/scripts/gen_kernel_tiling_data_def.py)
set(TILING_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(tiling_gen_file ${TILING_GEN_DIR}/kernel_tiling/kernel_tiling.h)
set(tiling_data_def_path ${ASCENDC_INCLUDE_DIR}/adv_api)

add_custom_command(OUTPUT ${tiling_gen_file}
    COMMAND python3  ${tiling_script} ${tiling_data_def_path} ${tiling_gen_file}
    DEPENDS ${tiling_script} ${tiling_data_def_path})

add_custom_target(gen_ut_kernel_tiling ALL
    DEPENDS ${tiling_gen_file})
##################################################

####################################################################
####################################################################
set(ASCENDC_TEST_HEADER_FILES
    ${ASCENDC_SYS_DIR}/include
    ${ASCENDC_SYS_DIR}/include/base
    ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/include
    ${LOCAL_DIR}/../common
    ${LOCAL_DIR}/common/
    ${LOCAL_DIR}/testcase/
    ${ASCENDC_DIR}
    ${ASCENDC_DIR}/include/
    ${ASCENDC_DIR}/include/micro_api/
    ${ASCENDC_DIR}/include/simt_api/
    ${ASCENDC_DIR}/include/basic_api/
    ${ASCENDC_DIR}/include/utils/
    ${ASCENDC_DIR}/include/c_api/
    ${ASCENDC_DIR}/impl/
    ${ASCENDC_DIR}/impl/c_api/
    ${ASCENDC_DIR}/impl/basic_api/
    ${ASCENDC_DIR}/impl/simt_api/
    ${ASCENDC_DIR}/impl/micro_api/
    ${ASCENDC_SYS_DIR}/tikcpp/tikcfw
    ${ASCENDC_SYS_DIR}/tikcpp/tikcfw/interface
    ${ASCENDC_SYS_DIR}/tikcpp/tikcfw/impl
    ${ASCENDC_SYS_DIR}/ascendc/include/highlevel_api/
    ${CMAKE_CURRENT_BINARY_DIR}/stub
    ${TILING_GEN_DIR}
)

set(ASCEND_DEP_LIB_DIR
    ${ASCENDC_SYS_DIR}/lib64/
    ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/
)

# common ut src files
file(GLOB ASCENDC_TEST_COMMON_SRC_FILES
    ${LOCAL_DIR}/common/common.cpp
)

# common test cases
file(GLOB ASCENDC_TEST_COMMON_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/common/*.cpp
    ${LOCAL_DIR}/testcase/tikcpp_case_common/*.cpp
)

# ascend910 test cases
file(GLOB ASCENDC_TEST_ascend910_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910/*.cpp
)

# ascend610 test cases
file(GLOB ASCENDC_TEST_ascend610_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend610/*.cpp
)

# ascend310p test cases
file(GLOB ASCENDC_TEST_ascend310p_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend310p/*.cpp
)

# ascend910B1 aic c api test cases
file(GLOB ASCENDC_TEST_ascend910B1_AIC_CAPI_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/c_api/asc_2201/cube_compute/*.cpp
    ${LOCAL_DIR}/testcase/c_api/asc_2201/cube_dmamove/*.cpp
)

# ascend910B1 aiv c api test cases
file(GLOB ASCENDC_TEST_ascend910B1_AIV_CAPI_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/c_api/asc_2201/simd_atomic/*.cpp
    ${LOCAL_DIR}/testcase/c_api/asc_2201/sync/*.cpp
    ${LOCAL_DIR}/testcase/c_api/asc_2201/sys_var/*.cpp
    ${LOCAL_DIR}/testcase/c_api/asc_2201/vector_compute/*.cpp
    ${LOCAL_DIR}/testcase/c_api/asc_2201/vector_datamove/*.cpp
)

# ascend910B1 aiv test cases
file(GLOB ASCENDC_TEST_ascend910B1_AIV_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_data_copy_slice.cpp
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_tscm_data.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910b1/tikcpp_case_ascend910b1_aiv/*.cpp
)

# ascend910B1 aic test cases
file(GLOB ASCENDC_TEST_ascend910B1_AIC_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910b1/tikcpp_case_ascend910b1_aic/*.cpp
    ${LOCAL_DIR}/testcase/matmul_api/tikcpp_case_ascend910b1/tikcpp_case_ascend910b1_aic/*.cpp
)

# ascend310B1 test cases
file(GLOB ASCENDC_TEST_ascend310B1_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_kernel_pop_stack_buffer.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend310b1/*.cpp
    ${LOCAL_DIR}/testcase/matmul_api/tikcpp_case_ascend310b1/*.cpp
)

# ascend610Lite test cases
file(GLOB ASCENDC_TEST_ascend610Lite_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_kernel_pop_stack_buffer.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend610lite/*.cpp
)

# ascend910_9599 basic test cases
file(GLOB ASCENDC_TEST_ascend910_9599_AIV_BASIC_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_kernel_pop_stack_buffer.cpp
    ${LOCAL_DIR}/testcase/common/test_kernel_tensor.cpp
    ${LOCAL_DIR}/testcase/common/test_tikcpp_fp16.cpp
    ${LOCAL_DIR}/testcase/common/test_type_check.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910_9599/*.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910_9599/tikcpp_case_ascend910_9599_aiv/tikcpp_case_ascend910_9599_aiv_basic/*.cpp
)

# ascend910_9599 framework test cases
file(GLOB ASCENDC_TEST_ascend910_9599_AIV_FRAMEWORK_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_kernel_pop_stack_buffer.cpp
    ${LOCAL_DIR}/testcase/common/test_kernel_tensor.cpp
    ${LOCAL_DIR}/testcase/common/test_tikcpp_fp16.cpp
    ${LOCAL_DIR}/testcase/common/test_type_check.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910_9599/*.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910_9599/tikcpp_case_ascend910_9599_aiv/tikcpp_case_ascend910_9599_aiv_framework/*.cpp
)

# ascend910_9599 high_level test cases
file(GLOB ASCENDC_TEST_ascend910_9599_AIV_HIGH_LEVEL_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_kernel_pop_stack_buffer.cpp
    ${LOCAL_DIR}/testcase/common/test_kernel_tensor.cpp
    ${LOCAL_DIR}/testcase/common/test_tikcpp_fp16.cpp
    ${LOCAL_DIR}/testcase/common/test_type_check.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910_9599/*.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910_9599/tikcpp_case_ascend910_9599_aiv/tikcpp_case_ascend910_9599_aiv_high_level/*.cpp
)

# ascend910_9599 micro test cases
file(GLOB ASCENDC_TEST_ascend910_9599_AIV_MICRO_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_kernel_pop_stack_buffer.cpp
    ${LOCAL_DIR}/testcase/common/test_kernel_tensor.cpp
    ${LOCAL_DIR}/testcase/common/test_tikcpp_fp16.cpp
    ${LOCAL_DIR}/testcase/common/test_type_check.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910_9599/*.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910_9599/tikcpp_case_ascend910_9599_aiv/tikcpp_case_ascend910_9599_aiv_micro/*.cpp
)

# ascend910_9599 simt test cases
file(GLOB ASCENDC_TEST_ascend910_9599_AIV_SIMT_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_kernel_pop_stack_buffer.cpp
    ${LOCAL_DIR}/testcase/common/test_kernel_tensor.cpp
    ${LOCAL_DIR}/testcase/common/test_tikcpp_fp16.cpp
    ${LOCAL_DIR}/testcase/common/test_type_check.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910_9599/*.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910_9599/tikcpp_case_ascend910_9599_aiv/tikcpp_case_ascend910_9599_aiv_simt/*.cpp
)

# ascend910_9599 aic test cases
file(GLOB ASCENDC_TEST_ascend910_9599_AIC_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/common/test_kernel_tensor.cpp
    ${LOCAL_DIR}/testcase/common/test_tikcpp_fp16.cpp
    ${LOCAL_DIR}/testcase/common/test_type_check.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910_9599/*.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_ascend910_9599/tikcpp_case_ascend910_9599_aic/*.cpp
)

# mc62cm12aa basic test cases
file(GLOB ASCENDC_TEST_mc62cm12aa_BASIC_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_kernel_pop_stack_buffer.cpp
    ${LOCAL_DIR}/testcase/common/test_kernel_tensor.cpp
    ${LOCAL_DIR}/testcase/common/test_tikcpp_fp16.cpp
    ${LOCAL_DIR}/testcase/common/test_type_check.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_mc62cm12aa/*.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_mc62cm12aa/tikcpp_case_mc62cm12aa_basic/*.cpp
)

# mc62cm12aa framework test cases
file(GLOB ASCENDC_TEST_mc62cm12aa_FRAMEWORK_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_kernel_pop_stack_buffer.cpp
    ${LOCAL_DIR}/testcase/common/test_kernel_tensor.cpp
    ${LOCAL_DIR}/testcase/common/test_tikcpp_fp16.cpp
    ${LOCAL_DIR}/testcase/common/test_type_check.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_mc62cm12aa/*.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_mc62cm12aa/tikcpp_case_mc62cm12aa_framework/*.cpp
)

# mc62cm12aa micro test cases
file(GLOB ASCENDC_TEST_mc62cm12aa_MICRO_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_kernel_pop_stack_buffer.cpp
    ${LOCAL_DIR}/testcase/common/test_kernel_tensor.cpp
    ${LOCAL_DIR}/testcase/common/test_tikcpp_fp16.cpp
    ${LOCAL_DIR}/testcase/common/test_type_check.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_mc62cm12aa/*.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_mc62cm12aa/tikcpp_case_mc62cm12aa_micro/*.cpp
)

# mc62cm12aa simt test cases
file(GLOB ASCENDC_TEST_mc62cm12aa_SIMT_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_case_common/test_kernel_pop_stack_buffer.cpp
    ${LOCAL_DIR}/testcase/common/test_kernel_tensor.cpp
    ${LOCAL_DIR}/testcase/common/test_tikcpp_fp16.cpp
    ${LOCAL_DIR}/testcase/common/test_type_check.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_mc62cm12aa/*.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_mc62cm12aa/tikcpp_case_mc62cm12aa_simt/*.cpp
)

# mc62cm12aa aic test cases
file(GLOB ASCENDC_TEST_mc62cm12aa_AIC_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/common/test_kernel_tensor.cpp
    ${LOCAL_DIR}/testcase/common/test_tikcpp_fp16.cpp
    ${LOCAL_DIR}/testcase/common/test_type_check.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_mc62cm12aa/*.cpp
    ${LOCAL_DIR}/testcase/basic_api/tikcpp_case_mc62cm12aa/tikcpp_case_mc62cm12aa_aic/*.cpp
)

file(GLOB ASCENDC_API_CHECK_CASE_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_api_check/*.cpp
)

foreach(product_type ${TIKCPP_UT_PRODUCT_TYPE_LIST})
    add_executable(tikcpp_utest_${product_type}
        ${ASCENDC_TEST_COMMON_SRC_FILES}
        ${LOCAL_DIR}/main_global.cpp
        ${LOCAL_DIR}/../common/alog_stub.cpp
        ${LOCAL_DIR}/../common/dlog_stub.cpp
        $<$<STREQUAL:${product_type},ascend910>:${ASCENDC_TEST_COMMON_CASE_SRC_FILES};${ASCENDC_API_CHECK_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},ascend910>:${LOCAL_DIR}/../common/tik_pv_wrapper.cpp>
        $<$<STREQUAL:${product_type},ascend610>:${ASCENDC_TEST_COMMON_CASE_SRC_FILES};${ASCENDC_API_CHECK_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},ascend610>:${LOCAL_DIR}/../common/tik_pv_wrapper.cpp>
        $<$<STREQUAL:${product_type},ascend310p>:${ASCENDC_TEST_COMMON_CASE_SRC_FILES};${ASCENDC_API_CHECK_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},ascend310p>:${LOCAL_DIR}/../common/tik_pv_wrapper.cpp>
        $<$<STREQUAL:${product_type},ascend910B1_AIC>:${ASCENDC_API_CHECK_CASE_SRC_FILES};${ASCENDC_TEST_ascend910B1_AIC_CAPI_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},ascend910B1_AIC>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend910B1_AIV>:${ASCENDC_API_CHECK_CASE_SRC_FILES};${ASCENDC_TEST_ascend910B1_AIV_CAPI_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},ascend910B1_AIV>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend310B1>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend610Lite>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV_BASIC>:${ASCENDC_API_CHECK_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV_BASIC>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV_FRAMEWORK>:${ASCENDC_API_CHECK_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV_FRAMEWORK>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV_MICRO>:${ASCENDC_API_CHECK_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV_MICRO>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV_SIMT>:${ASCENDC_API_CHECK_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV_SIMT>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},ascend910_9599_AIC>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},mc62cm12aa_AIC>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},mc62cm12aa_BASIC>:${ASCENDC_API_CHECK_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},mc62cm12aa_BASIC>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},mc62cm12aa_FRAMEWORK>:${ASCENDC_API_CHECK_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},mc62cm12aa_FRAMEWORK>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},mc62cm12aa_MICRO>:${ASCENDC_API_CHECK_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},mc62cm12aa_MICRO>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        $<$<STREQUAL:${product_type},mc62cm12aa_SIMT>:${ASCENDC_API_CHECK_CASE_SRC_FILES}>
        $<$<STREQUAL:${product_type},mc62cm12aa_SIMT>:${LOCAL_DIR}/../common/k3_pvwrap.cpp>
        ${ASCENDC_TEST_${product_type}_CASE_SRC_FILES}
    )

    add_dependencies(tikcpp_utest_${product_type} gen_ut_kernel_tiling mockcpp)

    # add soc version flags
    if(${product_type} STREQUAL "ascend910")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=100
            __NPU_ARCH__=1001
            __DAV_C100__
            __UT_CHIP_VER__=910
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910A
            ${ASCENDC_SYS_DIR}/simulator/Ascend910A/lib/
        )
    elseif(${product_type} STREQUAL "ascend610")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=200
            __NPU_ARCH__=2002
            __DAV_M200__
            __UT_CHIP_VER__=610
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend610
            ${ASCENDC_SYS_DIR}/simulator/Ascend610/lib/
        )
    elseif(${product_type} STREQUAL "ascend310p")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=200
            __NPU_ARCH__=2002
            __DAV_M200__
            __UT_CHIP_VER__=710
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend310P1
            ${ASCENDC_SYS_DIR}/simulator/Ascend310P1/lib/
        )
    elseif(${product_type} STREQUAL "ascend910B1_AIC")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=220
            __NPU_ARCH__=2201
            __DAV_C220__
            __DAV_C220_CUBE__
            __DAV_CUBE__
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910B1/
            ${ASCENDC_SYS_DIR}/simulator/Ascend910B1/lib/
        )
    elseif(${product_type} STREQUAL "ascend910B1_AIV")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=220
            __NPU_ARCH__=2201
            __DAV_C220__
            __DAV_C220_VEC__
            __DAV_VEC__
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910B1
            ${ASCENDC_SYS_DIR}/simulator/Ascend910B1/lib/
        )
    elseif(${product_type} STREQUAL "ascend310B1")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=300
            __NPU_ARCH__=3002
            __DAV_M300__
            __UT_CHIP_VER__=300
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend310B1/
            ${ASCENDC_SYS_DIR}/simulator/Ascend310B1/lib/
        )
    elseif(${product_type} STREQUAL "ascend610Lite")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __DAV_M310__
            __NPU_ARCH__=3102
            __UT_CHIP_VER__=310
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend610Lite/
            ${ASCENDC_SYS_DIR}/simulator/Ascend610Lite/lib/
        )
    elseif(${product_type} STREQUAL "ascend910_9599_AIV_BASIC")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __DAV_C310__
            __NPU_ARCH__=3101
            __ST_CHIP_VER__=310
            __DAV_C310_VEC__
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910_9599
            ${ASCENDC_SYS_DIR}/simulator/Ascend910_9599/lib/
        )
    elseif(${product_type} STREQUAL "ascend910_9599_AIV_FRAMEWORK")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __DAV_C310__
            __NPU_ARCH__=3101
            __ST_CHIP_VER__=310
            __DAV_C310_VEC__
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910_9599
            ${ASCENDC_SYS_DIR}/simulator/Ascend910_9599/lib/
        )
    elseif(${product_type} STREQUAL "ascend910_9599_AIV_HIGH_LEVEL")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __DAV_C310__
            __NPU_ARCH__=3101
            __ST_CHIP_VER__=310
            __DAV_C310_VEC__
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910_9599
            ${ASCENDC_SYS_DIR}/simulator/Ascend910_9599/lib/
        )
    elseif(${product_type} STREQUAL "ascend910_9599_AIV_MICRO")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __DAV_C310__
            __NPU_ARCH__=3101
            __ST_CHIP_VER__=310
            __DAV_C310_VEC__
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910_9599
            ${ASCENDC_SYS_DIR}/simulator/Ascend910_9599/lib/
        )
    elseif(${product_type} STREQUAL "ascend910_9599_AIV_SIMT")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __DAV_C310__
            __NPU_ARCH__=3101
            __ST_CHIP_VER__=310
            __DAV_C310_VEC__
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910_9599
            ${ASCENDC_SYS_DIR}/simulator/Ascend910_9599/lib/
        )
    elseif(${product_type} STREQUAL "ascend910_9599_AIC")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __DAV_C310__
            __NPU_ARCH__=3101
            __UT_CHIP_VER__=310
            __DAV_C310_CUBE__
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/Ascend910_9599
            ${ASCENDC_SYS_DIR}/simulator/Ascend910_9599/lib/
        )
    elseif(${product_type} STREQUAL "mc62cm12aa_AIC")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __NPU_ARCH__=5102
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/MC62CM12AA
            ${ASCENDC_SYS_DIR}/simulator/MC62CM12AA/lib/
        )
    elseif(${product_type} STREQUAL "mc62cm12aa_BASIC")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __NPU_ARCH__=5102
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/MC62CM12AA
            ${ASCENDC_SYS_DIR}/simulator/MC62CM12AA/lib/
        )
    elseif(${product_type} STREQUAL "mc62cm12aa_FRAMEWORK")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __NPU_ARCH__=5102
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/MC62CM12AA
            ${ASCENDC_SYS_DIR}/simulator/MC62CM12AA/lib/
        )
    elseif(${product_type} STREQUAL "mc62cm12aa_MICRO")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __NPU_ARCH__=5102
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/MC62CM12AA
            ${ASCENDC_SYS_DIR}/simulator/MC62CM12AA/lib/
        )
    elseif(${product_type} STREQUAL "mc62cm12aa_SIMT")
        target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
            UT_TEST
            __NPU_ARCH__=5102
        )
        target_link_directories(tikcpp_utest_${product_type} PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/MC62CM12AA
            ${ASCENDC_SYS_DIR}/simulator/MC62CM12AA/lib/
        )
    endif()

    target_compile_definitions(tikcpp_utest_${product_type} PRIVATE
        ASCENDC_CPU_DEBUG=1
        ASCENDC_OOM=1
    )

    target_include_directories(tikcpp_utest_${product_type} PRIVATE
        ${LOCAL_DIR}/stub/
        ${ASCENDC_TEST_HEADER_FILES}
        ${ASCENDC_SYS_DIR}/include/base/
        ${ASCENDC_SYS_DIR}/include/metadef/
        mmpa_headers
        slog_headers
    )

    target_compile_options(tikcpp_utest_${product_type} PRIVATE
        -g
        -fno-access-control
        -Werror
    )

    target_link_libraries(tikcpp_utest_${product_type} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_basic>
        -Wl,--no-as-needed
        tikicpulib_npuchk
        tikicpulib_cceprint
        tikicpulib_stubreg
        tikcpp_debug
        $<$<STREQUAL:${product_type},ascend910>:_pvmodel>
        $<$<STREQUAL:${product_type},ascend610>:_pvmodel>
        $<$<STREQUAL:${product_type},ascend310p>:_pvmodel>
        $<$<STREQUAL:${product_type},ascend910B1_AIC>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend910B1_AIV>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend310B1>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend610Lite>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV_BASIC>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV_FRAMEWORK>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV_MICRO>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend910_9599_AIV_SIMT>:pem_davinci>
        $<$<STREQUAL:${product_type},ascend910_9599_AIC>:pem_davinci>
        $<$<STREQUAL:${product_type},mc62cm12aa_AIC>:pem_davinci>
        $<$<STREQUAL:${product_type},mc62cm12aa_BASIC>:pem_davinci>
        $<$<STREQUAL:${product_type},mc62cm12aa_FRAMEWORK>:pem_davinci>
        $<$<STREQUAL:${product_type},mc62cm12aa_MICRO>:pem_davinci>
        $<$<STREQUAL:${product_type},mc62cm12aa_SIMT>:pem_davinci>
        c_sec
        mmpa
        error_manager
        -Wl,--as-needed
    )

    run_llt_test(
        TARGET tikcpp_utest_${product_type}
        TASK_NUM 1
    )
endforeach()

############################################tiling ut###################################
file(GLOB ASCENDC_TILING_TEST_SRC_FILES
    ${LOCAL_DIR}/testcase/tikcpp_tiling/*.cpp
)

file(GLOB ASCENDC_TILING_SRC_FILES
    ${ASCENDC_IMPL_DIR}/utils/platform/*.cpp
    ${ASCENDC_IMPL_DIR}/utils/context/*.cpp
    ${ASCENDC_IMPL_DIR}/utils/stub/*.cpp
    ${ASCENDC_IMPL_DIR}/utils/tiling/*.cpp
    ${ASCENDC_IMPL_DIR}/utils/tiling/platform/*.cpp
)

# tikcpp_tiling_utest
foreach(product_type ${PRODUCT_TYPE_LIST})
    add_executable(tikcpp_tiling_utest_${product_type}
        ${LOCAL_DIR}/main.cpp
        ${ASCENDC_TILING_SRC_FILES}
        ${ASCENDC_TILING_TEST_SRC_FILES}
        ${LOCAL_DIR}/../common/platform_stub.cpp
        ${LOCAL_DIR}/common/common.cpp
        ${LOCAL_DIR}/../common/alog_stub.cpp
        ${LOCAL_DIR}/../common/dlog_stub.cpp
    )
    add_dependencies(tikcpp_tiling_utest_${product_type}  gen_ut_kernel_tiling)
    # add soc version flags
    if(${product_type} STREQUAL "ascend910")
        target_compile_definitions(tikcpp_tiling_utest_${product_type} PRIVATE
        UT_TEST
        __CCE_AICORE__=100
        __NPU_ARCH__=1001
    )
    elseif(${product_type} STREQUAL "ascend610")
        target_compile_definitions(tikcpp_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=200
            __NPU_ARCH__=2002
        )
    elseif(${product_type} STREQUAL "ascend310p")
        target_compile_definitions(tikcpp_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=200
            __NPU_ARCH__=2002
        )
    elseif(${product_type} STREQUAL "ascend910B1_AIC")
        target_compile_definitions(tikcpp_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=220
            __NPU_ARCH__=2201
        )
    elseif(${product_type} STREQUAL "ascend910B1_AIV")
        target_compile_definitions(tikcpp_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=220
            __NPU_ARCH__=2201
        )
    elseif(${product_type} STREQUAL "ascend310B1")
        target_compile_definitions(tikcpp_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=300
            __NPU_ARCH__=3002
        )
    elseif(${product_type} STREQUAL "ascend610Lite")
        target_compile_definitions(tikcpp_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __NPU_ARCH__=3102
        )
    elseif(${product_type} STREQUAL "ascend910_9599_AIC")
        target_compile_definitions(tikcpp_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __DAV_C310__
            __NPU_ARCH__=3101
        )
    elseif(${product_type} STREQUAL "ascend910_9599_AIV")
        target_compile_definitions(tikcpp_tiling_utest_${product_type} PRIVATE
            UT_TEST
            __CCE_AICORE__=310
            __DAV_C310__
            __NPU_ARCH__=3101
        )
    endif()

    target_link_directories(tikcpp_tiling_utest_${product_type} PRIVATE
        ${ASCEND_DEP_LIB_DIR}
    )

    target_include_directories(tikcpp_tiling_utest_${product_type} PRIVATE
        ${ASCENDC_DIR}/
        ${ASCENDC_IMPL_DIR}/
        ${ASCENDC_SYS_DIR}/include/
        ${ASCENDC_SYS_DIR}/pkg_inc/
        ${ASCENDC_SYS_DIR}/pkg_inc/base/
        ${ASCENDC_INCLUDE_DIR}/
        ${ASCENDC_INCLUDE_DIR}/utils/context/
        ${ASCENDC_INCLUDE_DIR}/utils/tiling/platform
        ${ASCENDC_INCLUDE_DIR}/utils/stub
        ${ASCENDC_INCLUDE_DIR}/utils/stub/common
    )

    target_compile_options(tikcpp_tiling_utest_${product_type} PRIVATE
        -g
        -Werror
    )

    target_link_libraries(tikcpp_tiling_utest_${product_type} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_basic>
        -Wl,--no-as-needed
        c_sec
        metadef
        graph
        graph_base
        register
        opp_registry
        dl
        -Wl,--as-needed
    )

    run_llt_test(
        TARGET tikcpp_tiling_utest_${product_type}
        TASK_NUM 1
    )
endforeach()


# ############################################ascendc_kernel ut###################################
set(ASCEND_KERNEL_TEST_HEADERS_FILES
    ${LOCAL_DIR}/../common/
    ${ASCENDC_DIR}/include
    ${ASCENDC_DIR}/tools/build/common/
    ${ASCENDC_DIR}/tools/build/elf_tool/
    ${ASCENDC_DIR}/tools/build/asc_rt/
    ${ASCENDC_DIR}/tools/build/asc_pack_kernel/
    ${ASCENDC_SYS_DIR}/include
    ${ASCENDC_SYS_DIR}/include/base/
    ${ASCENDC_SYS_DIR}/include/metadef/
    ${ASCENDC_DIR}/tikcpulib/inc/
    ${ASCENDC_DIR}/tikcpulib/cpulib/api_check
    ${ASCENDC_DIR}/tikcpulib/cpulib/api_check/inc
    ${ASCENDC_DIR}/tools/build/common

)

file(GLOB ASCENDC_KERNEL_SRC_FILES
    ${ASCENDC_DIR}/impl/aicpu_api/*.cpp
    ${ASCENDC_DIR}/tools/build/elf_tool/*.c
    ${ASCENDC_DIR}/tools/build/asc_rt/*.cpp
    ${ASCENDC_DIR}/tools/build/asc_pack_kernel/*.c
    ${ASCENDC_DIR}/tests/unit/basic_api/ut/testcase/ascendc_kernel/*.cpp
    ${LOCAL_DIR}/../ascendc_kernel_header/*.cpp
    ${LOCAL_DIR}/main.cpp
)

file(GLOB TEST_ASCENDC_KERNEL_SRC_FILES
    ${LOCAL_DIR}/testcase/ascendc_kernel/*.cpp
)

add_executable(ascendc_utest_kernel
    ${ASCENDC_KERNEL_SRC_FILES}
    ${TEST_ASCENDC_KERNEL_SRC_FILES}
)
add_dependencies(ascendc_utest_kernel gen_ut_kernel_tiling)
target_compile_definitions(ascendc_utest_kernel PRIVATE
    UT_TEST
)
target_compile_definitions(ascendc_utest_kernel PRIVATE ASCENDC_CPU_DEBUG=1)
target_include_directories(ascendc_utest_kernel PRIVATE
    ${ASCEND_KERNEL_TEST_HEADERS_FILES}
    ${ASCENDC_SYS_DIR}/include/base/
    ${ASCENDC_SYS_DIR}/include
    ${ASCENDC_SYS_DIR}/include/acl/
    ${ASCENDC_SYS_DIR}/pkg_inc/
    ${ASCENDC_SYS_DIR}/pkg_inc/base/
    ${ASCENDC_SYS_DIR}/pkg_inc/runtime/
    ${ASCENDC_SYS_DIR}/pkg_inc/runtime/runtime/
    ${ASCENDC_SYS_DIR}/pkg_inc/profiling/
    ${ASCENDC_DIR}/tools/build/asc_rt/
    ${ASCENDC_DIR}/tools/build/elf_tool/
)
target_compile_options(ascendc_utest_kernel PRIVATE
    -g
    -Werror
)
target_link_directories(ascendc_utest_kernel PRIVATE
    ${ASCEND_DEP_LIB_DIR}
)
target_link_libraries(ascendc_utest_kernel PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_basic>
    # $<BUILD_INTERFACE:mmpa_headers>
    # $<BUILD_INTERFACE:metadef_headers>
    -Wl,--no-as-needed
    c_sec
    mmpa
    error_manager
    -Wl,--as-needed
)

run_llt_test(
    TARGET ascendc_utest_kernel
    TASK_NUM 1
)

# ############################################ ascc ut ############################################
# # bisheng流程:  bishengcc参数解析，调用ascendfe

# # bishengcc arguments
set(ASCC_TEST_HEADERS_FILES
    ${LOCAL_DIR}/../common/
    ${ASCENDC_DIR}/tools/ascc/ascc/include
    ${ASCENDC_DIR}/tools/ascc/common/include
    ${LOCAL_DIR}/testcase/ascc/
)

file(GLOB ASCC_SRC_FILES
    ${ASCENDC_DIR}/tools/ascc/ascc/src/*.cpp
    ${ASCENDC_DIR}/tools/ascc/common/src/*.cpp
)

file(GLOB TEST_ASCC_SRC_FILES
    ${LOCAL_DIR}/main.cpp
    ${LOCAL_DIR}/testcase/ascc/ascc/*.cpp
    ${LOCAL_DIR}/testcase/ascc/llvm/Support/*.cpp
)

add_executable(ascendc_utest_ascc
    ${ASCC_SRC_FILES}
    ${TEST_ASCC_SRC_FILES}
)
add_dependencies(ascendc_utest_ascc gen_ut_kernel_tiling)
target_compile_definitions(ascendc_utest_ascc PRIVATE
    UT_TEST
)
target_compile_definitions(ascendc_utest_ascc PRIVATE ASCENDC_CPU_DEBUG=1)
target_include_directories(ascendc_utest_ascc PRIVATE
    ${ASCC_TEST_HEADERS_FILES}
    ${ASCENDC_SYS_DIR}/include
    ${ASCENDC_SYS_DIR}/include/base/
)
target_compile_options(ascendc_utest_ascc PRIVATE
    -g
    -Werror
)
target_link_directories(ascendc_utest_ascc PRIVATE
    ${ASCEND_DEP_LIB_DIR}
)
target_link_libraries(ascendc_utest_ascc PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_basic>
    -Wl,--no-as-needed
    c_sec
    -Wl,--as-needed
)

run_llt_test(
    TARGET ascendc_utest_ascc
    TASK_NUM 1
)

set(ASCENDFE_TEST_HEADERS_FILES
    ${LOCAL_DIR}/../common/
    ${ASCENDC_DIR}/tools/ascc/ascendfe/include
    ${ASCENDC_DIR}/tools/ascc/common/include
    ${LOCAL_DIR}/testcase/ascc/
)

file(GLOB ASCENDFE_SRC_FILES
    ${LOCAL_DIR}/main.cpp
    ${ASCENDC_DIR}/tools/ascc/common/src/*.cpp
    ${ASCENDC_DIR}/tools/ascc/ascendfe/src/*.cpp
)

file(GLOB TEST_ASCENDFE_SRC_FILES
    ${LOCAL_DIR}/testcase/ascc/llvm/Support/*.cpp
    ${LOCAL_DIR}/testcase/ascc/ascendfe/*.cpp
)

add_executable(ascendc_utest_ascendfe
    ${ASCENDFE_SRC_FILES}
    ${TEST_ASCENDFE_SRC_FILES}
)
add_dependencies(ascendc_utest_ascendfe gen_ut_kernel_tiling)
target_compile_definitions(ascendc_utest_ascendfe PRIVATE
    UT_TEST
)
target_compile_definitions(ascendc_utest_ascendfe PRIVATE ASCENDC_CPU_DEBUG=1)
target_include_directories(ascendc_utest_ascendfe PRIVATE
    ${ASCENDFE_TEST_HEADERS_FILES}
    ${ASCENDC_SYS_DIR}/include
    ${ASCENDC_SYS_DIR}/include/base/
)
target_compile_options(ascendc_utest_ascendfe PRIVATE
    -g
    -Werror
)
target_link_directories(ascendc_utest_ascendfe PRIVATE
    ${ASCEND_DEP_LIB_DIR}
)
target_link_libraries(ascendc_utest_ascendfe PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_basic>
    -Wl,--no-as-needed
    c_sec
    -Wl,--as-needed
)

run_llt_test(
    TARGET ascendc_utest_ascendfe
    TASK_NUM 1
)

# # libasc_plugin: 独立流程
set(ASC_PLUGIN_TEST_HEADERS_FILES
    ${LOCAL_DIR}/../common/
    ${ASCENDC_DIR}/tools/ascc/asc_plugin/include/internal
    ${ASCENDC_DIR}/tools/ascc/asc_plugin/include/external
    ${LOCAL_DIR}/testcase/ascc/
)

file(GLOB ASC_PLUGIN_SRC_FILES
    ${ASCENDC_DIR}/tools/ascc/asc_plugin/src/*.cpp
)

file(GLOB TEST_ASC_PLUGIN_SRC_FILES
    ${LOCAL_DIR}/main.cpp
    ${LOCAL_DIR}/testcase/ascc/llvm/Support/*.cpp
    ${LOCAL_DIR}/testcase/ascc/asc_plugin/*.cpp
)

add_executable(ascendc_utest_asc_plugin
    ${ASC_PLUGIN_SRC_FILES}
    ${TEST_ASC_PLUGIN_SRC_FILES}
)
add_dependencies(ascendc_utest_asc_plugin  gen_ut_kernel_tiling)
target_compile_definitions(ascendc_utest_asc_plugin PRIVATE
    UT_TEST
)
target_compile_definitions(ascendc_utest_asc_plugin PRIVATE ASCENDC_CPU_DEBUG=1)
target_include_directories(ascendc_utest_asc_plugin PRIVATE
    ${ASC_PLUGIN_TEST_HEADERS_FILES}
    ${JSON_PATH}/include/
    ${ASCENDC_SYS_DIR}/include
)
target_compile_options(ascendc_utest_asc_plugin PRIVATE
    -g
    -Werror
)
target_link_directories(ascendc_utest_asc_plugin PRIVATE
    ${ASCEND_DEP_LIB_DIR}
)
target_link_libraries(ascendc_utest_asc_plugin PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_basic>
    -Wl,--no-as-needed
    c_sec
    dl
    -Wl,--as-needed
)

run_llt_test(
    TARGET ascendc_utest_asc_plugin
    TASK_NUM 1
)

# ############################################ aclrtc ut ############################################
set(ACLRTC_TEST_HEADERS_FILES
    ${LOCAL_DIR}/../common/
    ${ASCENDC_DIR}/tools/aclrtc/include
)

file(GLOB ACLRTC_SRC_FILES
    ${ASCENDC_DIR}/tools/aclrtc/src/*.cpp
)

file(GLOB TEST_ACLRTC_SRC_FILES
    ${LOCAL_DIR}/main.cpp
    ${LOCAL_DIR}/testcase/aclrtc/stub/*.cpp
    ${LOCAL_DIR}/testcase/aclrtc/*.cpp
)

add_executable(ascendc_utest_aclrtc
    ${ACLRTC_SRC_FILES}
    ${TEST_ACLRTC_SRC_FILES}
)
add_dependencies(ascendc_utest_aclrtc gen_ut_kernel_tiling)
target_compile_definitions(ascendc_utest_aclrtc PRIVATE
    UT_TEST
)
target_compile_definitions(ascendc_utest_aclrtc PRIVATE ASCENDC_CPU_DEBUG=1)
target_include_directories(ascendc_utest_aclrtc PRIVATE
    ${ACLRTC_TEST_HEADERS_FILES}
    ${ASCENDC_SYS_DIR}/include
    ${ASCENDC_SYS_DIR}/include/acl/
)
target_compile_options(ascendc_utest_aclrtc PRIVATE
    -g
    -Werror
)
target_link_directories(ascendc_utest_aclrtc PRIVATE
    ${ASCEND_DEP_LIB_DIR}
)
target_link_libraries(ascendc_utest_aclrtc PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_basic>
    -Wl,--no-as-needed
    c_sec
    dl
    -Wl,--as-needed
)

run_llt_test(
    TARGET ascendc_utest_aclrtc
    TASK_NUM 1
)
