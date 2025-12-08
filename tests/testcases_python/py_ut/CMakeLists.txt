cmake_minimum_required(VERSION 3.16.0)
project(ascendc_py_ut)

message(STATUS "ascendc_python_ut compile start")
# enable highlevel api install for include path
set(INSTALL_PATH ${CMAKE_BINARY_DIR})
set(DST_PATH adapter_ut)

add_custom_target(generate_hidden_dat_file_ut ALL
        COMMAND code_hidden ${ASCENDC_DIR}/tests/testcases_python/py_ut/adapter/stub_kernels ${ASCENDC_DIR}/tests/testcases_python/py_ut/adapter/stub_kernels/ascendc_impl.dat
        DEPENDS code_hidden
)

add_custom_target(install_adapter_api_ut ALL
        COMMAND bash ${ASCENDC_DIR}/tests/unit/basic_api/common/ci/install_files.sh ${ASCENDC_DIR} ${INSTALL_PATH} ${DST_PATH}
)

include(${ASCENDC_DIR}/tests/unit/basic_api/cmake/func.cmake)

run_python_llt_test(
        TARGET ascendc_aclrt_launch_kernel_python_ut
        SRC_FILES_DIR ${ASCENDC_DIR}/cmake/
        TEST_FILES_DIR ${ASCENDC_DIR}/tests/testcases_python/py_ut/aclrt_launch_kernel/
        PRE_TEST_COMMAND
        COVERAGERC_DIR
)

run_python_llt_test(
        TARGET ascendc_adapter_python_ut
        SRC_FILES_DIR ${ASCENDC_DIR}/tools/build/asc_op_compile_base/adapter/
        TEST_FILES_DIR ${ASCENDC_DIR}/tests/testcases_python/py_ut/adapter/
        ENV_FILE ${ASCENDC_DIR}/tests/unit/basic_api/common/ci/set_python_env.sh
        ENV_PARAMS ${ASCENDC_DIR}\ py_ut \ 910
        PRE_TEST_COMMAND
        COVERAGERC_DIR
)

run_python_llt_test(
        TARGET asc_op_compiler_python_ut
        SRC_FILES_DIR ${ASCENDC_DIR}/tools/build/asc_op_compile_base/asc_op_compiler/
        TEST_FILES_DIR ${ASCENDC_DIR}/tests/testcases_python/py_ut/asc_op_compiler/
        ENV_FILE ${ASCENDC_DIR}/tests/unit/basic_api/common/ci/set_python_env.sh
        ENV_PARAMS ${ASCENDC_DIR}\ py_ut \ 910
        PRE_TEST_COMMAND
        COVERAGERC_DIR
)

add_dependencies(ascendc_adapter_python_ut
    register
    gen_kernel_tiling
    install_adapter_api_ut
    generate_hidden_dat_file_ut
)

add_dependencies(asc_op_compiler_python_ut
    register
    gen_kernel_tiling
    install_adapter_api_ut
    generate_hidden_dat_file_ut
    asc_platform_test
)
