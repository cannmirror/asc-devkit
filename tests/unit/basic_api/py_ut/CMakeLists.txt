cmake_minimum_required(VERSION 3.14.1)
project(tikcpp_py_ut)
message(STATUS "tikcpp_python_ut compile start")

run_python_llt_test(
        TARGET tikcpp_utest_msobjdump
        SRC_FILES_DIR ${ASCENDC_TOOLS_ROOT_DIR}/msobjdump/msobjdump/
        TEST_FILES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testcase/msobjdump/
        EXPORT_PYTHONPATH ${CMAKE_CURRENT_SOURCE_DIR}/testcase/stub/
        PRE_TEST_COMMAND
        COVERAGERC_DIR
)

run_python_llt_test(
        TARGET tikcpp_utest_show_kernel_debug_data
        SRC_FILES_DIR ${ASCENDC_TOOLS_ROOT_DIR}/show_kernel_debug_data/show_kernel_debug_data
        TEST_FILES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testcase/show_kernel_debug_data
        EXPORT_PYTHONPATH ${CMAKE_CURRENT_SOURCE_DIR}/testcase/stub/
        PRE_TEST_COMMAND
        COVERAGERC_DIR
)

run_python_llt_test(
        TARGET tikcpp_utest_compile_trace_log
        SRC_FILES_DIR ${ASCENDC_TOOLS_ROOT_DIR}/tools/scripts/compile_trace_log
        TEST_FILES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testcase/compile_trace_log
        EXPORT_PYTHONPATH ${CMAKE_CURRENT_SOURCE_DIR}/testcase/stub/
        PRE_TEST_COMMAND
        COVERAGERC_DIR
)