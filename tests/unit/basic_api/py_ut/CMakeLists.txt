cmake_minimum_required(VERSION 3.16.0)
project(tikcpp_py_ut)
message(STATUS "tikcpp_python_ut compile start")

run_python_llt_test(
        TARGET tikcpp_utest_compile_trace_log
        SRC_FILES_DIR ${ASCENDC_DIR}/tools/scripts/compile_trace_log
        TEST_FILES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testcase/compile_trace_log
        EXPORT_PYTHONPATH ${CMAKE_CURRENT_SOURCE_DIR}/testcase/stub/
        PRE_TEST_COMMAND
        COVERAGERC_DIR
)