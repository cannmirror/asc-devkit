cmake_minimum_required(VERSION 3.14.1)

set(ASCEND_HOME_PATH ${ASCEND_CANN_PACKAGE_PATH})

set(ASCENDC_TOOLS_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/output)

set(CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/module
    ${CMAKE_MODULE_PATH}
)

include(cmake/config.cmake)
include(cmake/intf.cmake)
include(cmake/func.cmake)

add_subdirectory(ut)
# add_subdirectory(py_ut)