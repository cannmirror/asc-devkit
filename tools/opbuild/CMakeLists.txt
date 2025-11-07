#cmake_minimum_required(VERSION 3.5.1)
#project(opbuild)
#
#set(PKG_PATH /home/wk/code/c85/runtime/lib/runtime CACHE PATH "pkg path")
#set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/..)
#
#message(STATUS "install directory: ${CMAKE_INSTALL_PREFIX}")
#
add_executable(op_build op_build.cpp op_generator.cpp op_generator_factory.cpp op_cfg_generator.cpp
  op_proto_generator.cpp op_aclnn_generator.cpp op_custom_registry_generator.cpp op_aclnn_fallback_generator.cpp
  op_cpu_cfg_generator.cpp
)

#target_compile_options(op_build PRIVATE -g -D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(op_build PRIVATE -Wextra -Wfloat-equal -fno-common)

if(BUILD_OPEN_PROJECT)
  target_include_directories(op_build PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/base
    ${ASCEND_CANN_PACKAGE_PATH}/include/ge
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${CMAKE_CURRENT_SOURCE_DIR}/../build/common
  )
else()
  target_include_directories(op_build PRIVATE
    ${TOP_DIR}/abl/libc_sec/include
    ${TOP_DIR}/abl/mmpa/inc
    ${TOP_DIR}/abl/slog/inc/toolchain
    ${TOP_DIR}/asc/asc-devkit/tools/build/common
  )
endif()

#target_link_directories(op_build PRIVATE ${PKG_PATH}/lib64 ${CMAKE_INSTALL_PREFIX}/lib)
#target_link_libraries(op_build PRIVATE op_def graph graph_base register dl)
target_link_libraries(op_build PRIVATE
  $<BUILD_INTERFACE:intf_pub>
  $<BUILD_INTERFACE:mmpa_headers>
  metadef_headers
#   air_headers
  graph
  graph_base
  register
  dl
  alog
  mmpa
  error_manager
)

install(TARGETS op_build
  RUNTIME DESTINATION lib OPTIONAL
)

if(NOT BUILD_OPEN_PROJECT)
    install_package(
    PACKAGE op_build
    TARGETS op_build
    )
endif()