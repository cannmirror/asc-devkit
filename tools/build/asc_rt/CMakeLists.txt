add_library(ascend_runtime_base OBJECT ascendc_runtime.cpp aicpu_rt.cpp)

if(BUILD_OPEN_PROJECT)
  target_include_directories(ascend_runtime_base PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${ASCEND_CANN_PACKAGE_PATH}/include/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/include/ge
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/acl
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
  )
else()
  target_include_directories(ascend_runtime_base PRIVATE
    ${TOP_DIR}/ace/npuruntime/inc/runtime/
    ${TOP_DIR}/ace/npuruntime/inc/
    ${TOP_DIR}/ace/npuruntime/acl/inc/external/
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/metadef/inc/external/
    ${TOP_DIR}/abl/msprof/inc
    ${TOP_DIR}/abl/libc_sec/include
    ${TOP_DIR}/abl/mmpa/inc
    ${TOP_DIR}/abl/slog/inc/toolchain
    ${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tools/build/common
    ${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/include
    ${TOP_DIR}/runtime/include/external/acl
    ${TOP_DIR}/runtime/include/external
  )
endif()

target_compile_options(ascend_runtime_base PRIVATE
  -fvisibility-inlines-hidden
  -fvisibility=hidden
)

target_link_libraries(ascend_runtime_base PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    mmpa
    alog
    ascendcl
)

add_library(ascendc_runtime STATIC
   $<TARGET_OBJECTS:ascend_runtime_base>
   $<TARGET_OBJECTS:elf_tool_obj>
)

install(TARGETS ascendc_runtime
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL
)
