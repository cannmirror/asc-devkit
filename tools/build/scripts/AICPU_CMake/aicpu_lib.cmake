
set(__CUR_LIST_DIR ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_MODULE_PATH ${__CUR_LIST_DIR})

enable_language(AICPU)


# public targets protected for multiple include
if (NOT TARGET aicpu_host_intf)
# host public targets
add_library(aicpu_host_base_intf INTERFACE)
target_include_directories(aicpu_host_base_intf INTERFACE
    $ENV{ASCEND_HOME_PATH}/include ${__CUR_LIST_DIR}/..
    ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/aicpu_api)
target_link_directories(aicpu_host_base_intf INTERFACE $ENV{ASCEND_HOME_PATH}/lib64)
target_link_libraries(aicpu_host_base_intf INTERFACE ascendc_runtime profapi ascendalog runtime ascendcl c_sec mmpa error_manager)
target_compile_options(aicpu_host_base_intf INTERFACE
    -fPIC
    -O2
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-ftrapv -fstack-check>
    $<$<COMPILE_LANGUAGE:C>:-pthread -Wfloat-equal -Wshadow -Wformat=2 -Wno-deprecated -Wextra>
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
)
target_compile_definitions(aicpu_host_base_intf INTERFACE
    _GLIBCXX_USE_CXX11_ABI=0
    _FORTIFY_SOURCE=2
)

add_library(aicpu_host_intf INTERFACE)
target_link_libraries(aicpu_host_intf INTERFACE aicpu_host_base_intf)


# device public targets
add_library(aicpu_device_base_intf INTERFACE)
target_compile_options(aicpu_device_base_intf INTERFACE -O2 -fPIC -std=c++17 -fvisibility=hidden -fvisibility-inlines-hidden)
target_compile_definitions(aicpu_device_base_intf INTERFACE __AICPU_DEVICE__ _GLIBCXX_USE_CXX11_ABI=0 _FORTIFY_SOURCE=2)
target_include_directories(aicpu_device_base_intf INTERFACE
    ${__CUR_LIST_DIR}/..
    ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/aicpu_api)


add_library(aicpu_device_intf INTERFACE)
target_link_directories(aicpu_device_intf INTERFACE ${ASCEND_CANN_PACKAGE_PATH}/lib64/device/lib64)
target_link_libraries(aicpu_device_intf INTERFACE aicpu_device_base_intf aicpu_api)
endif()

function(aicpu_library target)
    file(GLOB src ${ARGN})
    set(_AICPU_GEN_STUB_SRC ${CMAKE_BINARY_DIR}/${target}_aicpu_stub.cpp)
    # CMake set source language for targets in same directory was not allowed
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${target}_aicpu)
    file(WRITE ${CMAKE_BINARY_DIR}/${target}_aicpu/CMakeLists.txt
         "add_library(${target}_aicpu SHARED ${src})              \n"
         "add_custom_command(TARGET ${target}_aicpu POST_BUILD    \n"
         "    COMMAND python3 ${__CUR_LIST_DIR}/gen_aicpu_stub.py \n"
         "            $<TARGET_FILE:${target}_aicpu>              \n"
         "            ${_AICPU_GEN_STUB_SRC}                      \n"
         ")"
    )
    add_subdirectory(${CMAKE_BINARY_DIR}/${target}_aicpu)
    set_target_properties(${target}_aicpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${target}_aicpu)
    set_source_files_properties(${src} TARGET_DIRECTORY ${target}_aicpu PROPERTIES LANGUAGE AICPU)
    target_link_libraries(${target}_aicpu PRIVATE aicpu_device_intf)
    add_custom_command(OUTPUT ${_AICPU_GEN_STUB_SRC}
        DEPENDS ${target}_aicpu
    )
    set_source_files_properties(${_AICPU_GEN_STUB_SRC} PROPERTIES GENERATED TRUE)
    add_library(${target} OBJECT ${_AICPU_GEN_STUB_SRC})
    target_link_libraries(${target} PUBLIC aicpu_host_intf)
    target_include_directories(${target} PUBLIC
        ${__CUR_LIST_DIR}/..
        ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/aicpu_api
    )
    add_dependencies(${target} ${target}_aicpu)
    set_target_properties(${target} PROPERTIES AICPU_TARGET TRUE)
    unset(_AICPU_GEN_STUB_SRC)
endfunction()

function(target_is_aicpu is_aicpu target)
    get_target_property(TARGET_TYPE ${target} TYPE)
    set(IS_AICPU_TARGET FALSE)
    if ("${TARGET_TYPE}" STREQUAL "OBJECT_LIBRARY")
        get_target_property(IS_AICPU_TARGET ${target} AICPU_TARGET)
    endif()
    set(${is_aicpu} ${IS_AICPU_TARGET} PARENT_SCOPE)
endfunction()

function(target_compile_options)
    target_is_aicpu(IS_AICPU_TARGET ${ARGV0})
    if (IS_AICPU_TARGET)
        list(SUBLIST ARGN 1 -1 remain_opts)
        _target_compile_options(${ARGV0}_aicpu ${remain_opts})
    else()
        _target_compile_options(${ARGN})
    endif()
endfunction()

function(target_compile_definitions)
    target_is_aicpu(IS_AICPU_TARGET ${ARGV0})
    if (IS_AICPU_TARGET)
        list(SUBLIST ARGN 1 -1 remain_opts)
        _target_compile_definitions(${ARGV0}_aicpu ${remain_opts})
    else()
        _target_compile_definitions(${ARGN})
    endif()
endfunction()

function(target_include_directories)
    target_is_aicpu(IS_AICPU_TARGET ${ARGV0})
    if (IS_AICPU_TARGET)
        list(SUBLIST ARGN 1 -1 remain_opts)
        _target_include_directories(${ARGV0}_aicpu ${remain_opts})
    else()
        _target_include_directories(${ARGN})
    endif()
endfunction()

function(target_link_libraries)
    target_is_aicpu(IS_AICPU_TARGET ${ARGV0})
    if (IS_AICPU_TARGET)
        list(SUBLIST ARGN 1 -1 remain_opts)
        _target_link_libraries(${ARGV0}_aicpu ${remain_opts})
    else()
        _target_link_libraries(${ARGN})
    endif()
endfunction()

function(target_link_options)
    target_is_aicpu(IS_AICPU_TARGET ${ARGV0})
    if (IS_AICPU_TARGET)
        list(SUBLIST ARGN 1 -1 remain_opts)
        _target_link_options(${ARGV0}_aicpu ${remain_opts})
    else()
        _target_link_options(${ARGN})
    endif()
endfunction()

function(target_link_directories)
    target_is_aicpu(IS_AICPU_TARGET ${ARGV0})
    if (IS_AICPU_TARGET)
        list(SUBLIST ARGN 1 -1 remain_opts)
        _target_link_directories(${ARGV0}_aicpu ${remain_opts})
    else()
        _target_link_directories(${ARGN})
    endif()
endfunction()
