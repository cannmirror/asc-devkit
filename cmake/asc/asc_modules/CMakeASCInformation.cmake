# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

include(CMakeCommonLanguageInclude)

# extension for the output of a compile for a single file
if(UNIX)
    set(CMAKE_ASC_OUTPUT_EXTENSION .o)
else()
    set(CMAKE_ASC_OUTPUT_EXTENSION .obj)
endif()

set(CMAKE_INCLUDE_FLAG_ASC "-I")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(debug_compile_options "-O0 -g")
endif()

set(CMAKE_DEPFILE_FLAGS_ASC "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")
if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER) AND CMAKE_GENERATOR MATCHES "Makefiles|WMake")
    # dependencies are computed by the compiler itself
    set(CMAKE_ASC_DEPFILE_FORMAT gcc)
    set(CMAKE_ASC_DEPENDS_USE_COMPILER TRUE)
endif()

# -shared to create .so for shared library
if(NOT DEFINED CMAKE_SHARED_LIBRARY_CREATE_ASC_FLAGS)
    set(CMAKE_SHARED_LIBRARY_CREATE_ASC_FLAGS ${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS})
endif()
# used for -Wl,-soname when creating shared library
if(NOT DEFINED CMAKE_SHARED_LIBRARY_SONAME_ASC_FLAG)
  set(CMAKE_SHARED_LIBRARY_SONAME_ASC_FLAG ${CMAKE_SHARED_LIBRARY_SONAME_C_FLAG})
endif()
# used for -Wl,-rpath when link executable has shared library
if(NOT DEFINED CMAKE_EXECUTABLE_RUNTIME_ASC_FLAG)
    set(CMAKE_EXECUTABLE_RUNTIME_ASC_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG})
endif()

# rule variable to compile a single .o file
# CMAKE_ASC_COMPILER: bisheng
if(NOT CMAKE_ASC_COMPILE_OBJECT)
    set(CMAKE_ASC_COMPILE_OBJECT "<CMAKE_ASC_COMPILER> <DEFINES> <INCLUDES> -fPIC ${debug_compile_options} \
<FLAGS> -o <OBJECT> -c -x asc <SOURCE>")
endif()

# Create a static archive incrementally for large object file counts.
if(NOT DEFINED CMAKE_ASC_ARCHIVE_CREATE)
    set(CMAKE_ASC_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
# add without checking duplication
if(NOT DEFINED CMAKE_ASC_ARCHIVE_APPEND)
    set(CMAKE_ASC_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_ASC_ARCHIVE_FINISH)
    set(CMAKE_ASC_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
endif()


# rule variable to create a shared module
if(NOT CMAKE_ASC_CREATE_SHARED_MODULE)
    set(CMAKE_ASC_CREATE_SHARED_MODULE ${CMAKE_ASC_CREATE_SHARED_LIBRARY})
endif()

# when language is set to ASC, execute when add_executable.
# FLAGS: -D
# ASC_LINK_FLAGS: link options
if(NOT CMAKE_ASC_LINK_EXECUTABLE)
    set(CMAKE_ASC_LINK_EXECUTABLE "<CMAKE_ASC_COMPILER> <FLAGS> <CMAKE_ASC_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o \
<TARGET> <LINK_LIBRARIES>")
endif()

# rule variable to create a shared library
if(NOT CMAKE_ASC_CREATE_SHARED_LIBRARY)
    set(CMAKE_ASC_CREATE_SHARED_LIBRARY "<CMAKE_ASC_COMPILER> <CMAKE_SHARED_LIBRARY_ASC_FLAGS> \
<LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_ASC_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> \
<OBJECTS> <LINK_LIBRARIES>")
endif()

set(CMAKE_ASC_INFORMATION_LOADED 1)   # 标记Cmake已经加载初始化ASC编程语言

if(CMAKE_ASC_RUN_MODE STREQUAL "sim")
    set(_ARCH_TO_DIR_MAP
        "dav-2201" "dav_2201"
        "dav-3101" "dav_3510"
        "dav-3510" "dav_3510"
    )
    list(FIND _ARCH_TO_DIR_MAP "${CMAKE_ASC_ARCHITECTURES}" _index)
    if(_index GREATER -1)
        math(EXPR _val_index "${_index} + 1")
        list(GET _ARCH_TO_DIR_MAP ${_val_index} _ASC_INTERNAL_DIR)
        set(_ASC_SIM_PATH "$ENV{ASCEND_HOME_PATH}/tools/simulator/${_ASC_INTERNAL_DIR}/lib")
        string(APPEND CMAKE_ASC_LINK_FLAGS " -Wl,-rpath,${_ASC_SIM_PATH} -Wl,-L${_ASC_SIM_PATH} -Wl,--disable-new-dtags")
        link_libraries(runtime_camodel npu_drv)
        message(STATUS "ASC Simulator enabled: ${_ASC_SIM_PATH}")
    else()
        message(FATAL_ERROR "Unsupported ASC architecture for simulator: ${CMAKE_ASC_ARCHITECTURES}")
    endif()
endif()