include(CMakeCommonLanguageInclude)

# extension for the output of a compile for a single file
if(UNIX)
    set(CMAKE_ASC_OUTPUT_EXTENSION .o)
else()
    set(CMAKE_ASC_OUTPUT_EXTENSION .obj)
endif()

set(CMAKE_INCLUDE_FLAG_ASC "-I")
# common compile options used for bisheng plugin
if(DEFINED SOC_VERSION)
    set(CMAKE_COMPILE_AS_ASC_FLAG "--npu-arch=${CCE_AICORE_ARCH} --npu-soc=${SOC_VERSION}")
endif()

# CMAKE_ASC_COMPILE_OBJECT not support pass list as argument, need to convert to string
# variable in CMAKE_ASC_COMPILE_OBJECT does not support generator expressions
# refer to host_intf.cmake and bisheng_intf.cmake
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(host_optimize_level "-O0")
else()
    set(host_optimize_level "-O2")
endif()
set(kernel_compile_options_list -O3 -std=c++17)
set(host_compile_options_list -fPIC ${host_optimize_level})
list(JOIN kernel_compile_options_list " " KERNEL_OPTIONS_LIST)
list(JOIN host_compile_options_list " " HOST_OPTIONS_LIST)

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
    set(CMAKE_ASC_COMPILE_OBJECT "<CMAKE_ASC_COMPILER> ${CMAKE_COMPILE_AS_ASC_FLAG} <DEFINES> <INCLUDES> -Xaicore-start \
${KERNEL_OPTIONS_LIST} -Xaicore-end -Xhost-start ${HOST_OPTIONS_LIST} -Xhost-end <FLAGS> -o <OBJECT> -c -x asc <SOURCE>")
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
