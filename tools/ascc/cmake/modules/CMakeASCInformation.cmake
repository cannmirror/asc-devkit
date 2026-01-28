include(CMakeCommonLanguageInclude)

# dict: key -> value
function(map_get_value map_name key out_var)
    list(FIND ${map_name} ${key} index)
    if(index EQUAL -1)
        set(${out_var} "KEY_NOT_FOUND" PARENT_SCOPE)
    else()
        math(EXPR value_index "${index} + 1")
        list(GET ${map_name} ${value_index} value)
        set(${out_var} ${value} PARENT_SCOPE)
    endif()
endfunction()

# Setup env variable: ASCEND_HOME_PATH
set(DEFAULT_ASCEND_PATH "/usr/local/Ascend/cann/")
if(NOT EXISTS $ENV{ASCEND_HOME_PATH})
    message(WARNING "Env variable ASCEND_HOME_PATH is not set. Set to default value ${DEFAULT_ASCEND_PATH}.")
    set(ASCEND_HOME_PATH ${DEFAULT_ASCEND_PATH})
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

# Check ASCEND_PRODUCT_TYPE is valid soc version. Current only supports 910B
set(ascend910b_list ascend910b1 ascend910b2 ascend910b2c ascend910b3 ascend910b4 ascend910b4-1 ascend910_9391
                    ascend910_9381 ascend910_9372 ascend910_9392 ascend910_9382 ascend910_9362)
set(SOC_MAP
    "ascend910b1" "Ascend910B1"         "ascend910b2" "Ascend910B2"          "ascend910b2c" "Ascend910B2C"
    "ascend910b3" "Ascend910B3"         "ascend910b4" "Ascend910B4"          "ascend910b4-1" "Ascend910B4-1"
    "ascend910_9391" "Ascend910_9391"   "ascend910_9381" "Ascend910_9381"    "ascend910_9372" "Ascend910_9372"
    "ascend910_9392" "Ascend910_9392"   "ascend910_9382" "Ascend910_9382"    "ascend910_9362" "Ascend910_9362"
)
if(NOT DEFINED ASCEND_PRODUCT_TYPE)
    message(FATAL_ERROR "ASCEND_PRODUCT_TYPE must be defined.")
endif()
string(TOLOWER "${ASCEND_PRODUCT_TYPE}" LOWER_SOC_VERSION)
if(NOT LOWER_SOC_VERSION IN_LIST ascend910b_list)
    message(FATAL_ERROR "ASCEND_PRODUCT_TYPE ${ASCEND_PRODUCT_TYPE} is unsupported, support list is ${ascend910b_list}")
endif()

# convert lower case soc to what ascc needed
map_get_value(SOC_MAP ${LOWER_SOC_VERSION} ASCEND_PRODUCT_TYPE)

# 第一次编译时顺序： CMakeDetermineASCCompiler.cmake -> CMakeASCInformation.cmake -> asc_config.cmake
# 增量编译时顺序：   CMakeASCInformation.cmake -> asc_config.cmake
message(STATUS "ASCEND_HOME_PATH: " ${ASCEND_HOME_PATH})
message(STATUS "ASCEND_PRODUCT_TYPE: " ${ASCEND_PRODUCT_TYPE})


set(CMAKE_COMPILE_AS_ASC_FLAG "-arch ${ASCEND_PRODUCT_TYPE}")   # common compile options used in ascc
set(CMAKE_INCLUDE_FLAG_ASC "-I")

# extension for the output of a compile for a single file
if(UNIX)
    set(CMAKE_ASC_OUTPUT_EXTENSION .o)
else()
    set(CMAKE_ASC_OUTPUT_EXTENSION .obj)
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

# rule variable to compile a single object file: 编译一个.o的命令
# CMAKE_ASC_COMPILER: ascc
# CMAKE_COMPILE_AS_ASC_FLAG:    -arch {soc_version}
if(NOT CMAKE_ASC_COMPILE_OBJECT)
    set(CMAKE_ASC_COMPILE_OBJECT
        "<CMAKE_ASC_COMPILER> <DEFINES> <INCLUDES> <FLAGS> ${CMAKE_COMPILE_AS_ASC_FLAG} -o <OBJECT> -c <SOURCE>")
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

# when language is set to ASC, execute when add_executable. Add default link libraries and path in default.
# FLAGS: -D
# ASC_LINK_FLAGS: link options
set(DEFAULT_LINK_LIBS "-lascendc_runtime -lruntime -lerror_manager -lprofapi -lunified_dlog -lmmpa -lascend_dump -lc_sec")
set(DEFAULT_LINK_PATH "-L${ASCEND_HOME_PATH}/lib64")
if(NOT CMAKE_ASC_LINK_EXECUTABLE)
    set(CMAKE_ASC_LINK_EXECUTABLE
        "<CMAKE_CXX_COMPILER> <FLAGS> <CMAKE_ASC_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> ${DEFAULT_LINK_PATH} ${DEFAULT_LINK_LIBS}")
endif()

# rule variable to create a shared library: 编译一个.so的命令
# CMAKE_CXX_COMPILER：gcc
# must link with libascendc_runtime.a for elf_tool.c.o, ascendc_runtime.cpp.o
if(NOT CMAKE_ASC_CREATE_SHARED_LIBRARY)
    set(CMAKE_ASC_CREATE_SHARED_LIBRARY
        "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_ASC_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_ASC_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES> ${DEFAULT_LINK_PATH} ${DEFAULT_LINK_LIBS}")
endif()

set(CMAKE_ASC_INFORMATION_LOADED 1)   # 标记Cmake已经加载初始化ASC编程语言
