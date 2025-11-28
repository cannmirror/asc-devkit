# CMakeDetermineASCCompiler.cmake用来初始化ascc的变量，增量编译时不会再次触发该文件
# 该文件用来找到编译器并识别它是什么
# Update ASCEND_CANN_PACKAGE_PATH, SOC_VERSION, CMAKE_BUILD_TYPE, CMAKE_INSTALL_PREFIX
# 1. Setup ASCEND_CANN_PACKAGE_PATH based on env variable ASCEND_HOME_PATH
set(DEFAULT_ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest/")
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    message(WARNING "ASCEND_CANN_PACKAGE_PATH is not set. Set to env variable ASCEND_HOME_PATH.")
    if(NOT EXISTS $ENV{ASCEND_HOME_PATH})
        message(WARNING "Env variable ASCEND_HOME_PATH is not set. Set to default value ${DEFAULT_ASCEND_PATH}.")
        set(ASCEND_CANN_PACKAGE_PATH ${DEFAULT_ASCEND_PATH} CACHE PATH "Path for CANN package")
    else()
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH} CACHE PATH "Path for CANN package")
    endif()
else()
    set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_CANN_PACKAGE_PATH} CACHE PATH "Path for CANN package")
endif()

message(STATUS "System processer: ${CMAKE_SYSTEM_PROCESSOR}")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_CANN_PACKAGE_LINUX_PATH ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    set(ASCEND_CANN_PACKAGE_LINUX_PATH ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux)
else ()
    message(FATAL_ERROR "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

find_program(CMAKE_ASC_COMPILER NAMES "bisheng" PATHS "${ASCEND_CANN_PACKAGE_LINUX_PATH}/ccec_compiler/bin/" "$ENV{PATH}" "$ENV{ASCEND_HOME_PATH}" DOC "ASC Compiler")

mark_as_advanced(CMAKE_ASC_COMPILER)
message(STATUS "CMAKE_ASC_COMPILER: " ${CMAKE_ASC_COMPILER})

set(CMAKE_ASC_SOURCE_FILE_EXTENSIONS asc)    # .asc后缀名自动用ascc, .cpp后缀名不会自动用bisheng, 必须要手动指定
set(CMAKE_ASC_COMPILER_ENV_VAR "ASC")        # Language命名为ASC

# 2. Check SOC_VERSION (if exists) is valid soc version. Current only supports 910B.
#    Need to store value of CCE_AICORE_ARCH
set(ascend910b_list Ascend910B1 Ascend910B2 Ascend910B2C Ascend910B3 Ascend910B4 Ascend910B4-1 Ascend910_9391
                    Ascend910_9381 Ascend910_9372 Ascend910_9392 Ascend910_9382 Ascend910_9362)
set(ascend310p_list Ascend310P1 Ascend310P3 Ascend310P5 Ascend310P7
                    Ascend310P3Vir01 Ascend310P3Vir02 Ascend310P3Vir04 Ascend310P3Vir08)
set(ascend910_95_list Ascend910_957b Ascend910_950z Ascend910_958b Ascend910_958a
                      Ascend910_9599 Ascend910_957d Ascend910_9581 Ascend910_9589 Ascend910_957c)
if(DEFINED SOC_VERSION)
    if(NOT ((SOC_VERSION IN_LIST ascend910b_list) OR (SOC_VERSION IN_LIST ascend310p_list) OR (SOC_VERSION IN_LIST ascend910_95_list)))
        message(FATAL_ERROR "SOC_VERSION ${SOC_VERSION} is unsupported, support list is ${ascend910b_list} ${ascend310p_list} ${ascend910_95_list}")
    endif()
endif()

if(SOC_VERSION IN_LIST ascend910b_list)
    set(CCE_AICORE_ARCH "dav-2201" CACHE STRING "Value for --npu-arch")
elseif(SOC_VERSION IN_LIST ascend310p_list)
    set(CCE_AICORE_ARCH "dav-2002" CACHE STRING "Value for --npu-arch")
elseif(SOC_VERSION IN_LIST ascend910_95_list)
    set(CCE_AICORE_ARCH "dav-3101" CACHE STRING "Value for --npu-arch")
endif()

# 3. CMAKE_BUILD_TYPE only support Release / Debug
set(build_type_list Release Debug)
if(NOT CMAKE_BUILD_TYPE)
    message(WARNING "CMAKE_BUILD_TYPE is not set. Set to default value Release.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)" FORCE)
elseif(NOT CMAKE_BUILD_TYPE IN_LIST build_type_list)
    message(FATAL_ERROR "CMAKE_BUILD_TYPE only support Release / Debug, current value is ${CMAKE_BUILD_TYPE}")
else()
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Build type Release/Debug (default Release)" FORCE)
endif()

# 4. CMAKE_INSTALL_PREFIX: default is out in current directory
if(NOT CMAKE_INSTALL_PREFIX)
    message(WARNING "CMAKE_INSTALL_PREFIX is not set. Set to default value ${CMAKE_CURRENT_SOURCE_DIR}/out.")
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/out CACHE PATH "path for install" FORCE)
else()
    set(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE PATH "CMake output path" FORCE)
endif()

# 第一次编译时顺序： CMakeDetermineASCCompiler.cmake -> CMakeASCInformation.cmake
# 增量编译时顺序：   CMakeASCInformation.cmake
find_program(CMAKE_ASC_LLD_LINKER NAMES "ld.lld" PATHS "${ASCEND_CANN_PACKAGE_LINUX_PATH}/ccec_compiler/bin/" DOC "ASC ld.lld Linker" NO_DEFAULT_PATH)

if(DEFINED SOC_VERSION)
    message(STATUS "SOC_VERSION: " ${SOC_VERSION})
endif()
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
message(STATUS "ASCEND_CANN_PACKAGE_PATH: " ${ASCEND_CANN_PACKAGE_PATH})
message(STATUS "ASCEND_CANN_PACKAGE_LINUX_PATH: " ${ASCEND_CANN_PACKAGE_LINUX_PATH})
message(STATUS "CMAKE_ASC_LLD_LINKER: ${CMAKE_ASC_LLD_LINKER}")

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeASCCompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeASCCompiler.cmake
    @ONLY
)