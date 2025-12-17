# CMakeDetermineASCCompiler.cmake is used to initialize ASC-related variables.
# And this file will not be triggered again during incremental compilation.
# This file is used to locate the compiler.
# Update SOC_VERSION, CMAKE_BUILD_TYPE, CMAKE_INSTALL_PREFIX
# 1. Setup env variable ASCEND_HOME_PATH
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    set(POSSIBLE_PATHS "/usr/local/Ascend/cann" "${HOME}/Ascend/cann")

    message(FATAL_ERROR "
    ================================================================================
    ERROR: ASCEND_HOME_PATH environment variable is not set!

    This variable is required to find CANN package.

    Possible solutions:
        Source the environment setup script: source <ascend_install_path>/set_env.sh

    Common installation locations:
    ${POSSIBLE_PATHS}
    ================================================================================
    ")
else()
    if(NOT EXISTS "$ENV{ASCEND_HOME_PATH}")
        message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH directory does not exist!")
    endif()
endif()

message(STATUS "System processer: ${CMAKE_SYSTEM_PROCESSOR}")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_CANN_PACKAGE_LINUX_PATH $ENV{ASCEND_HOME_PATH}/x86_64-linux)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    set(ASCEND_CANN_PACKAGE_LINUX_PATH $ENV{ASCEND_HOME_PATH}/aarch64-linux)
else ()
    message(FATAL_ERROR "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

find_program(CMAKE_ASC_COMPILER NAMES "bisheng" PATHS "${ASCEND_CANN_PACKAGE_LINUX_PATH}/ccec_compiler/bin/" "$ENV{PATH}" "$ENV{ASCEND_HOME_PATH}" DOC "ASC Compiler")

mark_as_advanced(CMAKE_ASC_COMPILER)
message(STATUS "CMAKE_ASC_COMPILER: " ${CMAKE_ASC_COMPILER})

set(CMAKE_ASC_SOURCE_FILE_EXTENSIONS asc)    # Specify .asc as the supported file extension
set(CMAKE_ASC_COMPILER_ENV_VAR "ASC")        # Name the language ASC

# Sequence for the first compilation: CMakeDetermineASCCompiler.cmake -> CMakeASCInformation.cmake
# Incremental compilation: CMakeASCInformation.cmake
find_program(CMAKE_ASC_LLD_LINKER NAMES "ld.lld" PATHS "${ASCEND_CANN_PACKAGE_LINUX_PATH}/ccec_compiler/bin/" DOC "ASC ld.lld Linker" NO_DEFAULT_PATH)

message(STATUS "ASCEND_CANN_PACKAGE_LINUX_PATH: " ${ASCEND_CANN_PACKAGE_LINUX_PATH})
message(STATUS "CMAKE_ASC_LLD_LINKER: ${CMAKE_ASC_LLD_LINKER}")

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeASCCompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeASCCompiler.cmake
    @ONLY
)