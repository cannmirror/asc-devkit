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

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SYSTEM_LOWER_PROCESSOR)
if(EXISTS $ENV{ASCEND_HOME_PATH}/${SYSTEM_LOWER_PROCESSOR}-linux/ccec_compiler/bin)
    set(AICPU_COMPILER_PATH $ENV{ASCEND_HOME_PATH}/${SYSTEM_LOWER_PROCESSOR}-linux/ccec_compiler/bin)
else()
    set(AICPU_COMPILER_PATH "$ENV{ASCEND_HOME_PATH}/compiler/ccec_compiler/bin")
endif()
find_program(CMAKE_AICPU_COMPILER 
    NAMES "bisheng" 
    PATHS "${AICPU_COMPILER_PATH}" 
    DOC "AICPU Compiler"
)

mark_as_advanced(CMAKE_AICPU_COMPILER)

message(STATUS "Detecting AICPU compiler: " ${CMAKE_AICPU_COMPILER})

set(CMAKE_AICPU_SOURCE_FILE_EXTENSIONS aicpu)
set(CMAKE_AICPU_COMPILER_ENV_VAR "AICPU")

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeAICPUCompiler.cmake.in
	${CMAKE_PLATFORM_INFO_DIR}/CMakeAICPUCompiler.cmake
    @ONLY
)
