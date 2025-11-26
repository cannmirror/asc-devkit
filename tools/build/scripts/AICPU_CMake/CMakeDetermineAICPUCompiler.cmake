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

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SYSTEM_LOWER_PROCESSOR)
if(EXISTS $ENV{ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_LOWER_PROCESSOR}-linux/ccec_compiler/bin)
    set(AICPU_COMPILER_PATH $ENV{ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_LOWER_PROCESSOR}-linux/ccec_compiler/bin)
else()
    set(AICPU_COMPILER_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}/compiler/ccec_compiler/bin")
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
