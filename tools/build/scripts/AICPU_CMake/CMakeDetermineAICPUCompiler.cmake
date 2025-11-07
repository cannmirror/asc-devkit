find_program(CMAKE_AICPU_COMPILER NAMES "bisheng" PATHS "$ENV{ASCEND_HOME_PATH}/compiler/ccec_compiler/bin" DOC "AICPU Compiler")
mark_as_advanced(CMAKE_AICPU_COMPILER)

message(STATUS "Detecting AICPU compiler: " ${CMAKE_AICPU_COMPILER})

set(CMAKE_AICPU_SOURCE_FILE_EXTENSIONS aicpu)
set(CMAKE_AICPU_COMPILER_ENV_VAR "AICPU")

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeAICPUCompiler.cmake.in
	${CMAKE_PLATFORM_INFO_DIR}/CMakeAICPUCompiler.cmake
    @ONLY
)
