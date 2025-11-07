# CMakeDetermineASCCompiler.cmake用来初始化ascc的变量，增量编译时不会再次触发该文件
find_program(CMAKE_ASC_COMPILER NAMES "bishengcc" PATHS "$ENV{PATH}" "$ENV{ASCEND_HOME_PATH}" DOC "ASC Compiler")
mark_as_advanced(CMAKE_ASC_COMPILER)

message(STATUS "CMAKE_ASC_COMPILER: " ${CMAKE_ASC_COMPILER})

set(CMAKE_ASC_SOURCE_FILE_EXTENSIONS asc)    # .asc后缀名自动用ascc, .cpp后缀名不会自动用ascc, 必须要手动指定
set(CMAKE_ASC_COMPILER_ENV_VAR "ASC")        # Language命名为ASC

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeASCCompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeASCCompiler.cmake
    @ONLY
)