cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(martrix_transpose_asc LANGUAGES ASC CXX)

set(SCENARIO_NUM "0" CACHE STRING "Case type to run (0-1)")

add_executable(demo
    matrix_transpose.asc
)

target_compile_definitions(demo PRIVATE
    SCENARIO_NUM=${SCENARIO_NUM}
)

target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
)

target_link_libraries(demo PRIVATE
    m
)
