add_library(elf_tool_obj OBJECT ascendc_elf_tool.c)

target_compile_options(elf_tool_obj PRIVATE 
    -DCMAKE_C_COMPILER_LAUNCHER=${CCACHE_PROGRAM}
    -DCMAKE_Cxx_COMPILER_LAUNCHER=${CCACHE_PROGRAM}
    -g
  )

if(BUILD_OPEN_PROJECT)
  target_include_directories(elf_tool_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/build/elf_tool
  )
else()
  target_include_directories(elf_tool_obj PRIVATE
    ${TOP_DIR}/asc/asc-devkit/tools/build/elf_tool
  )
endif()

target_link_libraries(elf_tool_obj PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec_static
)