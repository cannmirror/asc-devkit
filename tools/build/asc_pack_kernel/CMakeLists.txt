add_executable(ascendc_pack_kernel ascendc_pack_kernel.c $<TARGET_OBJECTS:elf_tool_obj>)

target_link_libraries(ascendc_pack_kernel PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec_static
)

target_compile_options(ascendc_pack_kernel PRIVATE 
    -DCMAKE_C_COMPILER_LAUNCHER=${CCACHE_PROGRAM}
    -DCMAKE_Cxx_COMPILER_LAUNCHER=${CCACHE_PROGRAM}
)

if(BUILD_OPEN_PROJECT)
  target_include_directories(ascendc_pack_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../elf_tool
  )
else()
  target_include_directories(ascendc_pack_kernel PRIVATE
    ${TOP_DIR}/asc/asc-devkit/tools/build/elf_tool
  )
endif()

install(TARGETS ascendc_pack_kernel
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL
)