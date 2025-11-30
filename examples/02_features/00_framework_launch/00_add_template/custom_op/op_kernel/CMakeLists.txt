npu_op_kernel_sources(ascendc_kernels
    OP_TYPE AddTemplateCustom
    KERNEL_DIR ./op_kernel/
    COMPUTE_UNIT Ascend910B1
    KERNEL_FILE add_template_custom.cpp
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)