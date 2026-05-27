# Add算子快速入门

本入门示例基于Ascend C SIMT实现Add算子，带你快速上手实践，涵盖Device端核函数实现、Host端调用以及编译运行的完整流程，帮助开发者建立整体认知。

开始前请参考[环境准备](../../环境准备.md)安装所需的CANN软件包，完整样例请见[Add](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_simt_api/00_introduction/01_add)。

- **算子功能介绍**：

    Add算子的数学表达式为：

    ![](../../../figures/zh-cn_formulaimage_0000002541636777.png)

    计算逻辑：逐元素完成 `z = x + y`。

- **算子设计**

    - **Device端核函数编程接口**
        - 核函数定义：通过 [\_\_global\_\_](../../../编程指南/语言扩展层/SIMT-BuiltIn关键字.md) 修饰符声明。
        - 数据分块（Tiling）：使用内置关键字 [threadIdx、blockIdx、blockDim](../../../编程指南/语言扩展层/SIMT-BuiltIn关键字.md) 确定每个线程负责处理的数据。
        - 数据搬入：无需额外接口，直接通过指针访问即可。
        - 数据计算：无需额外接口，直接使用 `+` 运算符完成。
        - 数据搬出：无需额外接口，直接通过指针访问即可。
    - **Host端运行时接口**
        - 内存分配：使用 `aclrtMallocHost`分配Host Memory，`aclrtMalloc`分配Device Memory。
        - 数据搬入：使用 `aclrtMemcpy` 将输入数据从Host Memory拷贝到Device Memory。
        - 启动NPU计算任务：通过 `<<<...>>>`语法糖启动核函数。
        - 同步等待：调用 `aclrtSynchronizeStream`或 `aclrtSynchronizeDevice`等待任务完成。
        - 数据搬出：使用 `aclrtMemcpy`将计算结果从Device Memory拷贝回Host Memory。
    
      > [!NOTE] 说明
      > - 请参见[Ascend-C概述与学习路径](../../Ascend-C概述与学习路径.md)技术附录章节，获取`Ascend C API 参考`和`CANN运行时接口`链接，以查阅更多接口信息。

- **算子代码实现**：

    后缀名为`*.asc`的代码文件包含Host端与Device端代码。

    - **Device端Kernel实现**：

        Device端部分示例如下：
        ```cpp
        __global__ void add_custom(float* x, float* y, float* z, uint64_t total_length)
        {
            // Calculate global thread ID
            int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            // Maps to the row index of output tensor
            if (idx >= total_length) {
                return;
            }
            z[idx] = x[idx] + y[idx];
        }
        ```

    - **Host端代码实现**：

        Host端通过<<<>>>语法糖调用Device端代码。
        ```cpp
        int32_t main(int argc, char const *argv[])
        {
            ...
            aclrtCreateStream(&stream);
            // Allocate host and device memory, and copy input data from host to device
            aclrtMallocHost((void **)(&z_host), total_byte_size);
            aclrtMalloc((void **)&x_device, total_byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMalloc((void **)&y_device, total_byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMalloc((void **)&z_device, total_byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMemcpy(x_device, total_byte_size, x_host, total_byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(y_device, total_byte_size, y_host, total_byte_size, ACL_MEMCPY_HOST_TO_DEVICE);

            // Configure kernel launch parameters
            uint32_t blocks_per_grid      = 48; // Number of thread blocks (Grid size)
            uint32_t threads_per_block = 256; // Number of threads per block (Block size)
            uint32_t dyn_ubuf_size        = 0;  // No dynamic memory required in this sample

            // Launch kernel <<<grid_dim, block_dim, dynamic_memory_size, stream>>>
            add_custom<<<blocks_per_grid, threads_per_block, dyn_ubuf_size, stream>>>(x_device, y_device, z_device, x.size());

            // Wait for the add_custom kernel to complete
            aclrtSynchronizeStream(stream);

            // Copy the result from device memory to host memory
            aclrtMemcpy(z_host, total_byte_size, z_device, total_byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> output((float *)z_host, (float *)(z_host + total_byte_size)); 
            ...
        }
        ```

- **算子编译与运行**：

    CMake 配置文件示例：
    ```
    cmake_minimum_required(VERSION 3.16)
    # find_package(ASC)是CMake中用于查找和配置Ascend C编译工具链的命令
    find_package(ASC REQUIRED)
    # 指定项目支持的语言包括ASC和CXX，ASC表示支持使用毕昇编译器对Ascend C编程语言进行编译
    project(kernel_samples LANGUAGES ASC CXX)
    
    add_executable(demo
        add.asc
    )
    
    # 通过编译选项设置NPU架构
    target_compile_options(demo PRIVATE   
       $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510 --enable-simt>
    )
    ```
    编译并执行：
    ```
    mkdir -p build && cd build; 
    cmake ..;make -j;
    ./demo
    ```
    > [!NOTE] 说明
    > - 该样例支持以下型号：
    >     - Ascend 950PR / Ascend 950DT
    > - 编译选项 `--npu-arch` 用于指定NPU架构版本，`dav-` 后为架构版本号，请替换为您实际使用的版本。各 AI 处理器型号对应的架构版本号请通过[AI 处理器型号和 \_\_NPU\_ARCH\_\_ 的对应关系](../../../编程指南/语言扩展层/SIMD-BuiltIn关键字.md#table65291052154114) 查询。
    > - 编译选项`--enable-simt` 用于启用SIMT编程场景。

如需进一步了解Ascend C的SIMD与SIMT编程模型，请参阅[Ascend C 编程模型概述](../../../编程指南/编程模型/编程模型概述.md)。
