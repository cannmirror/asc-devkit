# Add算子直调样例
## 概述
本样例以Add算子为样例，展示了一种更为简单的算子编译流程，支持main函数和Kernel函数在同一个cpp文件中实现。

## 支持的AI处理器
- Ascend 910C
- Ascend 910B
## 目录结构介绍
```
├── 01_add
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── add.asc                 // AscendC算子实现，tpipe管理内存 & 调用样例
│   └── basic_api_tque_add.asc      // AscendC算子实现,tque管理内存 & 调用样例
│   └── basic_api_memory_allocator_add.asc      // AscendC算子实现,使用LocalMemAllocator简化代码 & 调用样例
```

## 算子描述
- 算子功能：  

  Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：  
  ```
  z = x + y
  ```
- 算子规格：
  - basic_api_tque_add  
    <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
    </tr>
    <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
    <tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
    </table>
  - basic_api_memory_allocator_add  
    <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
    </tr>
    <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">x</td><td align="center">72 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
    <tr><td align="center">y</td><td align="center">72 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">72 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
    </table>

- 算子实现：
  - kernel实现  
    Add算子的数学表达式为：
    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

    Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。

    - basic_api_tque_add  
      使用tque管理内存，使用静态Tensor编程方法进行Add算子的编程。
    - basic_api_memory_allocator_add  
      使用LocalMemAllocator进行线性内存分配并简化代码，使用double buffer进行流水排布优化性能。
  - tiling实现  
    TilingData参数设计，TilingData参数本质上是和并行数据切分相关的参数，其中basic_api_tque_add算子使用了2个tiling参数，basic_api_memory_allocator_add算子使用了1个tiling参数.
    - basic_api_tque_add  
      使用的tiling参数为totalLength、tileNum。totalLength是指需要计算的数据量大小，tileNum是指每个核上总计算数据分块个数。比如，totalLength这个参数传递到kernel侧后，可以通过除以参与计算的核数，得到每个核上的计算量，这样就完成了多核数据的切分。
    - basic_api_memory_allocator_add  
      使用的tiling参数为singleCoreLength。singleCoreLength是指每个核上需要计算的数据量大小。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行
在本样例根目录下执行如下步骤，编译并执行算子。
  - 配置环境变量  
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/examples/00_introduction/01_add
    ```
    请根据当前环境上CANN开发套件包的[安装方式](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，选择对应配置环境变量的命令。
    - 默认路径，root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
      ```
    - 默认路径，非root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
      ```
    - 指定路径install_path，安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
      ```
    配置安装路径后，执行以下命令统一配置环境变量。
    ```bash
    # 配置CANN环境变量
    source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
    # 添加AscendC CMake Module搜索路径至环境变量
    export CMAKE_PREFIX_PATH=${ASCEND_INSTALL_PATH}/compiler/tikcpp/ascendc_kernel_cmake:$CMAKE_PREFIX_PATH
    ```

  - 样例执行
    ```bash
    mkdir -p build && cd build;   # 创建并进入build目录
    cmake ..;make -j;             # 编译工程
    ```

    执行add.asc、basic_api_tque_add.asc样例的命令如下所示：
    ```bash
    # 在build目录执行以下内容
    ./add_basic_api_tque                        # 执行样例
    ./add_simt                        # 执行样例
    ```
    执行结果如下，说明精度对比成功。
    ```bash
    [Success] Case accurary is verification passed.
    ```

    执行basic_api_memory_allocator_add.asc样例的命令如下所示：
    ```bash
    # 在样例根目录执行以下内容
    python3 scripts/gen_data.py   # 生成测试输入数据
    ./build/add_basic_api_memory_allocator                        # 执行编译生成的可执行程序，执行样例
    python3 scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
    ```
    执行结果如下，说明精度对比成功。
    ```bash
    test pass
    ```


## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/06 | 样例目录调整，新增本readme |