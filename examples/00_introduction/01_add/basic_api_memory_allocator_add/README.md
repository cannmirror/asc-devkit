# Add算子直调样例
## 概述
本样例介绍基于静态Tensor方式编程的场景下Add算子的实现方法，支持main函数和Kernel函数在同一个cpp文件中实现，并提供<<<>>>直调方法。

## 支持的AI处理器
- Ascend 910C
- Ascend 910B
## 目录结构介绍
```
├── basic_api_memory_allocator_add
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── add.asc      // AscendC算子实现,使用LocalMemAllocator简化代码 & 调用样例
```

## 算子描述
- 算子功能：  

  算子实现的是固定shape为72×4096的Add算子。

  Add的计算公式为：

  ```python
  z = x + y
  ```

  - x：输入，形状为\[72, 4096]，数据类型为float；
  - y：输入，形状为\[72, 4096]，数据类型为float；
  - z：输出，形状为\[72, 4096]，数据类型为float；

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">72 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">72 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">72 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom_v1 / add_custom_v2 / add_custom_v3 / add_custom_v4</td></tr>
  </table>

- 算子实现：
  本样例中实现的是固定shape为72*4096的Add算子。
  
  - kernel实现
  
    Add算子的数学表达式为：
  
    ```
    z = x + y
    ```
  
    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。
  
    Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。
  
    优化add_custom_v2中反向同步，替换为MTE2等待MTE3执行结束。减少分支判断的同时，算子性能因为double buffer的原因不受影响。另外使用LocalMemAllocator进行线性内存分配，Bank冲突不敏感场景可以使用这种方式简化分配。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行

- 配置环境变量  
  以命令行方式下载样例代码，master分支为例。
  ```bash
  cd ${git_clone_path}/examples/00_introduction/01_add/basic_api_memory_allocator_add
  ```
  请根据当前环境上CANN开发套件包的[安装方式](../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=/usr/local/Ascend/latest
    ```
  - 默认路径，非root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=$HOME/Ascend/latest
    ```
  - 指定路径install_path，安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=${install_path}/latest
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
  # 在样例根目录执行以下内容
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./add                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass
  ```


## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/06 | 样例目录调整，新增本readme |