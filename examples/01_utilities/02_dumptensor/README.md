# DumpTensor直调样例说明

## 概述
本样例通过Ascend C编程语言实现了Add算子和Mmad算子，同时在算子中添加DumpTensor调测，使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，给出了对应的端到端实现。

## 支持的AI处理器
- Ascend 910C
- Ascend 910B

## 目录结构介绍
```
├── 02_dumptensor
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── cube.asc                // AscendC算子实现 & 调用样例
│   └── vector.asc              // AscendC算子实现 & 调用样例
```

## 算子描述
- 算子功能：  
DumpTensor介绍：  
  使用DumpTensor可以Dump指定Tensor的内容，同时支持打印自定义的附加信息。此外，DumpAccChkPoint可以支持指定偏移位置的Tensor打印。

  本样例将Dump的内容保存到输出文件中，并对比文件中是否有Dump的内容，从而判断样例是否执行成功。

  - Cube场景Mmad算子介绍：  

    算子使用基础API包括DataCopy、LoadData、Mmad等，实现矩阵乘功能。

    计算公式为：

    ```
    C = A * B + Bias
    ```

    - A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
    - C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。
    - Bias为矩阵乘偏置，形状为\[N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。

  - Vector场景Add算子介绍

    Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：  
    ```
    z = x + y
    ```
- 算子规格：  
- Cube场景Mmad算子：  

  在核函数直调样例中，算子实现支持的shape为：M = 32, N = 32, K = 32。
  <table>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">mmad_custom</td></tr>
  </table>

- Vector场景Add算子：  
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

- 算子实现：  
- Cube场景Mmad算子：

  本样例中实现的是[m, n, k]固定为[32, 32, 32]的Mmad算子，并使用Ascend C基础API实现，同时添加DumpTensor用于Dump指定Tensor的内容。

  - kernel实现  
    Mmad算子的数学表达式为：
    $$
    C = A * B
    $$
    其中A的形状为[32, 32], B的形状为[32, 32], C的形状为[32, 32]。

    调用实现

    使用内核调用符<<<>>>调用核函数。

- Vector场景Add算子： 

  本样例中实现的是固定shape为8*2048的Add算子，同时添加DumpTensor用于Dump指定Tensor的内容。
  - kernel实现  
    Add算子的数学表达式为：
    ```
    z = x + y
    ```
    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

    Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。

  - 调用实现

    使用内核调用符<<<>>>调用核函数。

## 编译运行：  
- 配置环境变量  
  以命令行方式下载样例代码，master分支为例。
  ```bash
  cd ${git_clone_path}/examples/01_utilities/02_dumptensor
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

- 样例执行
  ```bash
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake ..;make -j;             # 编译工程
  ```
  执行mmad_custom_cube.asc样例的命令如下所示：
  ```bash
  python3 ../scripts/gen_data_cube.py   # 生成测试输入数据
  ./cube               # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result_cube.py output_cube/output.bin output_cube/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass
  ```
  执行mmad_custom_vector.asc样例的命令如下所示：
  ```bash
  python3 ../scripts/gen_data_vector.py      # 生成测试输入数据
  ./vector                # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result_vector.py output_vector/output_z.bin output_vector/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass
  ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/18 | 样例目录调整，新增本readme |
