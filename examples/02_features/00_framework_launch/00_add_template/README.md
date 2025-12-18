# Add算子tiling模板编程样例
## 概述
本样例以Add算子为示例，展示了Tiling模板编程。本样例使用自定义算子工程，编译并部署自定义算子包到自定义算子库中，并调用执行自定义算子。

## 支持的AI处理器
- Ascend 910B
## 目录结构介绍
```
└── 00_add_template
    ├── custom_op           // 算子编译工程目录
    │   ├── op_host             // host侧编译实现文件目录
    │   │   ├── add_template_custom.cpp           // kernel侧算子实现文件
    │   │   ├── add_template_custom_tiling.h      // tiling定义头文件
    │   │   └── tiling_key_add_template_custom.h  // 模板参数定义头文件
    │   │   └── CMakeLists.txt      // host侧CMake文件
    │   ├── op_kernel           // kernel侧编译实现文件目录
    │   │   └── add_template_custom.cpp           // host侧tiling定义
    │   │   └── CMakeLists.txt      // kernel侧CMake文件
    |   ├── build.sh            // 算子构建脚本
    |   ├── CMakeLists.txt      // 算子编译CMake文件
    |   ├── CMakePresets.json   // 算子编译配置文件
    |   └── run.sh              // 算子编译脚本
    ├── op_verify           // 算子执行工程目录
    │   ├── inc                 // 算子执行头文件目录
    │   │   ├── common.h            // 声明公共方法类，用于读取二进制文件
    │   │   ├── op_runner.h         // 算子描述声明文件，包含算子输入/输出，算子类型以及输入描述与输出描述
    │   │   └── operator_desc.h     // 算子运行相关信息声明文件，包含算子输入/输出个数，输入/输出大小等
    │   ├── run_out             // 算子执行时的输出目录
    │   ├── scripts
    │   │   ├── acl.json            // acl配置文件
    │   │   ├── add_template_custom.json  // 算子的原型定义json文件
    │   │   ├── verify_result.py    // 真值对比文件
    │   │   └── gen_data.py         // 输入数据和真值数据生成脚本文件
    │   ├── src                 // 算子执行工程实现目录
    │   │   ├── CMakeLists.txt      // 编译规则文件
    │   │   ├── common.cpp          // 公共函数，读取二进制文件函数的实现文件
    │   │   ├── main.cpp            // 单算子调用应用的入口
    │   │   ├── op_runner.cpp       // 单算子调用主体流程实现文件
    │   │   └── operator_desc.cpp   // 构造算子的输入与输出描述
    │   └── run.sh              // 算子执行工程脚本
    └── run.sh              // 算子一键编译执行脚本
```

## 算子描述
- 算子功能：  
Add算子实现了两个数据相加，返回相加结果的功能。本样例算子添加的模板参数包括输入的数据类型、shape等，根据模板参数，简化或统一算子的实现逻辑，开发者可以在模板参数中定义需要的信息，如输入输出的数据类型，其他扩展参数等。对应的数学表达式为：
  ```
  z = x + y
  ```
- 算子规格：
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
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_template_custom</td></tr>
  <tr><td rowspan="6" align="center">模板参数</td><td colspan="4" align="center">template&lt;typename D_T_X, typename D_T_Y, typename D_T_Z, int TILE_NUM, int IS_SPLIT&gt;</td>
      <tr><td>D_T_X</td><td colspan="1">typename</td><td colspan="2">数据类型(float16，float)</td></tr>
      <tr><td>D_T_Y</td><td colspan="1">typename</td><td colspan="2">数据类型(float16，float)</td></tr>
      <tr><td>D_T_Z</td><td colspan="1">typename</td><td colspan="2">数据类型(float16，float)</td></tr>
      <tr><td>TILE_NUM</td><td colspan="1">int</td><td colspan="2">切分数量</td></tr>
      <tr><td>IS_SPLIT</td><td colspan="1">int</td><td colspan="2">是否切分</td></tr>
  </tr>
  </table>

- 算子实现：
  - kernel实现  
    Add算子的数学表达式为：
    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

    Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。
  - tiling实现  
    分为Tiling模板设计以及TilingData参数设计：

    Tiling模板设计，本示例使用了5个模板参数，D_T_X、D_T_Y、D_T_Z分别是指输入x、输入y、输出z的数据类型，TILE_NUM是指每个核上总计算数据分块个数，IS_SPLIT是是否使能数据分块计算，IS_SPLIT为0时TILE_NUM无效。通过模板参数组合替代传统的TilingKey。

    TilingData参数设计，本示例算子使用了1个tiling参数，totalLength是指所有核需要计算的数据量总大小。

  - 调用实现  
    使用AscendC自定义算子工程编译并部署算子run包，再编译acl可执行文件并运行。

## 编译运行
在本样例根目录下执行如下步骤，编译并执行算子。
  - 配置环境变量  
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/examples/02_features/00_framework_launch/00_add_template
    ```
    请根据当前环境上CANN开发套件包的[安装方式](../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
    - 默认路径，root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=/usr/local/Ascend/cann
      ```
    - 默认路径，非root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=$HOME/Ascend/cann
      ```
    - 指定路径install_path，安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=${install_path}/cann
      ```
    配置安装路径后，执行以下命令统一配置环境变量。
    ```bash
    # 配置CANN环境变量
    source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
    ```
  - 样例执行
    ```bash
    bash run.sh -v float16
    ```
    -v 可选择tiling模板，可选参数有float16和float
    执行结果如下，说明精度对比成功。
    ```bash
    test pass
    ```


## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/20 | 样例目录调整，新增本readme |