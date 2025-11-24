# 兼容Scatter算子直调样例
## 概述
本样例介绍兼容Scatter算子实现及核函数直调方法，Ascend910B不支持Scatter的能力，使用标量搬出的方式进行兼容满足Scatter要求。
使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，给出了对应的端到端实现。

## 支持的AI处理器
- Ascend 910B

## 目录结构介绍
```
├── ScatterCustomSample
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   └── scatter_custom.asc         // AscendC算子实现 & 调用样例
```

## 算子描述
- 算子功能：  
本调用样例中实现的是对于Scatter功能变换的兼容,Ascend910B不支持Scatter指令，使用标量搬出的方式进行兼容。

  Scatter计算逻辑是：给定一个连续的输入张量和一个目的地址偏移张量，Scatter指令根据偏移地址生成新的结果张量后将输入张量分散到结果张量中。

  兼容Scatter算子逻辑是：对于部分有规律的离散计算，可以通过Loop循环搬出的方式来提升效率，对于完全离散的场景,只能通过标量搬出的方式进行处理。
- 算子规格：  

<table>  
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">Scatter</th></tr>  
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>  
<tr><td align="center">x</td><td align="center">-</td><td align="center">float16</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">y</td><td align="center">-</td><td align="center">uint32</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">-</td><td align="center">float16</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">attr属性</td><td align="center">value</td><td align="center">\</td><td align="center">float16</td><td align="center">\</td><td align="center">1.0</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">scatter_custom</td></tr>  
</table>

- 算子实现：  
样例中实现的是对于Scatter功能变换的兼容。
- kernel实现   

  兼容Scatter算子逻辑是：对于部分有规律的离散计算，可以通过Loop循环搬出的方式来提升效率，对于完全离散的场景,只能通过标量搬出的方式进行处理。   

  兼容Scatter算子的实现流程分为3个基本任务：CopyIn任务负责将Global Memory上的输入Tensor srcGm和dstGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal进行标量计算，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor dstGm中。

- 调用实现  
  使用内核调用符<<<>>>调用核函数。

## 编译运行：  
  - 配置环境变量  
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/examples/03_libraries/ScatterCustomSample
    ```
    请根据当前环境上CANN开发套件包的[安装方式](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，选择对应配置环境变量的命令。
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
    python3 ../scripts/gen_data.py   # 生成测试输入数据
    ./demo                        # 执行编译生成的可执行程序，执行样例
    python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
    ```
    执行结果如下，说明精度对比成功。
    ```bash
    test pass
    ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/18 | 样例目录调整，新增本readme |