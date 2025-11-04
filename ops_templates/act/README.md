# ACT

## 简介

ACT（Ascend C Templates），是一套基于Ascend C开发的算子模板库，致力于为昇腾硬件上的Cube类融合算子提供高性能、可定制的开发方式。

ACT将复杂的矩阵乘算子分解为不同层级的可重用实现，使开发者能够通过组装不同组件来实现自定义算子。ACT不仅提供了丰富的可复用基础组件，还支持用户自定义扩展，从而降低编程门槛，同时更易于在硬件上实现高性能。

ACT特点：

- 模块化分层设计：架构简单、组合性强、模块灵活拼接；
- 高可扩展性：接口可独立替换、复用，通过tag策略支持自定义扩展；
- 良好可调试性：支持Tiling自动推导与静态编译检查，调优点明确易用；
- NPU深度适配：贴合硬件特性，最大化发挥硬件性能。

## 目录结构说明

ACT代码目录结构如下：

```bash
├── docs                                # ACT文档
│   ├── 01_quickstart.md                # 快速入门
│   ├── 02_programming_guidelines.md    # 编程指南
│   ├── 03_layout.md                    # Layout介绍
│   ├── 04_design.md                    # ACT分层结构
│   └── figures                         # 文档图片
├── examples                            # 基于ACT的算子样例
│   ├── 00_basic_matmul
│   │   ├── CMakeLists.txt              # CMakeLists
│   │   ├── cmake                       # cpu和npu场景的cmake文件
│   │   │   └── ...
│   │   ├── run.sh                      # 用例执行总入口脚本
│   │   ├── main.cpp                    # 算子实现，main函数
│   │   └── testcase
│   │       └── case.csv                # 测试用例
│   ├── 01_misplace_core_matmul
│   │   └── ...
│   ├── 02_batch_matmul
│   │   └── ...
│   ├── 03_quant_matmul
│   │   └── ...
│   ├── 04_l2_misplace_core_matmul
│   │   └── ...
│   ├── 05_l2_misplace_core_batchmatmul
│   │   └── ...
│   ├── 06_l2_misplace_core_quant_matmul
│   │   └── ...
│   ├── 07_native_matmul
│   │   └── ...
│   ├── 08_sparse_matmul
│   │   └── ...
│   └── scripts                        # python脚本（用例执行、真值生成、精度比对）
│       └── ...
├── include                            # ACT模板头文件
│   ├── epilogue                       # 后处理代码头文件
│   │   └── ...
│   ├── matmul
│   │   ├── block                      # Block层代码头文件
│   │   │   └── ...
│   │   ├── device                     # Deivce层代码头文件
│   │   │   └── ...
│   │   ├── kernel                     # Kernel层代码头文件
│   │   │   └── ...
│   │   ├── matmul_intf.h              # Block层对外接口文件
│   │   ├── policy
│   │   │   └── ...
│   │   └── tile                       # Tile层代码头文件
│   │       └── ...
│   └── utils                          # 公共方法
│       └── ...
├── test                               # ACT UT用例
└── README.md
```

## 样例运行验证
开发者调用ACT实现自定义算子后，可通过单算子调用的方式验证算子功能。本仓提供部分算子实现及其调用样例，具体请参考[examples](./examples)目录下的样例。

以`00_basic_matmul`算子样例为例，说明样例运行验证的步骤：

1. （可选）修改用例

    查看样例下的`testcase/case.csv`文件，按照格式修改或增加用例。
    
   ```bash
   # 切换到测试用例的目录
   vim ./examples/00_basic_matmul/testcase/case.csv
   ```
    ```bash
    # 是否执行|用例名|m轴|n轴|k轴
    1,case001,1,1,1
    1,case002,2,2,2
    1,case003,128,256,128
    ...
    ```
   
2. 编译并执行算子样例

    进入样例根目录，执行如下命令:
    
    ```bash
    # 切换到样例的目录
    cd ./examples/00_basic_matmul
    # 编译执行命令
    bash run.sh -r npu -v Ascendxxxyy 
    ```
    
    若提示如下信息，则说明算子运行成功，精度比较通过。更详细的用例执行流程请参阅[快速入门](./docs/01_quickstart.md)。
    ```bash
    INFO:root:---------------RESULT---------------
    INFO:root:['case_name', 'wrong_num', 'total_num', 'result', 'task_duration']
    INFO:root:['case001', 0, 1, 'Success']
    INFO:root:['case002', 0, 4, 'Success']
    INFO:root:['case003', 0, 32768, 'Success']
   ```
## 相关文档
- [快速入门](./docs/01_quickstart.md) - 使用ACT实现算子的快速入门教程。
- [编程指南](./docs/02_programming_guidelines.md) - 使用ACT实现算子开发的教程。
- [Layout](./docs/03_layout.md) - 介绍ACT中Layout的概念和使用方式。
- [ACT分层结构](./docs/04_design.md) - 介绍ACT的编程模型、各分层的功能和接口。
