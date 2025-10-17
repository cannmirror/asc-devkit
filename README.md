# Ascend C

## 🔥Latest News

- [2025/11] Ascend C项目首次上线。

## 🚀概述

Ascend C是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）针对算子开发场景推出的编程语言，原生支持C和C++标准规范，兼具开发效率和运行性能。基于Ascend C编写的算子程序，通过编译器编译和运行时调度，运行在昇腾AI处理器上。使用Ascend C，开发者可以基于昇腾AI硬件，高效的实现自定义的创新算法。

Ascend C在CANN架构中的位置如下图所示：

![原理图](docs/figures/architecture.png)

Ascend C提供一组类库API，开发者可以使用标准C++语法和类库API进行编程。Ascend C编程类库分为算子模板库、AICore API（高阶API、基础API、HOST API、C++标准库）等。另外，Ascend C提供一系列用于算子开发的调试工具以及强大的算子工程能力。

### 算子模板库

- **ACT**

  [ACT](./ops_templates/act/README.md)（Ascend C Templates）是基于Ascend C开发的高性能Cube类算子模板库，专门用于昇腾硬件上的矩阵乘类融合算子定制化开发。该库采用模块化分层架构，将复杂算子分解为可灵活组合的层级组件，开发者通过拼接复用即可高效构建自定义算子。支持接口独立替换与自定义扩展，内置Tiling自动推导与静态检查能力以提升调试效率，并深度优化硬件适配，最大化释放NPU算力。整体兼顾开发效率与极致性能，显著降低高性能算子开发门槛。

- **ATVC**

  [ATVC](./ops_templates/atvc/README.md)（Ascend C Templates for Vector Compute）是一个为Ascend C典型Vector算子封装的模板头文件集合，提供了3类典型Vector算子的通用Tiling计算接口和kernel模板类，模板内部自动完成算子的数据搬入搬出等底层通用操作，兼顾开发效率与算子性能，帮助用户快速完成Ascend C算子开发。当前已支持ElementWise、Reduce、Broadcast三类算子。

### AI Core API
- **高阶API**

  高阶API（Advance API）是基于单核对常见算法的抽象和封装，用于提高编程开发效率。高阶API是通过封装基础API或微指令来实现的，主要包括数学库、Matmul、量化反量化、数据归一化等API，详细API列表请见[Ascend C高阶API列表](./docs/aicore/adv_api/README.md)。开发者可根据实际业务需要，选择合适的高阶API进行自定义算子开发，降低算子开发编程难度。开发者也可以根据需求对高阶API进行修改或开发其他高阶API，并将自定义高阶API编译部署到CANN软件环境中使用。
- **基础API**

  基础API一般是基于对硬件能力的抽象和封装，从下至上分为高维切分，连续计算等API，高维切分类API更接近硬件原生能力，越往上会更易用。除此之外，该目录还包含了一些AscendC框架API，诸如内存管理与同步类接口TPipe，用于用户进行基于Stage编程的能力，帮助用户较轻松解决的算子开发中的流水同步管理问题，提高编程开发效率。基础API中基本分为搬运类API和计算类API、详细API列表请见[Ascend C基础API列表](./docs/aicore/basic_api/README.md)。
- **HOST API**

  HOST API是AscendC用于用户在HOST侧，完成算子原型注册管理，获取平台信息以及进行算子Tiling调用及调试等接口。详细API列表请见[Ascend C Host API列表](./docs/aicore/host_api/README.md)。
- **工具类**

  工具类API当前包含C++标准库的实现。C++标准库中提供一些常见的c++标准库函数，提供模板类编程能力，主要包括算法、数学函数、容器函数、类型特征和通用工具等。使用C++标准库允许用户编写API时使用C++模板元编程能力，提高API的可读性和可维护性。

### 调试工具

  调试工具是Ascend C提供的用于帮助算子开发、算子调试、算子调优等一系列工具。其中的孪生调试工具用于CPU域调试精度，提升算子开发效率；model2trace、msobjdump、show_kernel_debug_data等工具用于对算子的相关调试信息进行解析。

### 算子工程

  算子工程用于调用Ascend C算子，包括Kernel Launch、 单算子API调用、算子入图、AI框架调用等功能。算子工程中的内置cmake用于完成算子编译和部署，其包含了多个Kernel编译脚本、编译过程所需要的一些工具，如用于生成算子信息库、原型库、单算子调用接口的opbuild可执行，用于动态库打包的elf工具等。


## 🔍目录结构说明

本代码仓目录结构如下：

```
├── cmake                               // Ascend C 构建源代码
├── docs                                // Ascend C API使用说明
├── examples                            // Ascend C API样例工程
├── impl                                // Ascend C API接口实现源代码
│   └── aicore                          // Ascend C AICore 编程接口实现源代码
│       ├── adv_api                     // Ascend C 高阶API实现源代码
│       ├── basic_api                   // Ascend C 基础API实现源代码
│       ├── host_api                    // Ascend C HOST API实现源代码
│       ├── micro_api                   // Ascend C 微指令API实现源代码
│       ├── simt_api                    // Ascend C SIMT API实现源代码
│       └── utils                       // Ascend C 工具类实现源代码
├── include                             // Ascend C API接口声明源代码
│   └── aicore                          // Ascend C AICore 编程接口声明源代码
│       ├── adv_api                     // Ascend C 高阶API声明源代码
│       ├── basic_api                   // Ascend C 基础API声明源代码
│       ├── host_api                    // Ascend C HOST API声明源代码
│       ├── micro_api                   // Ascend C 微指令API声明源代码
│       ├── simt_api                    // Ascend C SIMT API声明源代码
│       └── utils                       // Ascend C 工具类声明源代码
├── ops_templates                       // Ascend C 算子模板库源代码
│   ├── act                             // Ascend C 高性能Cube类算子模板库源代码
│   └── atvc                            // Ascend C 典型Vector算子模板库源代码
├── test                                // Ascend C API的UT用例
└── tools                               // Ascend C 工具源代码
```

## ⚡️快速入门

若您希望快速体验项目的构建和样例的执行，请访问如下文档获取简易教程。

- [构建](docs/00_quick_start.md)：介绍搭建环境、编译执行、本地验证等操作。
- [样例执行](examples/README.md)：介绍如何端到端执行样例代码。



## 📖相关文档

| 文档  |  说明   |
|---------|--------|
|[Ascend C资料书架](./docs/README.md)|总览Ascend C相关文档及视频资料。|
|[自定义开发API指南](./docs/01_adv_api_programming_guide.md)|介绍如何基于Ascend C开发高阶API。|
|[模板库](./ops_templates/README.md)|介绍Ascend C模板库。|
|[高阶API列表](./docs/aicore/adv-api/README.md)|总览Ascend C高阶API。|
|[基础API列表](./docs/aicore/basic-api/README.md)|总览Ascend C基础API。|
|[工具类API列表](./docs/aicore/utils/README.md)|总览Ascend C 工具类API。|
|[HOST API列表](./docs/aicore/simt-api/README.md)|总览Ascend C HOST API。|
|[变更日志](./CHANGELOG.md)|介绍各版本特性变更。|

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)