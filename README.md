# Ascend C

## 🔥Latest News

- [2025/11] Ascend C项目首次上线。

## 🚀概述

Ascend C是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）针对算子开发场景推出的编程语言，原生支持C和C++标准规范，兼具开发效率和运行性能。基于Ascend C编写的算子程序，通过编译器编译和运行时调度，运行在昇腾AI处理器上。使用Ascend C，开发者可以基于昇腾AI硬件，高效的实现自定义的创新算法。Ascend C及其周边模块的架构图如下所示：

![原理图](docs/figures/architecture.png)

Ascend C主要由类库和语言扩展层构成，同时提供全面的算子工程能力、调试调优工具链及一系列公共辅助函数。语言扩展层提供纯C接口，通常基于指针计算；类库分为多核算子样例、单核公共算法和单指令三类。多核算子样例类库对应算子模板库，包括由[CATLASS](https://gitcode.com/cann/catlass)和[ATCOS](https://gitcode.com/cann/atcos)组成的Cube类模板库，以及由[ATVC](https://gitcode.com/cann/atvc)和ATVOS组成的Vector类模板库。单核公共算法类库对应高阶API，这类API基于单核对常见算法进行抽象和封装，旨在提高编程开发效率。单指令类库分为SIMD（Single Instruction Multiple Data，单指令多数据）和SIMT（Single Instruction Multiple Thread，单指令多线程），其中SIMD类库接口包括基于内存上的Tensor的基础API，以及未来即将支持的基于寄存器上的Tensor的微指令API，基础API通常是对硬件能力的抽象和封装。

开发者使用Ascend C编写的算子代码，依赖毕昇编译器编译成二进制可执行文件和动态库等形式。借助Ascend C丰富的类库接口能力，Ascend C Python层当前提供了[PyAsc](https://gitcode.com/cann/pyasc)编程语言，原生支持Python标准规范，PyAsc接口与Ascend C API一一映射，具备完备的编程能力。

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
├── test                                // Ascend C API的UT用例
└── tools                               // Ascend C 工具源代码
```

## ⚡️快速入门

若您希望快速体验项目的构建和样例的执行，请访问如下文档获取简易教程。

- [构建](docs/quick_start.md)：介绍搭建环境、编译执行、本地验证等操作。
- [样例执行](examples/README.md)：介绍如何端到端执行样例代码。



## 📖相关文档

| 文档  |  说明   |
|---------|--------|
|[Ascend C资料书架](./docs/README.md)|总览Ascend C相关文档及视频资料。|
|[自定义开发API指南](./docs/adv_api_programming_guide.md)|介绍如何基于Ascend C开发高阶API。|
|[API列表](./docs/api/README.md)|Ascend C API列表。|


## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)