# Ascend C

## 🔥Latest News
- [2026/01] 优化样例、融合编译性能等；
  - 迁移高阶API样例[ascendc-api-adv](https://gitee.com/ascend/ascendc-api-adv)到本仓，并使用[<<<>>>调用方式](./examples/03_libraries)；
  - 联合毕昇编译器，优化融合编译性能；
- [2025/12] Ascend C项目新增对Kirin X90和Kirin 9030处理器的支持。鸿蒙开发者基于Ascend C的实践案例可参考：[cann-recipes-harmony-infer](https://gitcode.com/cann/cann-recipes-harmony-infer)。
- [2025/11] Ascend C项目基于Atlas A3 训练系列产品/Atlas A3 推理系列产品、Atlas A2 训练系列产品/Atlas A2 推理系列产品全面开源开放，包含以下新特性：
  - 实现分仓分包，支持分包独立安装部署，包括Ascend C算子开发所需的核心仓asc-devkit、调试工具仓[asc-tools](https://gitcode.com/cann/asc-tools)、Vector算子模板库仓[atvc](https://gitcode.com/cann/atvc)和[atvoss](https://gitcode.com/cann/atvoss)、Python前端仓[pyasc](https://gitcode.com/cann/pyasc)。
  - 编程API能力扩展
    - 新增语言扩展层C API，提供与业界相似的编程体验。
    - 基础API新增LocalMemAllocator内存分配接口。
  - 全面支持异构编译与<<<>>> 直调，通过文件后缀名“.asc”或编译选项“-x asc”使能异构编译。
  - 算子编译CMake接口标准化，提供Cmake module接口，支持不同编译场景。
  - 支持CPU&NPU孪生调试的能力一致性，一套代码同时支持CPU和NPU调试。
  - 编程指南全面优化。
  - 新增算子样例。

## 🚀概述

Ascend C是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）推出的昇腾AI处理器专用的算子程序开发语言，原生支持C和C++标准规范。Ascend C主要由类库和语言扩展层构成，提供多层级API，满足多维场景算子开发诉求；总体逻辑架构图如下所示：

<img src="docs/figures/architecture.png" alt="架构图"  width="850px" height="580px">

- 语言扩展层C API：纯C接口，支持数组分配内存、基于指针的计算接口，提供与业界一致的C语言编程体验，并开放芯片完备编程能力。Atlas A2/A3支持SIMD的纯C接口；Ascend 950PR/Ascend 950DT将支持与业界类似的纯SIMT编程能力、SIMD/SIMT混合编程能力；
- 基础API：单指令抽象的C++类库API，一般基于Tensor编程；逐步基于Layout完善Tensor编程能力；
- 高阶API：基于单核对常见算法进行抽象和封装，提供公共算法的实现；
- 算子模板库：基于模板提供算子的完整实现参考，简化Tiling开发，支持用户自定义扩展；
- Python前端：PyAsc基于Python前端，提供芯片底层完备编程能力，并将逐步基于Layout完善Tensor编程能力，新增SIMT编程等能力，实现基于Python接口开发高性能算子；

本仓主要包含Ascend C编程API和必要的cmake编译脚本，是算子开发所需的核心模块。


## 🔍目录结构说明

本代码仓目录结构如下：

```
├── cmake                               # Ascend C 构建源代码
├── docs                                # 项目文档介绍
├── examples                            # Ascend C API样例工程
├── impl                                # Ascend C API接口实现源代码
│   ├── adv_api                         # Ascend C 高阶API实现源代码
│   ├── aicpu_api                       # Ascend C AI CPU API实现源代码
│   ├── basic_api                       # Ascend C 基础API实现源代码
│   ├── c_api                           # Ascend C 语言扩展层C API实现源代码
│   └── utils                           # Ascend C 工具类实现源代码
├── include                             # Ascend C API接口声明源代码
│   ├── adv_api                         # Ascend C 高阶API声明源代码
│   ├── aicpu_api                       # Ascend C AI CPU API声明源代码
│   ├── basic_api                       # Ascend C 基础API声明源代码
│   ├── c_api                           # Ascend C 语言扩展层C API声明源代码
│   └── utils                           # Ascend C 工具类声明源代码
├── scripts                             # 打包相关脚本
├── tests                               # Ascend C API的UT用例
└── tools                               # Ascend C 工具源代码
```

## ⚡️快速入门

若您希望快速体验项目的构建和算子样例的执行，请访问如下文档获取简易教程。

- [编译构建](docs/quick_start.md)：介绍搭建环境、编译执行、本地验证等操作。
- [样例执行](examples/README.md)：提供算子开发样例，介绍端到端执行样例的方式。


## 📖相关文档

若您希望深入体验项目功能，扩展原有API或开发全新API、开发自定义算子，请访问如下文档获取详细教程。
- API开发
  | 文档  |  说明   |
  |---------|--------|
  |[API列表](./docs/api/README.md)|Ascend C API列表。|
  |[API贡献指南](./CONTRIBUTING.md)|介绍如何扩展或开发Ascend C API。|

- 算子开发
  | 文档  |  说明   |
  |---------|--------|
  |[Ascend C编程指南](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)|基于昇腾AI硬件，使用Ascend C编写算子程序，开发自定义算子。|
  |[Ascend C最佳实践](https://hiascend.com/document/redirect/CannCommunityAscendCBestPractice) | 基于已完成开发的Ascend C算子，介绍如何进一步优化算子性能。 |
  |[Ascend C编程指南（鸿蒙）](https://gitcode.com/cann/cann-recipes-harmony-infer/blob/master/docs/ascendc_develop_guide.md)|基于麒麟AI硬件，使用Ascend C编写算子程序，开发自定义算子。|

## 📌后续规划

- 基于Altas A2/A3发布语言扩展层纯C接口，提供基于数组分配内存能力，支持基于指针的计算接口，实现与业界类似的纯C编程体验；
- Ascend 950PR/Ascend 950DT将支持纯SIMT编程、SIMD与SIMT混合编程，并通过Layout进一步强化Tensor编程能力；
- 持续丰富语言扩展层C API(含SIMD、SIMT)和基础API的关键特性介绍，并基于融合编译与 <<<>>>调用完善样例；

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)