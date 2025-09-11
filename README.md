# Ascend C
## 概述

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


## 版本配套说明

- 本项目会创建与CANN软件版本适配的标签并发行，两者的配套关系请参见"[开放项目与CANN版本配套表](https://gitee.com/ascend/cann-community/blob/master/README.md#cannversionmap)"。**需注意，为确保您的源码定制开发顺利进行，请选择配套的CANN版本与GitCode标签源码，使用master分支可能存在版本不匹配风险。**

- 本项目支持的固件驱动版本与配套CANN软件支持的固件驱动版本相同，开发者可通过“[昇腾社区-固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community?product=2&model=28)”，根据产品型号与CANN软件版本获取配套的固件与驱动。

## 目录结构说明

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

## 环境准备
ascend-c项目支持由源码编译，进行源码编译前，请根据如下步骤完成相关环境准备。

1. **获取CANN开发套件包**

   请参见“[开放项目与CANN版本配套表](https://gitee.com/ascend/cann-community/blob/master/README.md#cannversionmap)”获取对应的CANN开发套件包`Ascend-cann-toolkit_<cann_version>_linux-<arch>.run`，CANN开发套件包支持的安装方式及操作系统请参见配套版本的[用户手册](https://hiascend.com/document/redirect/CannCommunityInstSoftware)。

2. **安装依赖**<a name=dependence></a>

   以下所列仅为ascend-c源码编译用到的依赖，其中python、gcc、cmake的安装方法请参见配套版本的[用户手册](https://hiascend.com/document/redirect/CannCommunityInstDepend)，选择安装场景后，参见“安装CANN > 安装依赖”章节进行相关依赖的安装。

   - python >= 3.7.0

   - gcc >= 7.3.0

   - cmake >= 3.16.0

   - googletest（可选，仅执行UT时依赖，建议版本[release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

     下载[googletest源码](https://github.com/google/googletest.git)后，执行以下命令安装：

     ```bash
     mkdir temp && cd temp                 # 在googletest源码根目录下创建临时目录并进入
     cmake .. -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
     make
     make install                         # root用户安装googletest
     # sudo make install                  # 非root用户安装googletest
     ```

3. **安装CANN开发套件包**<a name=canninstall></a>

   执行安装命令时，请确保安装用户对软件包具有可执行权限。

   - 使用默认路径安装

     ```bash
     # CANN开发套件包安装命令示例：
     ./Ascend-cann-toolkit_<cann_version>_linux-<arch>.run --install
     ```

     - 若使用root用户安装，安装完成后相关软件存储在`/usr/local/Ascend/ascend-toolkit/latest`路径下。
     - 若使用非root用户安装，安装完成后相关软件存储在`$HOME/Ascend/ascend-toolkit/latest`路径下。

   - 指定路径安装

     ```bash
     # CANN开发套件包安装命令示例：
     ./Ascend-cann-toolkit_<cann_version>_linux-<arch>.run --install --install-path=${install_path}
     ```

     安装完成后，相关软件存储在\${install_path}指定路径下。

4. **设置环境变量**<a name=envset></a>

   - 默认路径，root用户安装

     ```bash
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     ```

   - 默认路径，非root用户安装

     ```bash
     source $HOME/Ascend/ascend-toolkit/set_env.sh
     ```

   - 指定路径安装

     ```bash
     source ${install_path}/ascend-toolkit/set_env.sh
     ```

   **注意：若环境中已安装多个版本的CANN软件包，设置上述环境变量时，请确保${install_path}/ascend-toolkit/latest目录指向的是配套版本的软件包。**
   
   
## 源码下载
执行如下命令，下载ascend-c仓源码：
```bash
git clone -b ${tag_version} https://gitcode.com/cann/ascend-c.git
```
${tag_version}请替换为具体的标签名称，本源码仓与CANN版本的配套关系可参见"开放项目与CANN版本配套表"。

## 编译安装<a name=compile&install></a>

1. 编译

   ascend-c仓提供一键式编译安装能力，进入本开源仓代码根目录，执行如下命令：

   ```bash
   bash build.sh
   ```

   编译完成后会在`output`目录下生成CANN-ascend_c-*<cann_version>*-linux.*<arch>*.run软件包。
2. 安装

   在开源仓根目录下执行下列命令，根据设置的环境变量路径，将编译生成的run包安装到CANN包的装包路径，同时会覆盖原CANN包中的高阶API内容。

   ```bash
   # 设置CANN开发套件包环境变量，以root用户默认路径为例，如已设置，则可忽略该操作
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # 切换到run包生成路径下
   cd output
   # 安装run包
   ./CANN-ascend_c-<cann_version>-linux.<arch>.run
   ```

## UT测试（可选）

在开源仓根目录执行下列命令之一，将依次批跑tests目录下的用例，得到结果日志，用于看护编译是否正常。

```bash
bash build.sh -t
```

或

```bash
bash build.sh --test
```

## 样例运行验证（可选）

开发者调用高阶API实现自定义算子后，可通过单算子调用的方式验证算子功能。本代码仓提供部分算子实现及其调用样例，具体请参考[examples](./examples)目录下的样例。

## 相关文档

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

## 贡献指南<a name=contribute></a>

ascend-c仓欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[cann-community](https://gitcode.com/ascend/cann-community)了解行为准则，进行CLA协议签署，以及参与开源仓贡献的详细流程。

针对ascend-c仓，开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。

## 许可证
[CANN Open Software License Agreement Version 1.0](LICENSE)