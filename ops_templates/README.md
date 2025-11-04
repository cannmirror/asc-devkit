# ops templates

## 概述
Ascend C算子模板库（ops templates），是昇腾硬件上面向高性能算子开发场景的C++模板库。

Ascend C算子模板库旨在显著提升开发者在昇腾硬件上进行定制化算子开发的效率和性能。通过提供结构清晰、组件化且可复用的模板，Ascend C算子模板库降低了高性能算子开发的门槛，使开发者能够更加专注于核心计算逻辑。目前，Ascend C算子模板库包含两大不同计算类型的模板库：
- [ACT（Ascend C Templates）](act/README.md)，基于Ascend C开发的高性能Cube类算子模板库，用于昇腾硬件上矩阵乘类融合算子的定制化开发。

- [ATVC（Ascend C Templates for Vector Compute）](atvc/README.md)，是为基于Ascend C开发的典型Vector算子封装的一系列模板头文件的集合，用于快速开发使用昇腾Vector计算单元的典型算子。

## 环境准备

使用Ascend C算子模板库前，请根据如下步骤完成相关环境准备。
1. **确认产品型号**

   请确认产品型号为：Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件。

2. **安装依赖**

   安装以下依赖，安装方式请参考[安装依赖](../README.md#dependence)。

   - python >= 3.7.0

   - gcc >= 7.3.0

   - cmake >= 3.16.0

3. **安装CANN开发套件包**

   安装以下版本的CANN开发套件包，具体安装方式请参考[安装开发套件包](../README.md#canninstall)。
   
   - CANN >= 8.2.RC1.alpha003
   
4. **设置环境变量**

   安装CANN软件包后，请按照[设置环境变量](../README.md#envset) 的方式设置相关环境变量。

5. **编译安装ascend-c软件包**
   由于CANN开发套件包中不包含Ascend C 算子模板库的源码，用户需要基于本开源仓源码，编译和安装ascend-c软件包，然后使用Ascend C 算子模板库的能力，具体编译安装方式请参考[编译安装](../README.md#compile&install)。

