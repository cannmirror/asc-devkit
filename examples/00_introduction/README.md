# Introduce样例介绍
## 概述
简单的样例，适合初学者。

## 自定义算子样例说明 
样例通过Ascend C编程语言实现了自定义算子，并按照不同的算子调用方式分别给出了对应的<<<>>>直调实现。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |  支持的AI处理器 |
| ------------------------------------------------------------ | ---------------------------------------------------- | -- |
| [00_helloworld](./00_helloworld) | 基于Ascend C的HelloWorld自定义算子调用结构演示样例 | Ascend 910B |
| [01_add](./01_add) | 基于Ascend C的Add自定义Vector算子的核函数直调方法，支持main函数和Kernel函数在同一个cpp文件中实现 | Ascend 910B |
| [02_matmul](./02_matmul) | 基于Ascend C的Matmul算子的核函数直调方法 | Ascend 910B |
| [03_matmulleakyrelu](./03_matmulleakyrelu) | 基于Ascend C的MatmulLeakyRelu自定义算子的核函数直调方法，支持main函数和Kernel函数在同一个cpp文件中实现 | Ascend 910B |
| [04_addn](./04_addnh) | 基于Ascend C的AddN自定义Vector算子，介绍了单算子直调方法 | Ascend 910B |
| [05_broadcast](./05_broadcast) | 基于Ascend C的Broadcast自定义Vector算子，介绍了单算子直调方法  | Ascend 910B |
| [06_reduce](./06_reduce) | 基于Ascend C的Reduce自定义Vector算子，以直调的方式调用算子核函数 | Ascend 910B |
| [07_sub](./07_sub) | 基于Ascend C的Sub自定义Vector算子,介绍了单算子直调方法 | Ascend 910B |
| [08_unaligned_abs](./08_unaligned_abs) | 基于Ascend C的DataCopyPad的非对齐Abs算子核函数直调方法 | Ascend 910B |
| [09_unaligned_reducemin](./09_unaligned_reducemin) | 基于Ascend C的无DataCopyPad的非对齐ReduceMin算子核函数直调方法 | Ascend 910B |
| [10_unaligned_wholereduces](./10_unaligned_wholereduces) | 基于Ascend C的非对齐WholeReduceSum算子的核函数直调方法 | Ascend 910B |
| [11_vectoradd](./11_vectoradd) | 基于Ascend C的Add算子的核函数直调方法，算子支持单核运行 | Ascend 910B |

## 获取样例代码<a name="codeready"></a>

 可以使用以下两种方式下载，请选择其中一种进行源码准备。

 - 命令行方式下载（下载时间较长，但步骤简单）。

   ```bash
   # 开发环境，非root用户命令行中执行以下命令下载源码仓。git_clone_path为用户自己创建的某个目录。
   cd ${git_clone_path}
   git clone https://gitcode.com/cann/asc-devkit-dev.git
   ```
   **注：如果需要切换到其它tag版本，以v0.5.0为例，可执行以下命令。**
   ```bash
   git checkout v0.5.0
   ```
 - 压缩包方式下载（下载时间较短，但步骤稍微复杂）。

   **注：如果需要下载其它版本代码，请先请根据前置条件说明进行asc-devkit-dev仓分支切换。下载压缩包命名跟tag/branch相关，此处以master分支为例，下载的名字将会是asc-devkit-dev-master.zip**
   ```bash
   # 1. 仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。
   # 2. 将ZIP包上传到开发环境中的普通用户某个目录中，【例如：${git_clone_path}/asc-devkit-dev-master.zip】。
   # 3. 开发环境中，执行以下命令，解压zip包。
   cd ${git_clone_path}
   unzip asc-devkit-dev-master.zip
   ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/06 | 样例目录调整，新增本readme |