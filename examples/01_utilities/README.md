# Utilities样例介绍
## 概述
样例通过Ascend C编程语言实现了自定义算子，实现assert及debug功能、硬件平台信息的查询能力等，并给出了对应的<<<>>>直调实现。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_printf](./00_printf) | 本样例通过Ascend C编程语言实现了Matmul算子，同时在算子中添加printf调测 |
| [01_assert](./01_assert) | 本样例通过Ascend C编程语言实现了Matmul算子，同时在算子中添加assert调测，给出了对应的端到端实现 |
| [02_dumptensor](./02_dumptensor) | 本样例通过Ascend C编程语言实现了Add算子和Mmad算子，在算子中添加DumpTensor调测，给出了对应的端到端实现 |
| [03_cpudebug](./03_cpudebug) | 本样例通过Ascend C编程语言实现了Add算子的CPU Debug调测，给出了对应的端到端实现 |

## 获取样例代码<a name="codeready"></a>

 可以使用以下两种方式下载，请选择其中一种进行源码准备。

 - 命令行方式下载（下载时间较长，但步骤简单）。

   ```bash
   # 开发环境，非root用户命令行中执行以下命令下载源码仓。git_clone_path为用户自己创建的某个目录。
   cd ${git_clone_path}
   git clone https://gitcode.com/cann/asc-devkit-dev.git
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
| 2025/11/18 | 样例目录调整，新增本readme |