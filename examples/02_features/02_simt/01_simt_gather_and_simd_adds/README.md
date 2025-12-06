# SIMT & SIMD混合编程实现gather & adds算子样例

## 概述
本样例以实现gather和adds功能的算子为例，展示了SIMT和SIMD混合编程的算子开发方式。算子中使用SIMT编程方式实现离散内存访问操作gather，使用SIMD编程方式实现连续内存访问操作adds。

## 支持的AI处理器
- Ascend 910D

## 目录结构
```
├── 01_simt_gather_and_simd_adds
│   ├── CMakeLists.txt         # cmake编译文件
│   ├── gather_and_adds.asc    # Ascend C算子实现 & 调用样例
|   └── README.md
```

## 算子描述

- 算子功能：  
  gather & adds算子实现了从长度为10万的一维向量中获取指定索引的8192个数据，将获取到的数据分别加1，返回相加结果的功能。算子输出output第i个数据计算公式为：
  
  ```
  output[i] = input[index[i]] + 1
  ```

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">gather & adds</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">100000</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">index</td><td align="center">8192</td><td align="center">uint32_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">output</td><td align="center">8192</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gather_and_adds_kernel</td></tr>
  </table>

- 基础知识：  
  在SIMT编程中，可以直接读取和使用Global Memory上的数据。在Vector Core中，SIMT单元和SIMD单元共享片上存储，因此可以使用片上存储完成SIMT和SIMD的混合编程。本例中，首先使用SIMT编程方式获取指定索引位置的输入数据，存储到UB，再使用SIMD编程方式利用UB完成adds运算。

  具体来说，SIMT编程常通过组织线程的层次结构来实现数据的切分，函数中使用threadIdx等内置变量计算线程应处理的数据索引，完成索引对应数据的计算，函数被简化为标量计算。SIMD编程提供基于寄存器（Regbase）的编程API，可以直接操作Vector Core中的寄存器。单次API处理的数据量上限为寄存器大小，通过AscendC::GetVecLen函数获取该值。在计算过程中，需要多次调用微指令API完成单核数据的处理。

- 数据切分：  
  本例中算子输入index的形状为8192，可设置核数为8，每个核处理数据量为1024，设置线程数THREAD_COUNT为1024，每个线程处理1个数据，因此单个核只需调用1次simt_gather函数即可完成gather运算。

  > ⚠️ **注意** 当单核处理数据量大于设置的线程数时，需要切分数据到多个线程块，可使用asc_vf_call多次调用simt_gather函数启动多个线程块完成获取指定索引数据的操作。

  基于上述数据拆分，在simd_adds函数中，处理1024个数据的加1操作。

  > ⚠️ **注意** simd_adds中加1运算实际可以直接在simt_gather函数中快速实现，本例目的仅仅是通过一个简单用例展示SIMT和SIMD两种编程模式的混合编程方式，不是该算子最佳实践。

- 算子实现：  
  gather & adds算子的实现流程主要分为3个步骤：simt_gather，simd_adds和DataCopy。
  
  simt_gather负责从输入input（Global Memory）中获取指定索引的数据。基于上述数据切分，首先计算线程应处理数据的索引，然后通过赋值操作将数据存储到Local Memory上。
  ```
  int idx = asc_get_block_idx() * blockDim.x + threadIdx.x;
  ...
  uint32_t gather_idx = index[idx];
  ...
  gather_output[threadIdx.x] = input[gather_idx];
  ```

  simd_adds负责将Local Memory中数据做加1操作。使用MicroAPI::LoadAlign将数据从Local Memory搬运到寄存器上，调用MicroAPI::Adds完成加1运算并输出到目标寄存器，最后调用MicroAPI::StoreAlign将数据从寄存器搬运到Local Memory。重复上述操作即可完成1024个数据的加1运算。
  ```
  for (uint16_t i = 0; i < repeat_times; i++) {
      AscendC::MicroAPI::LoadAlign(src_reg0, input + i * one_repeat_size);
      AscendC::MicroAPI::Adds(dst_reg0, src_reg0, ADDS_ADDEND, mask_reg);
      AscendC::MicroAPI::StoreAlign(output + i * one_repeat_size, dst_reg0, mask_reg);
  }
  ```

  DataCopy负责将输出数据从Local Memory搬运至Global Memory上。

- 调用实现：    
  使用内核调用符<<<>>>调用核函数。

## 编译运行
在本样例根目录下执行如下步骤，编译并执行算子。
- 配置环境变量   
  请根据当前环境上CANN开发套件包的[安装方式](../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    source /usr/local/Ascend/cann/set_env.bash
    ```
  - 默认路径，非root用户安装CANN软件包
    ```bash
    source $HOME/Ascend/cann/set_env.bash
    ```
  - 指定路径install_path，安装CANN软件包
    ```bash
    source ${install_path}/cann/set_env.bash
    ```
- 样例执行
  ```bash
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake ..; make -j;            # 编译工程
  ./demo                        # 执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```
  [Success] Case accuracy is verification passed.
  ```