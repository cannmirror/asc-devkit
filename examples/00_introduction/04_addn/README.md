# Addn算子直调样例
## 概述
本样例基于addn算子工程，介绍了单算子直调方法。
## 支持的AI处理器
- Ascend 910B
## 目录结构介绍
```
├── 04_addn        
│   ├── CMakeLists.txt          // 编译工程文件
│   └── addn_custom.asc    // AscendC算子实现 & 调用样例
```

## 算子描述
- 算子功能：  

  AddN算子实现了两个数据相加，返回相加结果的功能，其中核函数的输入参数为动态输入，动态输入参数包含两个入参，x和y。对应的数学表达式为：  
  ```
  z = x + y
  ```
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">AddN</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x（动态输入参数srcList[0]）</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y（动态输入参数srcList[1]）</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">addn_custom</td></tr>
  </table>
- 算子实现：  

  动态输入特性是指，核函数的入参采用ListTensorDesc的结构存储输入数据信息。  
  构造TensorList数据结构，示例如下。
  ```cpp
  constexpr uint32_t SHAPE_DIM = 2;
    struct TensorDesc {
      uint32_t dim{SHAPE_DIM};
      uint32_t index;
      uint64_t shape[SHAPE_DIM] = {8, 2048};
    };

  constexpr uint32_t TENSOR_DESC_NUM = 2;
    struct ListTensorDesc {
      uint64_t ptrOffset;
      TensorDesc tensorDesc[TENSOR_DESC_NUM];
      uintptr_t dataPtr[TENSOR_DESC_NUM];
    } inputDesc;
  ```
  将申请分配的Tensor入参组合成ListTensorDesc的数据结构，示例如下。
  ```cpp
  inputDesc = {(1 + (1 + SHAPE_DIM) * TENSOR_DESC_NUM) * sizeof(uint64_t),
              {xDesc, yDesc},
              {(uintptr_t)xDevice, (uintptr_t)yDevice}};
  ``` 
  按照传入的数据格式，解析出对应的各入参，示例如下。

  ```cpp
  uint64_t buf[SHAPE_DIM] = {0};
  AscendC::TensorDesc<int32_t> tensorDesc;
  tensorDesc.SetShapeAddr(buf);
  listTensorDesc.GetDesc(tensorDesc, 0);
  uint64_t totalLength = tensorDesc.GetShape(0) * tensorDesc.GetShape(1);
  __gm__ uint8_t *x = listTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
  __gm__ uint8_t *y = listTensorDesc.GetDataPtr<__gm__ uint8_t>(1);
  ```
  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行
  - 配置环境变量   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/examples/00_introduction/04_addn/
    ```
    请根据当前环境上CANN开发套件包的[安装方式](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，选择对应配置环境变量的命令。
    - 默认路径，root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
      ```
    - 默认路径，非root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
      ```
    - 指定路径install_path，安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
      ```

  - 样例执行
    ```bash
    mkdir -p build && cd build;   # 创建并进入build目录
    cmake ..;make -j;             # 编译工程
    ./demo                       # 执行编译生成的可执行程序，执行样例
    ```
    执行结果如下，说明精度对比成功。
    ```bash
    [Success] Case accurary is verification passed.
    ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/06 | 样例目录调整，新增本readme |