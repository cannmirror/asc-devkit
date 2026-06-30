# Exp Tensor API 样例

## 概述

本样例基于Tensor API的 `Transform<Inst::Exp>`接口实现按元素取自然指数（$e^x$）。

本样例使用Tensor API的Transform模板接口进行矢量计算，使用MakeCopy/Copy进行数据搬运，并通过MakeTensor/ MakeMemPtr/MakeFrameLayout构建 Tensor 对象。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```plain
├── exp_tensor_api
│   ├── scripts
│   │   └── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── exp_tensor_api.asc      // Ascend C算子实现 & 调用样例
```

## 样例描述

- 样例功能：  
  按元素取自然指数，计算公式如下：

  $$
  dstLocal_i = Exp(srcLocal_i) = e^{srcLocal_i}
  $$

- 样例规格：  
  <table>
  <caption>表1：样例输入输出规格</caption>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center"> exp_tensor_api </td></tr>

  <tr><td rowspan="3" align="center">样例输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[64, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">样例输出</td></tr>
  <tr><td align="center">dst</td><td align="center">[64, 128]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">exp_tensor_api_custom</td></tr>
  </table>

- 样例实现：  
  本样例中实现的是固定 shape 为输入 src[1, 8192]，输出 dst[1, 8192] 的 exp_tensor_api 样例。

  - Kernel 实现

    使用 Tensor API 的 `Transform<Inst::Exp>(dstUb, srcUb)` 接口计算自然指数。

    主要流程：
    1. 使用 `MakeTensor` + `MakeMemPtr` + `MakeFrameLayout` 创建 GM 和 UB 上的 Tensor 对象
    2. 使用 `MakeCopy(CopyGM2UB{})` 构造 CopyAtom，通过 `Copy` 将数据从 GM 搬运到 UB
    3. 使用 `SetFlag` / `WaitFlag` 进行 MTE2→V 流水同步
    4. 使用 `Transform<Inst::Exp>` 执行自然指数计算
    5. 使用 `SetFlag` / `WaitFlag` 进行 V→MTE3 流水同步
    6. 使用 `MakeCopy(CopyUB2GM{})` 构造 CopyAtom，通过 `Copy` 将结果从 UB 搬运回 GM

  - 调用实现

    使用内核调用符 `<<<>>>` 调用核函数。

## 核心接口说明

### Transform

```cpp
template<typename CalcFunc, typename TraitType = Std::ignore_t, typename... Args>
__aicore__ inline void Transform(const Args&... args)
```

对于 Exp 一元计算，调用方式为：
```cpp
Transform<Inst::Exp>(dstTensor, srcTensor)
```

### MakeCopy / Copy

```cpp
// 构造 CopyAtom
auto copyInAtom = MakeCopy(CopyGM2UB{});
auto copyOutAtom = MakeCopy(CopyUB2GM{});

// 执行搬运
Copy(copyInAtom, dstTensor, srcTensor);
```

### MakeTensor

```cpp
auto tensor = MakeTensor(
    MakeMemPtr<Location::GM>(ptr),
    MakeFrameLayout<NDLayoutPtn>(_1{}, AscendC::Std::Int<8192>{}));
```

## 编译运行  

在本样例根目录下执行如下步骤，编译并执行算子。

- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../../docs/quick_start.md#prepare&install)，配置环境变量。
  ```bash
  source ${install_path}/cann/set_env.sh
  ```

- 样例执行

  在本样例目录下执行如下命令。
  ```bash
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # 编译工程，默认npu模式
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  ```

  使用 CPU调试 或 NPU仿真 模式时，添加 `-DCMAKE_ASC_RUN_MODE=cpu` 或 `-DCMAKE_ASC_RUN_MODE=sim` 参数即可。
  
  示例如下：

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # cpu调试模式
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU仿真模式
  ```

  > **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

- 编译选项说明

  | 选项 | 可选值 | 说明 |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu`（默认）、`cpu`、`sim` | 运行模式：NPU 运行、CPU调试、NPU仿真 |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU 架构：dav-3510 对应 Ascend 950PR/Ascend 950DT 及 Atlas A3 训练系列产品/Atlas A3 推理系列产品 |

- 执行结果

  执行结果如下，说明精度对比成功。

  ```bash
  test pass!