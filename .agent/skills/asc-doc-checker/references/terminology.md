# 术语规范参考

来源：CANN 社区版术语表

## 一、缓冲区术语（必须带 Buffer 后缀）

| 规范写法 | 错误写法 | 含义 |
|----------|----------|------|
| L0A Buffer | L0A | AI Core内部物理存储单元，存储矩阵计算左矩阵，对应TPosition::A2 |
| L0B Buffer | L0B | AI Core内部物理存储单元，存储矩阵计算右矩阵，对应TPosition::B2 |
| L0C Buffer | L0C | AI Core内部物理存储单元，存储矩阵计算结果，对应TPosition::CO1 |
| L1 Buffer | L1（指代物理缓冲区时） | AI Core内部物理存储单元，缓存矩阵输入数据，对应TPosition::A1/B1 |
| Unified Buffer | — | AI Core内部存储单元，用于矢量计算，对应TPosition::VECIN/VECOUT/VECCALC |
| UB | — | Unified Buffer的缩写，两者均可使用 |
| BiasTable Buffer | BiasTable / BT Buffer | 偏置存储，对应TPosition::C2 |
| Fixpipe Buffer | Fixpipe | 存储Fixpipe搬运量化参数，对应TPosition::C2PIPE2GM |
| SSBuffer | — | 分离模式下Cube Core和Vector Core通信使用的缓冲区 |

## 二、逻辑内存位置（TPosition）

| 规范写法 | 对应物理存储 | 用途 |
|----------|-------------|------|
| TPosition::A1 | L1 Buffer | 存放左矩阵 |
| TPosition::A2 | L0A Buffer | 存放小块左矩阵 |
| TPosition::B1 | L1 Buffer | 存放右矩阵 |
| TPosition::B2 | L0B Buffer | 存放小块右矩阵 |
| TPosition::C1 | L1 Buffer / Unified Buffer | 存放Bias偏置数据 |
| TPosition::C2 | BiasTable Buffer / L0C Buffer | 存放小块Bias数据 |
| TPosition::C2PIPE2GM | Fixpipe Buffer | 存放量化参数 |
| TPosition::CO1 | L0C Buffer | 存放小块矩阵计算结果 |
| TPosition::CO2 | Global Memory / Unified Buffer | 存放矩阵计算结果 |
| TPosition::VECIN | Unified Buffer | 矢量计算输入 |
| TPosition::VECCALC | Unified Buffer | 矢量计算临时变量 |
| TPosition::VECOUT | Unified Buffer | 矢量计算输出 |
| TPosition::TSCM | L1 Buffer | L1 Buffer空间，用于Matmul计算 |
| TPosition::LCM | Unified Buffer | 临时共享UB空间 |

## 三、缩写词标准对照

| 缩写 | 全称 | 含义 |
|------|------|------|
| AI Core | — | AI处理器的计算核 |
| AIC | — | AI Core分离模式下的Cube Core |
| AIV | — | AI Core分离模式下的Vector Core |
| Block | — | AI Core的逻辑核 |
| BlockID | — | 以0为起始的AI Core逻辑编号 |
| Core | — | 拥有独立Scalar计算单元的计算核 |
| Core ID | — | AI Core核的物理编号 |
| Cube | — | AI Core上的矩阵计算单元 |
| Cube Core | — | 矩阵计算核（Scalar + 矩阵单元 + 搬运单元） |
| Vector | — | AI Core上的矢量计算单元 |
| Vector Core | — | 矢量计算核（Scalar + 矢量单元 + 搬运单元） |
| Scalar | — | 标量计算单元，负责指令发射与标量运算 |
| DCache | Data Cache | 数据缓存 |
| ICache | Instruction Cache | 指令缓存 |
| DMA | Direct Memory Access | 直接内存访问单元 |
| GM | Global Memory | 设备端主内存 |
| MTE1 | Memory Transfer Engine 1 | L1→L0A/L0B数据搬运 |
| MTE2 | Memory Transfer Engine 2 | GM→L1/L0A/L0B/UB数据搬运 |
| MTE3 | Memory Transfer Engine 3 | UB→GM/L1数据搬运 |
| NPU | Neural-Network Processing Unit | 神经网络处理器单元 |
| OP | Operator | 算子 |
| SIMD | Single Instruction, Multiple Data | 单指令多数据 |
| SIMT | Single Instruction, Multiple Threads | 单指令多线程 |
| SPMD | Single-Program Multiple-Data | 单程序多数据并行模型 |
| UB | Unified Buffer | 统一缓冲区 |
| VL | Vector Length | RegTensor位宽（通常256Byte） |
| VF | Vector Function | 向量函数 |
| Ascend IR | Ascend Intermediate Representation | AI处理器专用中间表示 |
| DB | DoubleBuffer | 双缓冲 |
| LCM | Local Cache Memory | 临时共享UB空间 |
| L2 Cache | — | 二级缓存 |

## 四、编程范式核心术语

| 术语 | 含义 |
|------|------|
| CopyIn | 数据从GM搬运到Local Memory |
| Compute | 完成计算任务 |
| CopyOut | 计算结果从Local Memory搬运到GM |
| Pipe | 统一管理Device端内存资源的核心对象 |
| Kernel | 核函数，Device上执行的并行函数，__global__修饰 |
| Kernel Launch | 将kernel程序提交至硬件启动执行 |
| Tiling | 数据切分与分块 |
| TilingData | 切分相关参数（块大小、循环次数等） |
| TilingKey | 区分Kernel不同版本特例实现 |
| TilingFunc | Host侧计算Tiling的默认函数 |
| RegTensor | 矢量数据寄存器，Reg矢量计算基本单元 |
| MaskReg | 256位掩码寄存器 |
| AddrReg | 地址寄存器（存储地址偏移量） |
| Warp | 包含32个线程的线程集合 |
| Lane | Warp中的单个线程 |
| Dim3 | 定义三维线程结构的数据类型 |
| DataBlock | 矢量计算指令处理的数据单元，大小通常为32字节 |
| Repeat | 矢量计算指令执行一次迭代 |
| Repeat Stride | 下次Repeat与本次Repeat起始地址间的DataBlock个数 |
| Repeat Times | 矢量计算指令循环执行次数 |
| Mask | 控制矢量计算每次Repeat内参与计算的元素 |

## 五、数据排布格式

| 缩写 | 全称维度顺序 |
|------|-------------|
| NC1HWC0 | [N, C1, H, W, C0]，C0与硬件架构强相关 |
| NCHW | [Batch, Channels, Height, Width] |
| NHWC | [Batch, Height, Width, Channels] |
| ND | 普通N维张量格式 |

## 六、芯片产品名称规范

| 规范写法 | 错误写法 |
|----------|----------|
| Atlas 训练系列产品 | Atlas训练系列产品 |
| Atlas A2 训练系列产品 | Atlas A2训练系列产品 |
| Atlas A2 推理系列产品 | Atlas A2推理系列产品 |
| Atlas A3 训练系列产品 | Atlas A3训练系列产品 |
| Atlas A3 推理系列产品 | Atlas A3推理系列产品 |
| Atlas 推理系列产品 | Atlas推理系列产品 |
| Atlas 200I/500 A2 推理产品 | — |
| Ascend 950PR | ascend 950PR |
| Ascend 950DT | ascend 950DT |

## 七、数据通路写法

| 规范写法 | 错误写法 |
|----------|----------|
| VECOUT->GM | VECOUT → GM |
| UB->GM | UB → GM |
| L0C Buffer->GM | L0C->GM |

注：数据通路使用 `->` 表示方向，不使用 `→`。

## 八、其他术语

| 术语 | 含义 |
|------|------|
| Device | 安装昇腾AI处理器的硬件设备 |
| Host | 与Device连接的X86/ARM服务器 |
| Local Memory | AI Core内部存储，包括L1 Buffer、L0A/L0B/L0C Buffer、Unified Buffer等 |
| Global Memory/GM | 设备端主内存，AI Core外部存储 |
| GlobalTensor | 存放Global Memory全局数据的Tensor |
| LocalTensor | 存放AI Core中Local Memory本地数据的Tensor |
| Tensor | 张量，N维数据结构 |
| InferShape | 算子shape推导，仅在GE图模式时使用 |
| Fixpipe | 将矩阵计算结果从L0C Buffer搬运到GM或L1 Buffer的单元 |
| Workspace | 预分配的临时使用的Global Memory内存 |
| Preload | 预先将指令或数据加载到缓存中 |
| Reduce | 减维操作 |
| Broadcast | 广播，张量操作机制 |
| Elementwise | 元素级操作 |
| SuperKernel | 算子二进制融合技术 |
| Membase | 基于内存的架构 |
| Regbase | 基于寄存器的架构 |
