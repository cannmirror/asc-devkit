
# 资源管理类api样例介绍

## 概述

本路径下包含了与资源管理相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。

## 算子开发样例

|  目录名称                                                  |  功能描述                                             |
| ----------------------------------------------------------- | --------------------------------------------------- |
| [customized_tbuf_pool](./customized_tbuf_pool) | 本样例基于EXTERN_IMPL_BUFPOOL宏来辅助用户自定义TBufPool，适用于以下场景：开发者有自定义的内存块分配需求，比如不连续内存块、内存块在不同TQue之间共享等。 |
| [get_tpipe_ptr](./get_tpipe_ptr) | 样例基于GetTPipePtr获取全局TPipe指针，核函数无需显式传入TPipe指针，即可进行TPipe相关的操作。 |
| [init](./init) | 本样例基于Init实现内存和同步流水事件EventID的初始化。|
| [tbufpool_init_buf_pool](./tbufpool_init_buf_pool) | 本样例基于TBufPool::InitBufPool进行TBufPool资源划分，适用于将Tpipe::InitBufPool接口划分的整块资源，继续划分成小块资源的场景。 |
| [tpipe_init_buf_pool](./tpipe_init_buf_pool) | 本样例基于TPipe::InitBufPool初始化TBufPool内存资源池，适用于内存资源有限时，希望手动指定UB/L1内存资源复用的场景。本接口初始化后在整体内存资源中划分出一块子资源池。|