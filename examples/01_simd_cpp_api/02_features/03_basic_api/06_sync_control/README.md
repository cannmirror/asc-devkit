
# 同步控制类api样例介绍

## 概述

本路径下包含了与同步控制相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。

## 样例列表

| 目录名称                                                                   |  功能描述                                           |
|------------------------------------------------------------------------| ------------------------------------------------- |
| [ib_set_wait](./ib_set_wait)                                                     | 本样例基于IBSet和IBWait实现核间同步，适用于以下场景：当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。 |
| [sequential_block_sync](./sequential_block_sync) | 本样例基于InitDetermineComputeWorkspace、WaitPreBlock、NotifyNextBlock三个接口组合实现核间顺序同步，确保多核按blockIdx升序执行。 |
| [mutex](./mutex)                                                       | 本样例基于Mutex::Lock和Mutex::Unlock实现核内异步流水之间的同步，通过锁定指定流水再释放流水来实现流水的同步依赖。|
| [set_next_task_start](./set_next_task_start)                           | 本样例基于SetNextTaskStart接口实现Superkernel的子kernel并行。 |
| [sync_all](./sync_all)                                                 | 本样例基于SyncAll实现核间同步，适用于以下场景：不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题，通过调用本接口来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。|
| [wait_pre_task_end](./wait_pre_task_end)                                                 | 本样例基于SetPreTaskEnd接口实现Superkernel的子kernel并行。|
| [pipe_barrier](./pipe_barrier)                                                       | 本样例基于PipeBarrier实现核内同步，适用于以下场景：阻塞相同流水，具有数据依赖的相同流水之间需要插入此同步。|
| [data_sync_barrier](./data_sync_barrier)                                                       | 本样例介绍DataSyncBarrier的调用，适用于核内标量单元流水（PIPE_S）的同步。|
| [group_barrier](./group_barrier) | 本样例实现了两组存在依赖关系的AIV之间的正确同步，A组AIV计算完成后，B组AIV依赖该A组AIV的计算结果进行后续的计算，称A组为Arrive组，B组为Wait组。 |