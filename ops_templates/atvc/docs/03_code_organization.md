# ATVC代码组织结构
这篇文档描述了昇腾算子vector算子模板库的代码仓结构，主要包含的内容如下：
```
include/
├── atvc.h                                      // Vector模板编程入口头文件
├── common                                      // 不同模板公用API和C++基本类的拓展模板目录
├── elewise                                     // Elementwise模板目录
│   ├── common                                  // Elementwise的公共数据定义 
│   ├── utils                                   // Elementwise模板辅助工具目录
│   ├── elewise_op_template.h                   // Elementwise算子模板类
│   └── elewise_host.h                          // Elementwise算子host侧API
│   └── elewise_device.h                        // Elementwise算子device头文件集合
├── broadcast                                   // Broadcast模板目录
│   ├── common                                  // Broadcast模板各层公用文件目录
│   ├── utils                                   // Broadcast模板辅助工具目录
│   ├── tiling                                  // Broadcast模板host层目录
│   ├── broadcast_host.h                        // Broadcast算子host侧API
│   ├── broadcast_op_template.h                 // Broadcast算子模板类
│   └── broadcast_compute.h                     // Broadcast计算模板
│   └── broadcast_device.h                      // Broadcast算子device头文件集合
└── reduce                                      // Reduce模板目录
    ├── common                                  // Reduce模板各层公用文件目录
    ├── utils                                   // Reduce模板辅助工具目录
    ├── tiling                                  // Reduce模板host层目录
    ├── reduce_host.h                           // Reduce算子host侧API
    ├── reduce_op_template.h                    // Reduce算子模板类
    ├── reduce_sum.h                            // ReduceSum计算模板
    └── reduce_device.h                         // Reduce算子device头文件集合
