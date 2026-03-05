# Matmul API样例介绍
## 概述
本样例集介绍了Matmul API不同特性的典型用法，给出了对应的端到端实现。

## 算子开发样例
| 目录名称                                                                                       |  功能描述                                              |
|--------------------------------------------------------------------------------------------| ---------------------------------------------------- |
| [fixpipe_co12gm_quantization_f322f16](./fixpipe_co12gm_quantization_f322f16)               | 本样例介绍如何使用基础API Fixpipe将矩阵乘的结果从CO1搬入GM，并使能随路量化将矩阵乘结果由float类型数据量化为half类型 |
| [fixpipe_co12gm_tensor_quantization_s322f16](./fixpipe_co12gm_tensor_quantization_s322f16) | 本样例介绍如何使用基础API Fixpipe将矩阵乘的结果从CO1搬入GM，并使能随路量化将矩阵乘结果由int32_t类型数据量化为half类型 |
| [fixpipe_co12c1_quantization_s322f16](./fixpipe_co12c1_quantization_s322f16)               | 本样例介绍如何使用基础API Fixpipe将矩阵乘的结果从CO1搬入C1，并使能随路量化将矩阵乘结果由int32_t类型数据量化为half类型 |
| [fixpipe_co12c1_quantization_f322s8](./fixpipe_co12c1_quantization_f322s8)                 | 本样例介绍如何使用组合API Fixpipe或基础API DataCopy将矩阵乘的结果从L0C搬出到L1，并支持随路quant, relu能力组合，与unitflag能力组合，输入half类型数据，输出由float类型量化为int8_t类型, int8_t输出默认开启channel merge能力。 |
| [fixpipe_co12c1_quantization_s322f16_v2](./fixpipe_co12c1_quantization_s322f16_v2)               | 本样例介绍如何使用组合API Fixpipe或基础API DataCopy将矩阵乘的结果从L0C搬出到L1，并支持随路quant, relu能力组合，与unitflag能力组合，输入int8_t类型数据，输出由int32_t类型量化为half类型。 |
| [fixpipe_co12gm_quantization_f322s8](./fixpipe_co12gm_quantization_f322s8)                 | 本样例介绍如何使用组合API Fixpipe或基础API DataCopy将矩阵乘的结果从L0C搬出到GM，并支持随路NZ2ND，unitflag与随路quant, relu能力组合，输入half类型数据，输出由float类型量化为int8_t类型 |
| [fixpipe_co12gm_quantization_s322f16](./fixpipe_co12gm_quantization_s322f16)               | 本样例介绍如何使用组合API Fixpipe或基础API DataCopy将矩阵乘的结果从L0C搬出到GM，并支持随路NZ2ND，unitflag与随路quant, relu能力组合，输入int8_t类型数据，输出由int32_t类型量化为half类型 |
| [fixpipe_co12c1_tensor_quantization_s322f16](./fixpipe_co12c1_tensor_quantization_s322f16) | 本样例介绍如何使用基础API Fixpipe将矩阵乘的结果从CO1搬入C1，并使能随路tensor量化将矩阵乘结果由int32_t类型数据量化为half类型 |
| [fixpipe_nz2dn_tensor_quantization_f322f16](./fixpipe_nz2dn_tensor_quantization_f322f16)   | 本样例介绍如何使用基础API的Fixpipe将矩阵乘的结果从CO1搬入GM，完成NZ2DN分形转换，并使能随路量化将矩阵乘结果由float类型数据量化为half类型 |
| [load_data](./load_data)                                                                   | 本样例介绍基于基础API LoadData实现A1至A2和B1至B2的数据搬运，其中A1至A2使用Load3D搬运，B1至B2使用Load2D搬运 |
| [load_data_with_transpose_b8](./load_data_with_transpose_b8)                               | 本样例介绍基础API LoadDataWithTranspose b8数据类型下的使用 |
| [load_data_with_transpose_b16](./load_data_with_transpose_b16)                             | 本样例介绍基础API LoadDataWithTranspose b16数据类型下的使用 |
| [load_data_with_transpose_b32](./load_data_with_transpose_b32)                             | 本样例介绍基础API LoadDataWithTranspose b32数据类型下的使用 |
| [batch_matmul](./batch_matmul)                                                             | 本样例介绍了一次完成BatchNum个Matmul矩阵乘法 |
| [batch_matmul_bias_no_batch](./batch_matmul_bias_no_batch)                                 | 本样例介绍调用Matmul高阶API实现BatchMatmul单算子复用Bias矩阵的场景|
| [matmul](./matmul)                                                                         | 本样例介绍调用Matmul API实现matmul单算子 |
| [matmul_splitm](./matmul_splitm)                                                           | 本样例介绍Matmul API使用SplitM模板策略的场景 |
| [matmul_splitk](./matmul_splitk)                                                           | 本样例介绍调用Matmul高阶API实现多核切K场景下的单算子。多核切K的应用场景为矩阵乘的M、N较小，不能在M、N方向开启多核，需要切K满足将切分后的矩阵分配到更多核上并行处理。|
| [matmul_vecout](./matmul_vecout)                                                           | 本样例介绍Matmul API输入矩阵为VECOUT的场景 |
| [matmul_bias_bf16_tscm](./matmul_bias_bf16_tscm)                                           | 本样例介绍Matmu API Bias输入为TSCM bfloat16类型的场景 |
| [matmul_tscm_mdl_setorgshape](./matmul_tscm_mdl_setorgshape)                               | 本样例介绍Matmul API MDL模板 A矩阵输入为TSCM的场景|
| [matmul_perf](./matmul_perf)                                                               | 本样例介绍Matmul API实现三种性能优化特性（纯Cube模式、MDL模板、UnitFlag）的单算子|
| [matmul_triangle](./matmul_triangle)                                                       | 本样例通过使用Matmul模板参数MatmulPolicy中TrianUpperMatmulPolicy（上三角模板策略）和TrianLowerMatmulPolicy（下三角模板策略），实现了上下三角矩阵计算的单算子 |
| [matmul_partial_output](./matmul_partial_output)                                           | 本样例介绍Matmul高阶API实现开启Partial Output功能的单算子 |
| [matmul_unitflag](./matmul_unitflag)                                                       | 本样例介绍Matmul API实现MDL模板开启UnitFlag功能的单算子 |
| [matmul_unaligned](./matmul_unaligned)                                                     | 本样例介绍Matmul高阶API实现多核非对齐切分的单算子 |
| [matmul_channelsplit](./matmul_channelsplit)                                               | 本样例介绍Matmul API实现矩阵乘输出Channel拆分功能的单算子 |
| [matmul_quant](./matmul_quant)                                                             | 本样例介绍Matmul API实现int8类型输入、half类型输出的Matmul反量化场景的算子，支持同一系数的反量化模式和向量的反量化模式 |
| [matmul_int4](./matmul_int4)                                                               | 本样例介绍Matmul API实现int4数据类型输入，int32数据类型输出的单算子 |
| [matmul_iterate_n_batch](./matmul_iterate_n_batch)                                         | 本样例介绍调用Matmul高阶API实现NBatchMatmul单算子，算子实现nNum次批量处理Matmul计算 |
| [matmul_tscm_src_vecout](./matmul_tscm_src_vecout)                                         | 本样例介绍Matmul API使用数据来源为VECOUT的用户自定义TSCM的输入 |
| [matmul_nz](./matmul_nz)                                                                   | 本样例介绍Matmul API输入矩阵内轴非256B对齐的场景下，在AIV核上使用DataCopyPad实现ND转换NZ格式的单算子 |
| [matmul_nbuffer33](./matmul_nbuffer33)                                                     | 本样例介绍Matmul API实现NBuffer33算法的单算子，以及介绍模板参数MatmulPolicy中NBuffer33MatmulPolicy的使用方式 |
| [matmul_k_reorder_load](./matmul_k_reorder_load)                                           | 本样例介绍Matmul API使能K轴错峰加载数据的场景 |
| [matmul_async_iterate](./matmul_async_iterate)                                             | 本样例介绍调用Matmul API实现异步场景下的Matmul矩阵乘法，实现方式为调用Iterate和GetTensorC输出到VECIN |
| [matmul_async_iterate_all](./matmul_async_iterate_all)                                     | 本样例介绍调用Matmul API实现异步场景下的Matmul矩阵乘法，实现方式为调用IterateAll输出到GM |
| [matmul_b8](./matmul_b8)                                                                   | 本样例介绍了调用Matmul高阶API实现A、B矩阵为hifloat8、fp8_e4m3fn、fp8_e5m2数据类型输入，并使用MDL模板的Matmul单算子 |
| [matmul_a2b2_share](./matmul_a2b2_share)                                                   | 本样例介绍调用Matmul API实现开启A2和B2全局管理的单算子 |
| [matmul_callback](./matmul_callback)                                                       | 本样例介绍Matmul API模板参数MatmulCallbackFunc的自定义使用方式 |
| [matmul_l0c_extend](./matmul_l0c_extend)                                                   | 本样例介绍Matmul API用户自主管理CO1的Iterate接口的自定义使用方式 |
| [matmul_l2cache](./matmul_l2cache)                                                         | 本样例介绍调用Matmul API实现L2 Cache切分的功能的Matmul单算子 |
| [matmul_mixdualmaster](./matmul_mixdualmaster)                                             | 本样例介绍通过配置模板参数中enableMixDualMaster使能Matmul双主模式MixDualMaster的使用方式 |
| [matmul_nd_align](./matmul_nd_align)                                                       | 本样例介绍调用Matmul API在输入矩阵的N方向非对齐场景下，矩阵乘输出时使能N方向对齐的实现方式 |
| [matmul_sparse](./matmul_sparse)                                                           | 本样例介绍使用Matmul API实现左矩阵A为稀疏矩阵，右矩阵B为4:2稠密化后的矩阵的Sparse Matmul场景的矩阵乘计算 |
| [matmul_l0cache](./matmul_l0cache)                                                         | 本样例介绍Matmul API中使能L0缓存的使用方式 |
| [matmul_tscm](./matmul_tscm)                                                               | 本样例介绍Matmul API中用户自定义TSCM输入的使用方式 |
| [batch_matmul_tscm](./batch_matmul_tscm)                                                   | 本样例介绍了调用Matmul高阶API实现左矩阵A为TSCM输入进行BatchMatmul计算的单算子 |
| [batch_mmad](./batch_mmad)                                                                 | 本样例介绍在输入为float数据类型并且左、右矩阵均不转置的场景下，带batch的矩阵乘法，其中从GM-->L1、L0C-->GM、L0C-->L1这三条通路分别采用了DataCopy ND2NZ和Fixpipe批量搬运数据，从L1-->L0A/L0B以及Mmad执行矩阵乘这两个步骤则是循环batch次，每次循环内只处理一对左、右矩阵 |
| [matmul_gemv](./matmul_gemv)                                                               | 本样例介绍调用Matmul NORM模板实现矩阵向量乘的单算子 |
| [matmul_mndb](./matmul_mndb)                                                               | 本样例介绍调用Matmul API实现M或N轴方向流水并行的单算子 |
| [matmul_mx_norm_even](./matmul_mx_norm_even)                                               | 本样例介绍了在Mx数据格式下，Scale的K方向为偶数的带有量化系数的矩阵乘法，即MxMatmul计算场景 |
| [matmul_mx_norm_odd](./matmul_mx_norm_odd)                                                 | 本样例介绍了在Mx数据格式下，Scale的K方向为奇数的带有量化系数的矩阵乘法，即MxMatmul计算场景 |
| [matmul_mx_scalea_trans](./matmul_mx_scalea_trans)                                         | 本样例介绍了在Mx数据格式下，scaleA开启转置、scaleB不开启转置的带有量化系数的矩阵乘法，即MxMatmul计算场景 |
| [matmul_mx_scaleb_trans](./matmul_mx_scaleb_trans)                                         | 本样例介绍了在Mx数据格式下，scaleA不开启转置、scaleB开启转置的带有量化系数的矩阵乘法，即MxMatmul计算场景 |
| [matmul_mx_typepara](./matmul_mx_typepara)                                                 | 本样例介绍了在Mx数据格式下，左量化系数矩阵scaleA载入L1时，scaleA的K方向上开启多倍缓存，从而实现带有量化系数的矩阵乘法，即MxMatmul计算场景。 |
| [matmul_mx_ub_tscm_nz](./matmul_mx_ub_tscm_nz)                                             | 本样例介绍了在Mx数据格式下，A、B矩阵内存逻辑位置使用VECOUT，scaleA、scaleB矩阵内存逻辑位置使用TSCM，4个输入矩阵都是NZ格式的带有量化系数的矩阵乘法 |
| [matmul_preload](./matmul_preload)                                                         | 本样例介绍调用Matmul MDL模板实现使能M或N方向预加载功能的单算子 |
| [matmul_constant](./matmul_constant)                                                       | 本样例介绍调用Matmul MDL模板使能Tiling常量化的单算子 |
| [matmul_column_major](./matmul_column_major)                                               | 本样例介绍A、B、C矩阵为COLUMN_MAJOR格式排布的矩阵乘的单算子 |
| [mmad](./mmad)                                                                             | 本样例介绍基于基础API Mmad实现矩阵乘 |
| [mmad_load3dv2](./mmad_load3dv2)                                                           | 本样例介绍LoadData3DV2指令将A、B矩阵从L1搬运到L0A/L0B的过程，其中 A 和 B 分别表示矩阵乘法的左右输入矩阵。LoadData3DV2指令参数配置及执行指令前后各个矩阵数据排布变化，均配合示意图进行了说明 |
| [mmad_s8_f16_f32_with_A_B_transpose_option](./mmad_s8_f16_f32_with_A_B_transpose_option)                                                           | 本样例介绍了在 int8_t / half / float 三种数据类型下，以及左、右矩阵均不转置 / 左矩阵不转置、右矩阵转置 / 左矩阵转置、右矩阵不转置 / 左、右矩阵均转置 共 12 种矩阵乘法场景中，相关指令的使用方法，其中 A 和 B 分别表示矩阵乘法的左右输入矩阵。|
| [mmad_unitflag](./mmad_unitflag)                                                                             | 本样例介绍是否使能unitFlag对于Mmad指令执行矩阵乘法性能的影响。 |
| [mmad_with_bias](./mmad_with_bias)                                                         | 本样例介绍基于基础API Mmad实现带Bias的矩阵乘 |
| [mmad_with_sparse](./mmad_with_sparse)                                                     | 本样例介绍基础API MmadWithSparse调用 |
| [bare_mix](./bare_mix)                                                                     | 本样例介绍分核计算实现的CV融合算子bare_mix |
| [matmul_ibshareAB](./matmul_ibshareAB)                                                     | 本样例介绍了调用Matmul高阶API实现开启IBShare功能的单算子。IBShare的功能是复用L1 Buffer上相同的A矩阵或者B矩阵数据，减少数据搬运开销。本样例为A矩阵和B矩阵同时复用场景 |
| [matmul_leaky_relu_async](./matmul_leaky_relu_async)                                       | 本样例介绍MatmulLeakyRelu算子实现及核函数直调方法 |
| [matmul_no_ibshareAB](./matmul_no_ibshareAB)                                               | 本样例介绍了调用Matmul高阶API实现不开启IBShare功能的单算子。IBShare的功能是复用L1 Buffer上相同的A矩阵或者B矩阵数据，减少数据搬运开销。本样例为A矩阵和B矩阵同时不复用场景 |
| [matmul_ibshareB](./matmul_ibshareB)                                                       | 本样例介绍多个AIV的B矩阵GM地址相同场景下，实现共享L1 Buffer上B矩阵数据的Matmul矩阵乘法，计算公式为：C = A * B + Bias |
| [basic_block_matmul](./basic_block_matmul)                                                 | 本样例实现无尾块且tiling的base块大小固定的场景下的Matmul矩阵乘法，计算公式为：C = A * B + Bias |
