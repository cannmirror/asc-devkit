# C API
C API文档目录，整体使用时可以引入asc_simd.h，C API列表如下：

## 数据结构

|结构名|说明|
|-----------------------|-----------------------|
| [asc_load3d_v2_config](struct/asc_load3d_v2_config.md) | Load3Dv2接口的repeat参数 |
| [asc_store_atomic_config](struct/asc_store_atomic_config.md) | 原子操作使能位与原子操作类型的值 |

## 矢量计算

矢量计算类API，单独使用时可以引入vector_compute.h，此类API列表如下：

|   API名称   |   说明   |
|----------|-----------|
| [asc_get_cmp_mask](vector_compute/asc_get_cmp_mask.md) | 获取Compare操作的比较结果。 |
| [asc_set_cmp_mask](vector_compute/asc_set_cmp_mask.md) | 为Select操作设置用于选择的掩码。 |
| [asc_get_rsvd_count](vector_compute/asc_get_rsvd_count.md) | 获取GatherMask操作后剩余的元素数量。 |
| [asc_set_mask_count](vector_compute/asc_set_mask_count.md) | 设置Mask模式为Counter模式。 |
| [asc_set_mask_norm](vector_compute/asc_set_mask_norm.md) | 设置Mask模式为Normal模式。 |
| [asc_set_vector_mask](vector_compute/asc_set_vector_mask.md) | 设置Mask值。 |
| [asc_add](vector_compute/asc_add.md) | 按元素求和。 |
| [asc_add_scalar](vector_compute/asc_add_scalar.md) | 矢量内每个元素与标量求和。 |
| [asc_sub](vector_compute/asc_sub.md) | 按元素求差。 |
| [asc_sub_scalar](vector_compute/asc_sub_scalar.md) | 矢量内每个元素与标量求差。 |
| [asc_mul](vector_compute/asc_mul.md) | 按元素求积。 |
| [asc_mul_scalar](vector_compute/asc_mul_scalar.md) | 矢量内每个元素与标量求积。 |
| [asc_div](vector_compute/asc_div.md) | 按元素求商。 |
| [asc_exp](vector_compute/asc_exp.md) | 按元素取自然指数。 |
| [asc_relu](vector_compute/asc_relu.md) | 按元素做线性整流Relu。 |
| [asc_max](vector_compute/asc_max.md) | 按元素求最大值。 |
| [asc_max_scalar](vector_compute/asc_max_scalar.md) | 矢量内每个元素与标量求最大值。 |
| [asc_min](vector_compute/asc_min.md) | 按元素求最小值。 |
| [asc_datablock_reduce_sum](vector_compute/asc_datablock_reduce_sum.md) | 对每个DataBlock内所有元素求和。 |
| [asc_datablock_reduce_max](vector_compute/asc_datablock_reduce_max.md) | 对每个DataBlock内所有元素求最大值。 |
| [asc_datablock_reduce_min](vector_compute/asc_datablock_reduce_min.md) | 对每个DataBlock内所有元素求最小值。 |
| [asc_repeat_reduce_sum](vector_compute/asc_repeat_reduce_sum.md) | 对每个Repeat内所有元素求和。 |
| [asc_repeat_reduce_max](vector_compute/asc_repeat_reduce_max.md) | 对每个Repeat内所有元素求最大值。 |
| [asc_repeat_reduce_min](vector_compute/asc_repeat_reduce_min.md) | 对每个Repeat内所有元素求最小值。 |
| [asc_get_reduce_max_cnt](vector_compute/asc_get_reduce_max_cnt.md) | 获取执行asc_repeat_reduce_max操作后的最大值，以及第一个最大值时的索引。 |
| [asc_get_reduce_min_cnt](vector_compute/asc_get_reduce_min_cnt.md) | 获取执行asc_repeat_reduce_min操作后的最小值，以及第一个最小值时的索引。 |
| [asc_brcb](vector_compute/asc_brcb.md) | 将源操作数中的每一个数填充到目的操作数的一个DataBlock中。 |
| [asc_duplicate](vector_compute/asc_duplicate.md) | 将一个变量或立即数填充到一个矢量中。 |
| [asc_select](vector_compute/asc_select.md) | 根据掩码，从两个源操作数中选取元素，输出到目的操作数。 |
| [asc_bfloat162float](vector_compute/asc_bfloat162float.md) | 数据类型转换。将bfloat16_t类型的数据转换为float类型。 |
| [asc_bfloat162int32](vector_compute/asc_bfloat162int32.md) | 数据类型转换。将bfloat16_t类型的数据转换为int32_t类型。 |
| [asc_float2bfloat16](vector_compute/asc_float2bf16.md) | 数据类型转换。将float类型的数据转换为bfloat16_t类型。 |
| [asc_float2float](vector_compute/asc_float2float.md) | 数据类型转换。将float类型的数据转换为float类型。 |
| [asc_float2half](vector_compute/asc_float2half.md) | 数据类型转换。将float类型的数据转换为half类型。 |
| [asc_half2float](vector_compute/asc_half2float.md) | 数据类型转换。将half类型的数据转换为float类型。 |
| [asc_half2int4](vector_compute/asc_half2int4.md) | 数据类型转换。将half类型的数据转换为int4b_t类型。 |
| [asc_half2int8](vector_compute/asc_half2int8.md) | 数据类型转换。将half类型的数据转换为int8_t类型。 |
| [asc_half2int16](vector_compute/asc_half2int16.md) | 数据类型转换。将half类型的数据转换为int16_t类型。 |
| [asc_half2int32](vector_compute/asc_half2int32.md) | 数据类型转换。将half类型的数据转换为int32_t类型。 |
| [asc_int42half](vector_compute/asc_int42half.md) | 数据类型转换。将int4b_t类型的数据转换为half类型。 |
| [asc_int82half](vector_compute/asc_int82half.md) | 数据类型转换。将int8_t类型的数据转换为half类型。 |
| [asc_uint82half](vector_compute/asc_uint82half.md) | 数据类型转换。将uint8_t类型的数据转换为half类型。 |
| [asc_int162float](vector_compute/asc_int162float.md) | 数据类型转换。将int16_t类型的数据转换为float类型。 |
| [asc_int322int16](vector_compute/asc_int322int16.md) | 数据类型转换。将int32_t类型的数据转换为int16_t类型。 |
| [asc_int322int64](vector_compute/asc_int322int64.md) | 数据类型转换。将int32_t类型的数据转换为int64_t类型。 |
| [asc_int642int32](vector_compute/asc_int642int32.md) | 数据类型转换。将int64_t类型的数据转换为int32_t类型。 |
| [asc_deq_int162b8](vector_compute/asc_deq_int162b8.md) | 将int16_t类型转换为int8_t或uint8_t类型，并将数据存放在每个DataBlock的上半块或下半块。 |
| [asc_set_deq_scale](vector_compute/asc_set_deq_scale.md) | 设置DEQSCALAR寄存器的值。 |
| [asc_eq](vector_compute/asc_eq.md) | 比较src0与src1在对应索引位置的元素大小。若比较结果为真，则输出结果的对应比特位设为1，否则设为0。 |
| [asc_transpose](vector_compute/asc_transpose.md) | 用于实现16*16的二维矩阵数据块转置。 |
| [asc_sqrt](vector_compute/asc_sqrt.md) | 对元素进行开方。 |
| [asc_vaxpy](vector_compute/asc_axpy.md) | 源操作数中每个元素与标量求积后和目的操作数中的对应元素相加。 |
| [asc_lt](vector_compute/asc_lt.md) | 按元素判断src0 < src1是否成立，若成立则输出结果上的对应比特位为1，否则为0。 |
| [asc_ne_scalar](vector_compute/asc_ne_scalar.md) | 按元素判断是否不等于输入标量，若成立则输出结果上的对应比特位为1，否则为0。 |
| [asc_gather_datablock](vector_compute/asc_gather_datablock.md) | 根据偏移地址按照DataBlock的粒度将源操作数收集到目的操作数中。 |
| [asc_int162half](vector_compute/asc_int162half.md) | 数据类型转换。将int16_t类型的数据转换为half类型。 |
| [asc_rcp](vector_compute/asc_rcp.md) | 执行矢量的取倒数运算。 |
| [asc_shiftright](vector_compute/asc_shiftright.md) | 对源操作数中的每个元素执行右移。 |
| [asc_mul_add](vector_compute/asc_mul_add.md) | 执行矢量的乘加运算。 |
| [asc_eq_scalar](vector_compute/asc_eq_scalar.md) | 执行矢量与标量的比较运算，如果值相等则输出1，否则输出0。 |
| [asc_gather](vector_compute/asc_gather.md) | 将源操作数按照给定的偏移按元素收集到目的操作数中。 |
| [asc_min_scalar](vector_compute/asc_min_scalar.md) | 源操作数矢量逐元素与标量相比，取较小值。 |
| [asc_gt](vector_compute/asc_gt.md) | 按元素比较两个矢量的大小关系，若比较后的结果为真，则输出结果的对应比特位为1，否则为0。 |
| [asc_vdeq_int162b8](vector_compute/asc_vdeq_int162b8.md) | 将int16_t类型转化为int8_t或uint8_t类型，并将数据存放在每个DataBlock的上半块或下半块。 |
| [asc_int322float](vector_compute/asc_int322float.md) | 将int32_t类型数据转换为float类型。 |
| [asc_abs](vector_compute/asc_abs.md) | 按元素取绝对值 |
| [asc_add_relu](vector_compute/asc_add_relu.md) | 按元素求和，再进行Relu计算（结果和0对比取较大值），并提供转换最终结果的数据类型的功能(s162s8、f322f16、f162s8)。 |
| [asc_and](vector_compute/asc_and.md) | 执行矢量与运算。 |
| [asc_axpy](vector_compute/asc_axpy.md) | 源操作数src中每个元素与标量value求积后和目的操作数dst中的对应元素相加 |
| [asc_bitsort](vector_compute/asc_bitsort.md) | Score和Index分别存储在src0和src1中，按Score进行排序（Score大的元素排前面），排序后的Score与其对应的Index一起以（Score，Index）的结构存储在dst中。 |
| [asc_deq_int322half](vector_compute/asc_deq_int322half.md) | 对输入的int32_t类型的数据按元素做量化并转换为half类型 |
| [asc_float2int16](vector_compute/asc_float2int16.md) | 将float类型数据转换为int16_t类型。 |
| [asc_float2int32](vector_compute/asc_float2int32.md) | 将float类型数据转换为int16_t类型 |
| [asc_float2int64](vector_compute/asc_float2int64.md) | 将float类型数据转换为int16_t类型 |
| [asc_ge](vector_compute/asc_ge.md) | Ge（greater than or equal to），逐元素比较src0 >= src1是否成立，成立则输出结果为1，否则输出结果为0，每个元素的比较结果占一个bit。 |
| [asc_ge_scalar](vector_compute/asc_ge_scalar.md) | 按元素判断src >= value是否成立，若成立则输出结果为1，否则为0。 |
| [asc_gt_scalar](vector_compute/asc_gt_scalar.md) | src中的每个元素逐个与标量value比较大小，如果某个位置上的元素大于value，则输出结果dst上的对应比特位为1，否则为0。 |
| [asc_half2uint8](vector_compute/asc_half2uint8.md) | 将half类型数据转换为uint8_t类型，支持多种舍入模式。 |
| [asc_int642float](vector_compute/asc_int642float.md) | 将int64_t类型数据转换为float类型。 |
| [asc_le](vector_compute/asc_le.md) | 按元素判断src0 <= src1是否成立，若成立则输出结果为1，否则为0。 |
| [asc_leakyrelu](vector_compute/asc_leakyrelu.md) | 执行矢量Leaky Relu运算。 |
| [asc_le_scalar](vector_compute/asc_le_scalar.md) | 按元素判断src <= value是否成立，若成立则输出结果为1，否则为0。 |
| [asc_log](vector_compute/asc_log.md) | 按元素取自然对数。 |
| [asc_lt_scalar](vector_compute/asc_lt_scalar.md) | 执行矢量中每个位置和标量比较，如果值小于标量值则为1，否则为0，结果为每个bit位按小端序排布 |
| [asc_mrgsort4](vector_compute/asc_mrgsort4.md) | 将已经排好序的最多4条队列，合并排列成1条队列，结果按照score域由大到小排序。 |
| [asc_get_vms4_sr](vector_compute/asc_get_vms4_sr.md) | 获取执行asc_mrgsort4操作后的队列中，每个队列已经理过的Region Proposal个数 |
| [asc_fma](vector_compute/asc_fma.md) | 按元素将src0和src1相乘并和dst相加，将最终结果存放进dst中。 |
| [asc_mul_add_relu](vector_compute/asc_mul_add_relu.md) | 按元素将src0和dst相乘并加上src1，再进行Relu计算（结果和0对比取较大值），最终结果存放进dst中。 |
| [asc_mul_cast_half2int8](vector_compute/asc_mul_cast_half2int8.md) | 按元素求积，并将结果转换为int8_t类型 |
| [asc_mul_cast_half2uint8](vector_compute/asc_mul_cast_half2uint8.md) | 按元素求积，并将结果转换为uint8_t类型。 |
| [asc_ne](vector_compute/asc_ne.md) | 按元素判断src0 != src1是否成立，若成立则输出结果为1，否则为0。 |
| [asc_not](vector_compute/asc_not.md) | 按元素做按位取反，计算公式如下。 |
| [asc_or](vector_compute/asc_or.md) | 每对元素按位或运算。 |
| [asc_reduce](vector_compute/asc_reduce.md) | Reduce（归约）是将一组数据通过指定运算聚合（聚合指将将一组分散、多个的数据通过某种规则合并成一个或少数几个的操作）为单个、少量结果的核心操作，包括ReduceMax、ReduceMin和ReduceSum等。 |
| [asc_rsqrt](vector_compute/asc_rsqrt.md) | 按元素进行开方后取倒数的计算。 |
| [asc_set_va_reg](vector_compute/asc_set_va_reg.md) | 用于设置transpose的地址，将操作数地址序列与地址寄存器关联。 |
| [asc_shiftleft](vector_compute/asc_shiftleft.md) | 将所有元素左移distance位。 |
| [asc_sub_relu](vector_compute/asc_sub_relu.md) | 按元素求差，再进行Relu计算（结果和0对比取较大值），并提供转换最终结果的数据类型的功能(s162s8、f322f16、f162s8)。 |
| [asc_transto5hd](vector_compute/asc_transto5hd.md) | 数据格式转换，一般用于将NCHW格式转换成NC1HWC0格式。 |



## 数据搬运

数据搬运类API，单独使用时可以引入vector_datamove.h和cube_datamove.h，此类API列表如下：

|   API名称   |   说明   |
|----------|-----------|
| [asc_copy_l0c2gm](cube_datamove/asc_copy_l0c2gm.md) | 将L0C中的数据搬运到GM中。 |
| [asc_set_l13d_rpt](cube_datamove/asc_set_l13d_rpt.md) | 用于设置Load3Dv2接口的repeat参数。 |
| [asc_fill_l0a](cube_datamove/asc_fill_l0a.md) | 将L0A Buffer的Local Memory初始化为某一具体数值。 |
| [asc_fill_l0b](cube_datamove/asc_fill_l0b.md) | 将L0B Buffer的Local Memory初始化为某一具体数值。 |
| [asc_fill_l1](cube_datamove/asc_fill_l1.md) | 将L1 Buffer的Local Memory初始化为某一具体数值。 |
| [asc_set_l13d_size](cube_datamove/asc_set_l13d_size.md) | 设置[asc_copy_l12l0a](cube_datamove/asc_copy_l12l0a.md)/[asc_copy_l12l0b](cube_datamove/asc_copy_l12l0b.md)的3D格式搬运接口在L1 Buffer的边界值。 |
| [asc_load_image_to_cbuf](cube_datamove/asc_load_image_to_cbuf.md) | 将图像数据从Global Memory搬运到L1 Buffer。 |
| [asc_copy_l12bt](cube_datamove/asc_copy_l12bt.md) | 将数据从L1 Buffer搬运到BiasTable Buffer中，BiasTable Buffer用于存放矩阵计算中的Bias。 |
| [asc_copy_l12fb](cube_datamove/asc_copy_l12fb.md) | 将数据从L1 Buffer搬运到Fixpipe Buffer中，Fixpipe Buffer用于存放量化参数。 |
| [asc_copy_l12l0a](cube_datamove/asc_copy_l12l0a.md) | 用于搬运存放在L1 Buffer里的512B大小的矩阵到L0A Buffer里。 |
| [asc_copy_l12l0b](cube_datamove/asc_copy_l12l0b.md) | 用于搬运存放在L1 Buffer里的512B大小的矩阵到l0b Buffer里。 |
| [asc_copy_l12l0b_sparse](cube_datamove/asc_copy_l12l0b_sparse.md) | 用于搬运存放在L1 Buffer里的512B大小的稠密权重矩阵到L0B Buffer里，同时读取128B大小的索引矩阵用于稠密矩阵的稀疏化。 |
| [asc_copy_l12l0b_trans](cube_datamove/asc_copy_l12l0b_trans.md) | 该接口实现带转置的2D格式数据从L1 Buffer到L0B Buffer的加载。 |
| [asc_set_l0c_copy_params](cube_datamove/asc_set_l0c_copy_params.md) | DataCopy（CO1->GM、CO1->A1）过程中进行随路格式转换（NZ格式转换为ND格式）时，通过调用该接口设置格式转换的相关配置。 |
| [asc_set_l0c_copy_prequant](cube_datamove/asc_set_l0c_copy_prequant.md) | 数据搬运过程中进行随路量化时，通过调用该接口设置量化流程中的标量量化参数。 |
| [copy_gm2l1](cube_datamove/copy_gm2l1.md) | 将数据从Global Memory (GM) 搬运到 Level 1 cache (L1)。 |
| [copy_gm2l1_nd2nz](cube_datamove/copy_gm2l1_nd2nz.md) | 将数据从Global Memory (GM) 搬运到 Level 1 cache (L1)，支持在数据搬运时进行ND格式到NZ格式的转换。 |
| [asc_set_l13d_padding](cube_datamove/asc_set_l13d_padding.md)| 设置Pad属性描述，用于在调用asc_copy_l12l0a接口时配置填充数值。 |
| [asc_copy_gm2ub](vector_datamove/asc_copy_gm2ub.md) | 将数据从Global Memory (GM) 搬运到 Unified Buffer (UB)。 |
| [asc_copy_gm2ub_align](vector_datamove/asc_copy_gm2ub_align.md) | 提供数据非对齐搬运的功能，将数据从Global Memory (GM) 搬运到 Unified Buffer (UB)，并支持8位/16位/32位数据类型搬运。 |
| [asc_copy_ub2gm](vector_datamove/asc_copy_ub2gm.md) | 将数据从Unified Buffer (UB) 搬运到 Global Memory (GM)。 |
| [asc_copy_ub2gm_align](vector_datamove/asc_copy_ub2gm_align.md) | 将数据从Unified Buffer (UB) 搬运到 Global Memory (GM)，支持8位/16位/32位分块拷贝操作。 |


## 维测接口
|   API名称   |   说明   |
|----------|-----------|
| [assert](debug/assert.md) | 在算子Kernel侧实现代码中需要增加断言的地方使用assert检查代码，并格式化输出一些调测信息。 |
| [dump](debug/dump.md) | 将对应内存上的数据打印出来,同时支持打印自定义的附加信息（仅支持uint32_t类型的信息）。 |
| [printf](debug/printf.md) | 该接口提供NPU域调试场景下的格式化输出功能。 |

## 标量操作

标量操作类API，单独使用时可以引入scalar_compute.h，此类API列表如下：

|   API名称   |   说明   |
|----------|-----------|
| [asc_clz](scalar_compute/asc_clz.md)| 计算参数前导零的数量（二进制从最高位到第一个1共有多少个0）。 |
| [asc_set_nthbit](scalar_compute/asc_set_nthbit.md)| 计算一个uint64_t类型数字的指定二进制位置为1，其余位保持不变。 |
| [asc_sflbits](scalar_compute/asc_sflbits.md)| 计算一个int64_t类型数字的二进制中，从最高数值位开始与符号位相同的连续比特位的个数。 |
| [asc_clear_nthbit](scalar_compute/asc_clear_nthbit.md)| 位操作函数，用于将一个uint64_t整数bits的第idx位设置为0。 |
| [asc_ffs](scalar_compute/asc_ffs.md) | FindFirstSet接口，输入数据的二进制表示中从最低位向最高位查找第一个值为1的位，并返回其位置，如果没找到则返回-1。 |
| [asc_ffz](scalar_compute/asc_ffz.md) | 获取一个uint64_t类型数字的二进制表示中从最低有效位开始的第一个0出现的位置，如果没找到则返回-1。 |
| [asc_popc](scalar_compute/asc_popc.md) | 获取一个uint64_t类型数字的二进制中1的个数。 |
| [asc_zero_bits_cnt](scalar_compute/asc_zero_bits_cnt.md) | 获取一个uint64_t类型数字的二进制中0的个数。 |

## 矩阵计算

标量操作类API，单独使用时可以引入cube_compute.h，此类API列表如下：

|   API名称   |   说明   |
|----------|-----------|
| [asc_set_mmad_direction_m](cube_compute/asc_set_mmad_direction_m.md)| 设置mmad计算时优先通过M/N中的N方向，然后通过M方向产生结果，M为矩阵的行，N为矩阵的列。 |
| [asc_set_mmad_direction_n](cube_compute/asc_set_mmad_direction_n.md)| 设置mmad计算时优先通过M/N中的M方向，然后通过N方向产生结果，M为矩阵的行，N为矩阵的列。 |
| [asc_enable_hf32_trans](cube_compute/asc_enable_hf32_trans.md)| 设置HF32模式取整方式，需要先使用asc_enable_hf32开启HF32取整模式。 |
| [asc_mmad](cube_compute/asc_mmad.md) | 完成矩阵乘加操作。 |
| [asc_mmad_sparse](cube_compute/asc_mmad_sparse.md) | 完成矩阵乘加操作，传入的左矩阵A为稀疏矩阵，右矩阵B为稠密矩阵。 |
| [asc_set_fp32_mode](cube_compute/asc_set_fp32_mode.md) | 无 |
| [asc_enable_hf32](cube_compute/asc_enable_hf32.md) | 用于设置Mmad计算开启HF32模式，开启该模式后L0A Buffer/L0B Buffer中的FP32数据将在参与Mmad计算之前被舍入为HF32。 |


## 同步控制

同步控制类API，单独使用时可以引入sync.h，此类API列表如下：

|   API名称   |   说明   |
|----------|-----------|
| [asc_sync_notify](sync/asc_sync_notify.md)| 设置同步标志。 |
| [asc_sync_wait](sync/asc_sync_wait.md)| 等待同步标志。 |
| [asc_sync_pipe](sync/asc_sync_pipe.md)| 等待指定流水线操作完成。 |
| [asc_sync](sync/asc_sync.md)| 等待所有流水线操作完成。 |
| [asc_sync_vec](sync/asc_sync_vec.md)| 同步所有流水线。 |
| [asc_sync_mte3](sync/asc_sync_mte3.md)| 等待PIPE_MTE3流水完成。 |
| [asc_sync_mte2](sync/asc_sync_mte2.md)| 等待PIPE_MTE2流水完成。 |
| [asc_sync_data_barrier](sync/asc_sync_data_barrier.md) | 用于阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可以通过参数控制）执行结束。 |


## 系统变量

系统变量类API，单独使用时可以引入sys_var.h，此类API列表如下：

|   API名称   |   说明   |
|----------|-----------|
| [asc_get_block_num](sys_var/asc_get_block_num.md) | 获取AI核数。 |
| [asc_get_block_idx](sys_var/asc_get_block_idx.md) | 获取当前运行核的索引。 |
| [asc_get_core_id](sys_var/asc_get_core_id.md) | 获取当前核的编号。 |
| [asc_get_sub_block_id](sys_var/asc_get_sub_block_id.md) | 获取AI Core上Vector核的ID。 |
| [asc_get_sub_block_num](sys_var/asc_get_sub_block_num.md) | 分离模式下，获取一个AI Core上Cube Core（AIC）或者Vector Core（AIV）的数量。 |
| [asc_set_ctrl](sys_var/asc_set_ctrl.md) | 设置CTRL寄存器（控制寄存器）的值。 |
| [asc_get_ctrl](sys_var/asc_get_ctrl.md) | 读取CTRL寄存器（控制寄存器）的值。 |
| [asc_get_phy_buf_addr](sys_var/asc_get_phy_buf_addr.md) | 基于偏移量获取片上实际物理地址。 |
| [asc_get_system_cycle](sys_var/asc_get_system_cycle.md) | 获取当前系统cycle数。 |
| [asc_get_arch_ver](sys_var/asc_get_arch_ver.md) | 获取当前AI处理器架构版本号。 |
| [asc_get_program_counter](sys_var/asc_get_program_counter.md) | 获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。 |


## 缓存控制

缓存控制类API，单独使用时可以引入cache_ctrl.h，此类API列表如下：

|   API名称   |   说明   |
|----------|-----------|
| [asc_datacache_preload](cache_ctrl/asc_datacache_preload.md)| 从源地址所在的特定GM地址预加载数据到Data Cache中。 |
| [asc_dcci](cache_ctrl/asc_dcci.md) | 用于刷新Cache， 保证Cache的一致性。 |
| [asc_get_icache_preload_status](cache_ctrl/asc_get_icache_preload_status.md) | 获取ICache的Preload的状态。 |
| [asc_icache_preload](cache_ctrl/asc_icache_preload.md) | 从指令所在DDR地址预加载数据到对应的cacheline中。 |

## 原子操作

原子操作类API，单独使用时可以引入atomic.h，此类API列表如下：

|   API名称   |   说明   |
|----------|-----------|
| [asc_set_atomic_add_bfloat](simd_atomic/asc_set_atomic_add.md) | 设置对后续的从Unified Buffer/L0C Buffer/L1 Buffer到Global Memory的数据传输开启原子累加。累加的数据类型为bfloat16_t。 |
| [asc_set_atomic_add_float](simd_atomic/asc_set_atomic_add.md) | 设置对后续的从Unified Buffer/L0C Buffer/L1 Buffer到Global Memory的数据传输开启原子累加。累加的数据类型为float。 |
| [asc_set_atomic_add_float16](simd_atomic/asc_set_atomic_add.md) | 设置对后续的从Unified Buffer/L0C Buffer/L1 Buffer到Global Memory的数据传输开启原子累加。累加的数据类型为half。 |
| [asc_set_atomic_add_int](simd_atomic/asc_set_atomic_add.md) | 设置对后续的从Unified Buffer/L0C Buffer/L1 Buffer到Global Memory的数据传输开启原子累加。累加的数据类型为int32_t。 |
| [asc_set_atomic_add_int8](simd_atomic/asc_set_atomic_add.md) | 设置对后续的从Unified Buffer/L0C Buffer/L1 Buffer到Global Memory的数据传输开启原子累加。累加的数据类型为int8_t。 |
| [asc_set_atomic_add_int16](simd_atomic/asc_set_atomic_add.md) | 设置对后续的从Unified Buffer/L0C Buffer/L1 Buffer到Global Memory的数据传输开启原子累加。累加的数据类型为int16_t。 |
| [asc_set_atomic_max_bfloat](simd_atomic/asc_set_atomic_max.md) | 设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的bfloat16_t数据与GM中已有数据进行逐元素比较，并将最大值写入GM。 |
| [asc_set_atomic_max_float](simd_atomic/asc_set_atomic_max.md) | 设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的float数据与GM中已有数据进行逐元素比较，并将最大值写入GM。 |
| [asc_set_atomic_max_float16](simd_atomic/asc_set_atomic_max.md) | 设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的half数据与GM中已有数据进行逐元素比较，并将最大值写入GM。 |
| [asc_set_atomic_max_int](simd_atomic/asc_set_atomic_max.md) | 设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的int32_t数据与GM中已有数据进行逐元素比较，并将最大值写入GM。 |
| [asc_set_atomic_max_int8](simd_atomic/asc_set_atomic_max.md) | 设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的int8_t数据与GM中已有数据进行逐元素比较，并将最大值写入GM。 |
| [asc_set_atomic_max_int16](simd_atomic/asc_set_atomic_max.md) | 设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的int16_t数据与GM中已有数据进行逐元素比较，并将最大值写入GM。 |
| [asc_set_atomic_min_bfloat](simd_atomic/asc_set_atomic_min.md) | 设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的bfloat16_t数据与GM中已有数据进行逐元素比较，并将最小值写入GM。 |
| [asc_set_atomic_min_float](simd_atomic/asc_set_atomic_min.md) | 设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的float数据与GM中已有数据进行逐元素比较，并将最小值写入GM。 |
| [asc_set_atomic_min_float16](simd_atomic/asc_set_atomic_min.md) | 设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的half数据与GM中已有数据进行逐元素比较，并将最小值写入GM。 |
| [asc_set_atomic_min_int](simd_atomic/asc_set_atomic_min.md) | 设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的int32_t数据与GM中已有数据进行逐元素比较，并将最小值写入GM。 |
| [asc_set_atomic_min_int8](simd_atomic/asc_set_atomic_min.md) | 设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的int8_t数据与GM中已有数据进行逐元素比较，并将最小值写入GM。 |
| [asc_set_atomic_min_int16](simd_atomic/asc_set_atomic_min.md) | 设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的int16_t数据与GM中已有数据进行逐元素比较，并将最小值写入GM。 |
| [asc_set_store_atomic_config](simd_atomic/asc_set_store_atomic_config.md)| 设置原子操作使能位与原子操作类型的值。 |
| [asc_get_store_atomic_config](simd_atomic/asc_get_store_atomic_config.md)| 获取原子操作使能位与原子操作类型的值。 |
| [asc_set_atomic_none](simd_atomic/asc_set_atomic_none.md) | 清空原子操作的状态。 |

## 其他操作

|   API名称   |   说明   |
|----------|-----------|
| [asc_init](misc/asc_init.md)| 初始化NPU状态。 |
