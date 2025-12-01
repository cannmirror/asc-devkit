# C API

## 数据结构

|结构名|说明|
|-----------------------|-----------------------|
| [asc_binary_config](struct/asc_binary_config.md) | 双目高维切分计算API读写控制参数 |
| [asc_unary_config](struct/asc_unary_config.md) | 单目高维切分计算API读写控制参数 |
| [asc_copy_config](struct/asc_copy_config.md) | 数据搬运API读写控制参数 |
| [asc_block_reduce_config](struct/asc_block_reduce_config.md) | block reduce类计算API配置参数 |
| [asc_repeat_reduce_config](struct/asc_repeat_reduce_config.md) | repeat reduce类计算API配置参数 |
| [asc_duplicate_config](struct/asc_duplicate_config.md) | asc_duplicate API配置参数 |
| [asc_brcb_config](struct/asc_brcb_config.md) |asc_brcb API配置参数 |

## 矢量计算

|   API名称   |   说明   |
|----------|-----------|
| [asc_get_cmp_mask](vector_compute/asc_get_cmp_mask.md) | 获取Compare操作的比较结果。 |
| [asc_set_cmp_mask](vector_compute/asc_set_cmp_mask.md) | 为Select操作设置用于选择的掩码。 |
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
| [asc_max](vector_compute/asc_max.md) | 按元素求最大值。 |
| [asc_min](vector_compute/asc_min.md) | 按元素求最小值。 |
| [asc_datablock_reduce_sum](vector_compute/asc_datablock_reduce_sum.md) | 对每个DataBlock内所有元素求和。 |
| [asc_datablock_reduce_max](vector_compute/asc_datablock_reduce_max.md) | 对每个DataBlock内所有元素求最大值。 |
| [asc_datablock_reduce_min](vector_compute/asc_datablock_reduce_min.md) | 对每个DataBlock内所有元素求最小值。 |
| [asc_repeat_reduce_sum](vector_compute/asc_repeat_reduce_sum.md) | 对每个Repeat内所有元素求和。 |
| [asc_repeat_reduce_max](vector_compute/asc_repeat_reduce_max.md) | 对每个Repeat内所有元素求最大值。 |
| [asc_repeat_reduce_min](vector_compute/asc_repeat_reduce_min.md) | 对每个Repeat内所有元素求最小值。 |
| [asc_brcb](vector_compute/asc_brcb.md) | 将源操作数中的每一个数填充到目的操作数的一个DataBlock中。 |
| [asc_duplicate](vector_compute/asc_duplicate.md) | 将一个变量或立即数填充到一个矢量中。 |
| [asc_select](vector_compute/asc_select.md) | 根据掩码，从两个源操作数中选取元素，输出到目的操作数。 |
| [asc_bf162float](vector_compute/asc_bf162float.md) | 数据类型转换。将bfloat16_t类型的数据转换为float类型。 |
| [asc_bf162int32](vector_compute/asc_bf162int32.md) | 数据类型转换。将bfloat16_t类型的数据转换为int32_t类型。 |
| [asc_float2bf16](vector_compute/asc_float2bf16.md) | 数据类型转换。将float类型的数据转换为bfloat16_t类型。 |
| [asc_float2float](vector_compute/asc_float2float.md) | 数据类型转换。将float类型的数据转换为float类型。 |
| [asc_float2half](vector_compute/asc_float2half.md) | 数据类型转换。将float类型的数据转换为half类型。 |
| [asc_half2float](vector_compute/asc_half2float.md) | 数据类型转换。将half类型的数据转换为float类型。 |
| [asc_half2int4](vector_compute/asc_half2int4.md) | 数据类型转换。将half类型的数据转换为int4b_t类型。 |
| [asc_half2int16](vector_compute/asc_half2int16.md) | 数据类型转换。将half类型的数据转换为int16_t类型。 |
| [asc_half2int32](vector_compute/asc_half2int32.md) | 数据类型转换。将half类型的数据转换为int32_t类型。 |

## 数据搬运

|   API名称   |   说明   |
|----------|-----------|
| [asc_copy_ub2gm](vector_datamove/asc_copy_ub2gm.md) | 将UB中的数据搬运到GM中。 |
| [asc_copy_gm2ub](vector_datamove/asc_copy_gm2ub.md) | 将GM中的数据搬运到UB中。 |

## 同步控制

|   API名称   |   说明   |
|----------|-----------|
| [asc_sync_notify](sync/asc_sync_notify.md)| 设置同步标志。 |
| [asc_sync_wait](sync/asc_sync_wait.md)| 等待同步标志。 |
| [asc_sync](sync/asc_sync.md)| 等待指定流水线或所有流水线操作完成。 |
| [asc_sync_vec](sync/asc_sync_vec.md)| 同步所有流水线。 |
| [asc_sync_mte3](sync/asc_sync_mte3.md)| 等待PIPE_TYPE_MTE3流水完成。 |
| [asc_sync_mte2](sync/asc_sync_mte2.md)| 等待PIPE_TYPE_MTE2流水完成。 |

## 系统变量

|   API名称   |   说明   |
|----------|-----------|
| [asc_get_block_num](sys_var/asc_get_block_num.md) | 获取AI核数。 |
| [asc_get_block_idx](sys_var/asc_get_block_idx.md) | 获取当前运行核的索引。 |
| [asc_get_core_id](sys_var/asc_get_core_id.md) | 获取当前核的编号。 |
| [asc_get_sub_block_id](sys_var/asc_get_sub_block_id.md) | 获取AI Core上Vector核的ID。 |
| [asc_get_sub_block_dim](sys_var/asc_get_sub_block_dim.md) | 分离模式下，获取一个AI Core上Cube Core（AIC）或者Vector Core（AIV）的数量。 |
| [asc_set_ctrl](sys_var/asc_set_ctrl.md) | 设置CTRL寄存器（控制寄存器）的值。 |
| [asc_get_ctrl](sys_var/asc_get_ctrl.md) | 读取CTRL寄存器（控制寄存器）的值。 |
| [asc_get_phy_buf_addr](sys_var/asc_get_phy_buf_addr.md) | 基于偏移量获取片上实际物理地址。 |
| [asc_get_system_cycle](sys_var/asc_get_system_cycle.md) | 获取当前系统cycle数。 |

## 原子操作

|   API名称   |   说明   |
|----------|-----------|
| [asc_data_cache_clean_and_invalid](simd_atomic/asc_data_cache_clean_and_invalid.md)| 用于刷新Cache， 保证Cache的一致性。 |

## 其他操作

|   API名称   |   说明   |
|----------|-----------|
| [asc_init](misc/asc_init.md)| 初始化NPU状态。 |
