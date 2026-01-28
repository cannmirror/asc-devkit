# asc_copy_l0c2l1

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :----------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |

## 功能说明

矩阵计算完成后，对结果进行量化处理，之后将处理结果搬运到GM中。量化模式共分为8种，分别为：
- DEQF16：int32_t量化成half。量化结果不支持INF_NAN模式。
- F322BF16：float量化成bfloat16_t。量化结果不支持INF_NAN模式。
- F322F16：float量化成half。量化结果不支持INF_NAN模式。
- QF322B8_PRE：float量化成uint8_t/int8_t。scalar量化。
- REQ8：int32_t量化成uint8_t/int8_t。scalar量化。
- VDEQF16：int32_t量化成half。量化结果不支持INF_NAN模式。
- VQF322B8_PRE：float量化成uint8_t/int8_t。scalar量化。
- VREQ8：int32_t量化成uint8_t/int8_t。矢量量化。

## 函数原型

- 常规搬运

    ```c++
    // DEQF16模式
    __aicore__ inline void asc_copy_l0c2l1_deqf16(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_deqf16(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_deqf16(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_deqf16(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_deqf16(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    
    // F322BF16模式
    __aicore__ inline void asc_copy_l0c2l1_f322bf16(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_f322bf16(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_f322bf16(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_f322bf16(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_f322bf16(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
  
    // F322F16模式
    __aicore__ inline void asc_copy_l0c2l1_f322f16(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_f322f16(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_f322f16(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_f322f16(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_f322f16(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
  
    // QF322B8_PRE模式
    __aicore__ inline void asc_copy_l0c2l1_qf322b8_pre(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_qf322b8_pre(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_qf322b8_pre(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_qf322b8_pre(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_qf322b8_pre(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
  
    // REQ8模式
    __aicore__ inline void asc_copy_l0c2l1_req8(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_req8(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_req8(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_req8(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_req8(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
  
    // VDEQF16模式
    __aicore__ inline void asc_copy_l0c2l1_vdeqf16(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_vdeqf16(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_vdeqf16(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_vdeqf16(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_vdeqf16(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
  
    // VQF322B8_PRE模式
    __aicore__ inline void asc_copy_l0c2l1_vqf322b8_pre(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_vqf322b8_pre(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_vqf322b8_pre(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_vqf322b8_pre(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_vqf322b8_pre(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
  
    // VREQ8模式
    __aicore__ inline void asc_copy_l0c2l1_vreq8(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_vreq8(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_vreq8(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_vreq8(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_vreq8(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    ```
- 同步搬运

    ```c++
    // DEQF16模式
    __aicore__ inline void asc_copy_l0c2l1_deqf16_sync(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_deqf16_sync(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_deqf16_sync(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_deqf16_sync(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_l0c2l1_deqf16_sync(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
  
    // F322BF16模式
    __aicore__ inline void asc_copy_copy_f322bf16_sync(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_f322bf16_sync(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_f322bf16_sync(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_f322bf16_sync(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_f322bf16_sync(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
  
    // F322F16模式
    __aicore__ inline void asc_copy_copy_f322f16_sync(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_f322f16_sync(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_f322f16_sync(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_f322f16_sync(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_f322f16_sync(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
  
    // QF322B8_PRE模式
    __aicore__ inline void asc_copy_copy_qf322b8_pre_sync(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_qf322b8_pre_sync(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_qf322b8_pre_sync(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_qf322b8_pre_sync(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_qf322b8_pre_sync(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
  
    // REQ8模式
    __aicore__ inline void asc_copy_copy_req8_sync(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_req8_sync(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_req8_sync(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_req8_sync(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_req8_sync(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    
    // VDEQF16模式
    __aicore__ inline void asc_copy_copy_vdeqf16_sync(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_vdeqf16_sync(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_vdeqf16_sync(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_vdeqf16_sync(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_vdeqf16_sync(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
  
    // VQF322b8_pre模式
    __aicore__ inline void asc_copy_copy_vqf322b8_pre_sync(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_vqf322b8_pre_sync(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_vqf322b8_pre_sync(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_vqf322b8_pre_sync(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_vqf322b8_pre_sync(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
  
    // VREQ8模式
    __aicore__ inline void asc_copy_copy_vreq8_sync(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_vreq8_sync(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_vreq8_sync(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_vreq8_sync(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    __aicore__ inline void asc_copy_copy_vreq8_sync(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split, bool nd2nz_en)
    ```

## 参数说明


| 参数名       | 输入/输出 | 描述                                                                                                                               |
|:----------|:------|:---------------------------------------------------------------------------------------------------------------------------------|
| dst       | 输出    | 目的操作数（矢量）的起始地址。                                                                                                                           |
| src       | 输入    | 源操作数的起始地址。                                                                                                                           |
| sid       | 输入    | 保留，未使用，填0即可。                                                                                                                     |
| n_size    | 输入    | 源NZ矩阵在N方向上的大小。<br/>&bull; 不使能NZ2ND功能，取值范围：[1, 4095]；<br/>&bull; 使能NZ2ND功能，取值范围：[1, 4095]。                                        |
| m_size   | 输入    | 源NZ矩阵在M方向上的大小。<br/>&bull; 不使能NZ2ND功能，取值范围：[1, 65535]；<br/>&bull; 使能NZ2ND功能，取值范围：[1, 8192]。                                       |
| dst_stride_dst_d | 输入    | 目的相邻ND矩阵起始地址之间的偏移。                                                                                                               |
| src_stride   | 输入    | 源NZ矩阵中相邻Z排布的起始地址偏移，取值范围：[0, 65535]，单位：C0_Size（16*sizeof(T)，T为src的数据类型）。                                                          |
| uint_flag_mode   | 输入    | 与unit_flag参数相关，取值如下：<br/>&bull;0 保留值；<br/>&bull;2 使能unit_flag，硬件执行完指令之后，不会设置寄存器；<br/>&bull;3 使能unit_flag，硬件执行完指令后，会将unit_flag关闭。 |
| relu_pre   | 输入    | 使能relu。                                                                                                                          |
| channel_split   | 输入    | 是否使能通道拆分的功能，默认false，不使能该功能。仅在src和dst都为float时才能使能通道拆分，且不能同时使能channel_split和NZ2ND功能。                                               |
| nd2nz_en   | 输入    | 使能nz2nd开关。<br/>&bull;false：不使能；<br/>&bull;true：使能。                                                                               |

## 返回值说明

无

## 流水类型

PIPE_FIX

## 约束说明

- dst的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

## 调用示例

```cpp
//total_length指参与搬运的数据总长度
constexpr uint64_t total_length = 128;
__cbuf__ half dst[total_length];
asc_copy_l0c2l1_deqf16(dst, src, 0, n_size, m_size, dst_stride_dst_d, src_stride, unit_flag_mode, relu_pre, channel_split, nz2nd_en);
```
