# Matrix Compute API Example Introduction

## Overview

This example set introduces typical usage of different features of matrix computation APIs, providing corresponding end-to-end implementations. The naming of examples in the directory follows the pattern "API_name_path", as shown below:

1. **load_data_**: Examples starting with "load_data_" introduce relevant APIs for L1 Buffer -> L0 Buffer.
2. **mmad_**: Examples starting with "mmad_" introduce the matrix multiplication Mmad API.
3. **fixpipe_**: Examples starting with "fixpipe_" introduce relevant APIs for L0C Buffer → GM/L1 Buffer/UB.

## Example List

| Directory Name | Function Description |
|--------------------------------------------------------------------------------------------| ---------------------------------------------------- |
| [fixpipe_l0c2gm](./fixpipe_l0c2gm) | This example introduces how to use Fixpipe to move matrix multiplication results from CO1 (L0C Buffer) to GM (Global Memory) |
| [fixpipe_l0c2ub](./fixpipe_l0c2ub) | This example introduces how to use Fixpipe to move matrix multiplication results from CO1 (L0C Buffer) to UB (Unified Buffer) |
| [fixpipe_l0c2l1](./fixpipe_l0c2l1) | This example introduces how to use Fixpipe to move matrix multiplication results from CO1 (L0C Buffer) to L1 (L1 Buffer) |
| [load_data_l12l0](./load_data_l12l0) | This example introduces the usage of relevant instructions for 14 matrix multiplication scenarios with B4/B8/B16/B32 data types, covering combinations of left/right matrix transposition and non-transposition. It focuses on data movement from A1 to A2 and B1 to B2 using basic API LoadData, including Load2D, Load3D, and LoadDataWithTranspose |
| [load_data_2dv2_l12l0](./load_data_2dv2_l12l0) | This example introduces data movement from A1 to A2 and B1 to B2 using basic API LoadData, including Load2Dv2 |
| [load_data_2dmx_l12l0](./load_data_2dmx_l12l0) | This example introduces the usage of relevant instructions for 6 quantized matrix multiplication scenarios with FP4/FP8 data types, covering combinations of left/right matrix transposition and non-transposition. It focuses on data movement from L1 to L0 for matrices A, scaleA, B, and scaleB using basic API LoadData |
| [mmad_load3dv2](./mmad_load3dv2) | This example introduces the process of moving matrices A and B from L1 to L0A/L0B using the LoadData3DV2 instruction, where A and B respectively represent the left and right input matrices of matrix multiplication. The parameter configuration of the LoadData3DV2 instruction and the data layout changes before and after executing the instruction are explained with diagrams |
| [batch_mmad](./batch_mmad) | This example introduces batch matrix multiplication with float input data type and both left and right matrices not transposed. DataCopy ND2NZ and Fixpipe batch data movement are used for the GM-->L1, L0C-->GM, and L0C-->L1 paths. The L1-->L0A/L0B and Mmad matrix multiplication steps loop batch times, processing one pair of left and right matrices per loop |
| [mmad](./mmad) | This example introduces matrix multiplication with ND format input and B4/B8/B16/B32 input data types (specifically using int4_t/int8_t/bfloat16/float as examples), explaining how to implement matrix multiplication computation (C = A x B + Bias) through the Mmad instruction |
| [mmad_unitflag](./mmad_unitflag) | This example introduces how to use the unitFlag feature when calling the Mmad instruction |
| [mmad_gemv](./mmad_gemv) | This example introduces matrix multiplication in Gemv (M=1) mode |
| [mmad_with_sparse](./mmad_with_sparse) | This example introduces the basic API MmadWithSparse calling example |