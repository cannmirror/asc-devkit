# Memory Vector Compute Operator Examples

## Overview

This example set introduces typical usage of different features of Memory vector compute operators, providing end-to-end implementations.

## Example List

| Directory Name                                                                                                                 | Description                                                                        | 
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------| 
| [arithmetic](./arithmetic) | This example demonstrates the usage of basic arithmetic interfaces based on LeakyRelu |
| [block_reduce_min_max_sum](./block_reduce_min_max_sum)                                                                               | This example implements reduction computation using BlockReduceMax/BlockReduceMin/BlockReduceSum                         |
| [cast](./cast) | This example implements data type and precision conversion between source and destination tensors using Cast |
| [compare](./compare)                                                                                             | This example implements data comparison functionality in multiple scenarios using Compare and Compares interfaces, performing element-wise comparison.                                              |
| [select](./select)                                                                                               | This example implements data selection functionality in multiple scenarios using the Select interface, selecting elements from two vectors or between a vector and a scalar based on a mask and writing them to a destination vector                                                                                            |
| [create_vec_index](./create_vec_index) | This example demonstrates the method of creating a vector index with a specified starting value using CreateVecIndex |
| [brcb](./brcb) | This example implements data padding using Brcb, which can be used to fill 8 numbers from the input tensor into 8 datablocks of the result tensor each time |
| [duplicate](./duplicate) | This example implements data padding using Duplicate, which can be used to copy a variable or immediate value multiple times and fill it into a vector |
| [element_wise_compound_compute](./element_wise_compound_compute) | This example introduces the usage of Ascend C vector compute compound interfaces |
| [gather](./gather)         | This example implements data selection functionality in multiple scenarios using GatherMask, Gather, Gatherb and other interfaces, selecting elements from source operands and writing them to destination operands. |
| [mrg_sort](./mrg_sort) | This example implements merging up to 4 pre-sorted queues into 1 queue using Sort32 and MrgSort basic APIs, with results sorted by score field in descending order |
| [pair_reduce_sum](./pair_reduce_sum) | This example implements sum reduction for adjacent odd-even element pairs using PairReduceSum |
| [reduce_computation](./reduce_computation)         | This example implements reduction computation using ReduceMax/ReduceMin/ReduceSum interfaces              |
| [region_proposal_sort](./region_proposal_sort) | This example introduces the usage of Region Proposal related sorting interfaces |
| [transpose](./transpose) | This example implements data transpose functionality using Transpose and TransDataTo5HD interfaces, including 16*16 2D matrix block transpose, conversion between [N,C,H,W] and [N,H,W,C] 4D matrices, and NCHW format to NC1HWC0 format conversion. |
| [whole_reduce_min_max_sum](./whole_reduce_min_max_sum)                                                                               | This example introduces the usage of reduction interfaces in multiple scenarios, including WholeReduceMax, WholeReduceMin, WholeReduceSum, RepeatReduceSum, and the usage of WholeReduceMax/Min combined with GetReduceRepeatMaxMinSpr to obtain global extremum and indices.                  |
| [interleave_pair](./interleave_pair) | This example implements element interleaving and de-interleaving functionality using Interleave and DeInterleave interfaces |
| [element_wise_logic](./element_wise_logic) | This example implements bitwise logic operations using And, Ors, ShiftLeft, ShiftRight interfaces |