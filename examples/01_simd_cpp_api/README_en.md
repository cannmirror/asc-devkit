# Sample Execution and Verification

After developers implement custom operators using Ascend C API, they can verify operator functionality through single-operator invocation. This code repository provides some operator implementations and their invocation examples, as shown below.

## Sample List
|  Directory Name                                                   |  Description                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_introduction](./00_introduction) | Simple examples based on Ascend C, implementing custom operators through Ascend C programming language, with corresponding <<<>>> direct invocation implementations |
| [01_utilities](./01_utilities) | Simple examples based on Ascend C, introducing system tool usage methods such as on-board printing, exception detection, and CPU twin debugging through APIs like printf, assert, and DumpTensor, suitable for debugging phases |
| [02_features](./02_features) | Feature examples based on Ascend C, introducing features such as Aclnn (ge graph entry) projects, LocalMemAllocator, and separate memory allocation for Barrier |
| [03_libraries](./03_libraries) | Usage examples based on Ascend C API libraries, introducing API libraries such as math libraries and activation functions through <<<>>> direct invocation implementations |
| [04_best_practices](./04_best_practices) | Performance optimization practices based on Ascend C, focusing on key operator and memory access tuning, aimed at improving runtime efficiency on Ascend platforms |
| [05_compatibility_guide](./05_compatibility_guide) | Compatibility examples for incompatible features |