# Sample Execution and Verification

After developers implement custom operators using Ascend C API, they can verify operator functionality through single operator calls. This repository provides sample operator implementations and their invocation examples, as described below.

## Sample List
|  Directory Name                                                   |  Description                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_introduction](./00_introduction) | Simple examples based on Ascend C, demonstrating custom operator implementations through Ascend C programming language with corresponding C_API implementations |
| [01_utilities](./01_utilities) | Simple examples based on Ascend C, demonstrating system utility usage including on-board printing, exception detection, and CPU twin debugging through APIs such as printf, assert, and DumpTensor, suitable for debugging phases |
| [02_features](./02_features) | Feature examples based on Ascend C, demonstrating operator sample implementation methods using Framework, C_API, and other features |