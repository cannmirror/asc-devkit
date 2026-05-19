# SIMT InsertHashTable Operator Example

## Overview

This example introduces the InsertHashTable operator and demonstrates operator implementation for large-scale concurrent thread access to Global Memory based on SIMT.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions
- \> CANN 9.0.0

## Directory Structure

```
├── 00_simt_insert_hash_table
│   ├── CMakeLists.txt          # cmake build file
│   ├── insert_hash_table.asc   # Ascend C operator implementation & invocation example
|   └── README.md
```

## Background

A Hash Table is an efficient data structure that uses a hash function to map a "key" to a specific position in a fixed-size array (bucket array), enabling fast lookup, insertion, and deletion operations.

- Hash function: A function that accepts a key as input and computes an integer (hash value), which is mapped to an index of the array through some operation.
- Bucket array: An array where each element (called a "bucket") stores key-value pairs.

The basic process for inserting a key-value pair into a hash table is:

- Use the hash function to compute the hash value of the key.
- Map the hash value to an index of the bucket array (typically by taking the remainder of the hash table capacity).
- Store the key-value pair at the bucket[index] position in the array.

Hash collision: Since the capacity of a hash table (bucket array size) is much smaller than the output range of the hash function, different keys may be mapped to the same index.

A common method to resolve hash collisions is open addressing: when a collision occurs, the algorithm searches for the next "empty" bucket using linear probing to store the data.

Since the storage location of each key-value pair is determined by the computed hash value, which is typically random and scattered, and due to the existence of hash collisions, multiple conditional judgments are required before actually writing data. Therefore, hash tables are not suitable for implementation based on the SIMD programming model. In contrast, in SIMT programming, each thread can independently handle branch judgments and supports scattered memory access, making it more advantageous for implementing hash tables.

## Algorithm Analysis

SIMT achieves efficient processing of large amounts of data through concurrent execution of many threads, but it also introduces two issues: multi-thread write conflicts and inter-core data synchronization.

### Multi-thread Write Conflict Issue

When multiple threads operate on the same memory region, resource conflicts are inevitable. When keys inserted by two threads produce hash collisions, multiple threads will attempt to write data to the same position in the bucket array, so you must ensure that only one thread can obtain write permission to the bucket. In the program implementation, a flag bit "flag" is added to the Bucket structure to mark the write permission of the current bucket. Threads use the atomic instruction asc_atomic_cas() to modify the flag to obtain write permission to the bucket.

### Inter-core Data Synchronization Issue

When a hash collision occurs, the thread needs to determine whether the key stored in the current bucket is the same as the key to be inserted. This requires the current thread to read data written by other threads and ensure data integrity. In the program implementation, a flag bit "state" is added to the Bucket structure to indicate the write status of the key value. In the write thread, after writing the key, set the state flag to 1, and call asc_threadfence() between the two operations to ensure that when state is set to 1, the write operation of the key has completed. In the read thread, poll the state value through a while loop until state is set to 1, and then read the key value.

```C++
struct Bucket {
    int64_t key;            // key
    uint32_t state;         // key value write status flag
    uint32_t flag;          // atomic operation flag bit
    float value[32];        // value
};
```

## Operator Description

- Operator function:
  The InsertHashTable operator inserts N key-value pairs into a hash table with capacity Z, where key is an int64_t type number and value is a float type Tensor of length M.

- Operator specification:
  <table>
  <tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="4" align="center">insert_hash_table</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">keys</td><td align="center">1, N</td><td align="center">int64_t</td><td align="center">ND</td></tr>
  <tr><td align="center">values</td><td align="center">N, M</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">table_addr</td><td align="center">1, Z</td><td align="center">Bucket</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">insert_hash_table</td></tr>
  </table>

- Operator implementation:
  The implementation flow of the InsertHashTable operator is that each warp processes one key-value pair. Thread 0 in the warp is responsible for finding an available bucket based on the hash value of the key, and threads 0-31 are responsible for storing the value into the bucket. When thread 0 searches for an available bucket, it uses open addressing to resolve hash collisions, uses the asc_atomic_cas() interface to resolve multi-thread conflicts, and uses the asc_threadfence() interface to resolve inter-core data synchronization issues.

- Invocation implementation:
  Use the kernel launch operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.
- Configure environment variables
  Select the corresponding command to configure environment variables based on the [installation method](../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on the current environment.
  - Default path, CANN software package installed by root user
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN software package installed by non-root user
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installation
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example execution
  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..; make -j;            # Build the project
  ./demo                        # Run the example
  ```
  The following output indicates successful accuracy verification.
  ```
  [Success] find all key-value in hash table.
  ```