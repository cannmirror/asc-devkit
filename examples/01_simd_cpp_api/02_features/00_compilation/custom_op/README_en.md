# Custom Operator Project Build, Package, and Deployment Sample

## Overview

This sample demonstrates the workflow of building, packaging into a custom operator package, and deploying to a CANN environment based on a simple custom operator.

## Supported Products

This sample supports the following product models:

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products
- Atlas 200I/500 A2 Inference Products
- Atlas Inference Series Products

> Note: This sample involves multiple operator examples. Please refer to the actual supported product models for each operator example.

## Directory Structure

```
├── CMakeLists.txt
├── framework
│   ├── CMakeLists.txt
│   ├── onnx_plugin
│   │   ├── CMakeLists.txt
│   │   └── leaky_relu_custom_plugin.cc
│   └── tf_plugin
│       ├── CMakeLists.txt
│       └── tensorflow_add_custom_plugin.cc
├── op_host
│   ├── CMakeLists.txt
│   ├── add_custom
│   │   └── add_custom_host.cpp
│   ├── add_custom_template
│   │   └── add_custom_template.cpp
│   ├── add_custom_tiling_sink
│   │   ├── add_custom_tiling_sink.cpp
│   │   ├── add_custom_tiling_sink_tiling.cpp
│   │   └── add_custom_tiling_sink_tiling.h
│   └── leaky_relu_custom
│       └── leaky_relu_custom_host.cpp
└── op_kernel
    ├── CMakeLists.txt
    ├── add_custom
    │   ├── add_custom_kernel.cpp
    │   └── add_custom_tiling.h
    ├── add_custom_template
    │   ├── add_custom_template.cpp
    │   ├── add_custom_template_tiling.h
    │   └── tiling_key_add_custom_template.h
    ├── add_custom_tiling_sink
    │   ├── add_custom_tiling_sink_kernel.cpp
    │   └── add_custom_tiling_sink_tiling_struct.h
    └── leaky_relu_custom
        ├── leaky_relu_custom_kernel.cpp
        └── leaky_relu_custom_tiling.h
```

## Sample Description

The Add computation formula is:

```
z = x + y
```

AddCustomTilingSink, AddCustomTemplate, and Add have the same kernel function. Specifically:

- AddCustomTemplate demonstrates Tiling template programming with template parameters including input data type, shape, etc. Based on template parameters, the sample implementation logic is simplified or unified. Developers can define required information in template parameters, such as input and output data types and other extended parameters.
- AddCustomTilingSink demonstrates the Tiling sinking scenario, where the Tiling function is registered to both host and device for execution through `DEVICE_IMPL_OP_OPTILING`.

The LeakyRelu computation formula is:

$$
y=
\begin{cases}
x, \quad x\geq 0\\
a*x, \quad x<0
\end{cases}
$$

Where a is a scalar value.

## Sample Specification Description

- Add

  <table border="2" align="center">
  <caption>Table 1: Add Sample Specification Description</caption>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">Add</td></tr>
  <tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">add_custom, add_custom_tiling_sink</td></tr>
  </table>

- AddCustomTemplate

  <table border="2" align="center">
  <caption>Table 2: AddCustomTemplate Sample Specification Description</caption>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">Add</td></tr>
  <tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">add_template_custom</td></tr>
  <tr><td rowspan="6" align="center">Template Parameters</td><td colspan="4" align="center">template&lt;typename D_T_X, typename D_T_Y, typename D_T_Z, int TILE_NUM, int IS_SPLIT&gt;</td>
      <tr><td>D_T_X</td><td colspan="1">typename</td><td colspan="2">Data type (half, float)</td></tr>
      <tr><td>D_T_Y</td><td colspan="1">typename</td><td colspan="2">Data type (half, float)</td></tr>
      <tr><td>D_T_Z</td><td colspan="1">typename</td><td colspan="2">Data type (half, float)</td></tr>
      <tr><td>TILE_NUM</td><td colspan="1">int</td><td colspan="2">Tile count</td></tr>
      <tr><td>IS_SPLIT</td><td colspan="1">int</td><td colspan="2">Whether to split</td></tr>
  </tr>
  </table>

- LeakyRelu

  <table border="2" align="center">
  <caption>Table 3: LeakyRelu Sample Specification Description</caption>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">LeakyRelu</td></tr>
  <tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[8, 200, 1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">negative_slope</td><td align="center">0.0</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">y</td><td align="center">[8, 200, 1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">leaky_relu_custom</td></tr>
  </table>

## Code Implementation Description

- Add

  - Kernel implementation:

    Implemented based on the vector computation interface `Add` provided by Ascend C.

  - Tiling implementation:

    TilingData parameter design. The `AddCustomTilingData` parameter is essentially related to parallel data splitting. This sample uses two tiling parameters: `totalLength` and `tileNum`. `totalLength` is the size of data to be computed, and `tileNum` is the number of data blocks to be computed on each core.

- AddCustomTemplate

  - Kernel implementation:

    Same as Add.

  - Tiling template design:

    This sample uses five template parameters. `D_T_X`, `D_T_Y`, and `D_T_Z` are the data types for input x, input y, and output z respectively. `TILE_NUM` is the number of data blocks to be computed on each core. `IS_SPLIT` indicates whether to enable data block computation. When `IS_SPLIT` is 0, `TILE_NUM` is invalid. Template parameters replace the traditional TilingKey.

  - TilingData parameter design:

    This sample uses one tiling parameter. `totalLength` is the total amount of data to be computed across all cores.

- AddCustomTilingSink

  - Kernel implementation:

    Same functionality as Add. The kernel specifies the sample to run in AIC and AIV hybrid scenario through the `KERNEL_TASK_TYPE_DEFAULT` interface to satisfy the Tiling sinking operator condition. All Tiling function logic is implemented separately in `add_custom_tiling_sink_tiling.cpp` and the sinking Tiling function is registered through the `DEVICE_IMPL_OP_OPTILING` interface.

- LeakyRelu

  - Kernel implementation:

    Implemented based on the high-level API interface `LeakyRelu`.

  - Tiling implementation:

    TilingData parameter design. The `LeakyReluCustomTilingData` parameter is essentially related to parallel data splitting. This sample uses three tiling parameters: `totalLength`, `tileNum`, and `negativeSlope`. `totalLength` and `tileNum` are similar to the Add sample. `negativeSlope` represents the negative axis slope coefficient of LeakyRelu, passed to the kernel side as a computation parameter.

## Build and Run

Execute the following steps in the sample root directory to build, package, and deploy the custom sample package.

- Configure environment variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on the current environment.

  - Default path, CANN package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Build, package, and deploy sample

  ```bash
  mkdir -p build && cd build
  cmake .. && make -j binary package
  ./custom_opp_*.run
  ```

  The following output indicates successful execution:

  ```log
  SUCCESS
  ```

## Build Cache Acceleration (Optional)

This sample supports accelerating repeated builds through ccache, providing both local cache and distributed cache modes.

### Prerequisites

- `ccache` installed, `ccache >= 4.6.1` recommended

  ```bash
  apt install ccache
  ```

- For distributed cache scenarios, confirm that the current version supports Redis storage

  ```bash
  ccache --version
  # Expected: Features include redis-storage
  ```

### Local Cache

After enabling `-DENABLE_CCACHE=ON`, `cmake` will connect `ccache` to the build process. When source code, build commands, compiler, and build directory path are identical, repeated builds can directly reuse local cache results.

Enable through cmake parameter without modifying CMakeLists.txt:

```bash
mkdir -p build && cd build
cmake -DENABLE_CCACHE=ON .. && make -j binary package
```

Verify whether local cache is effective as follows:

```bash
ccache -Cz
# First build: write cache
rm -rf build && mkdir -p build && cd build
cmake -DENABLE_CCACHE=ON .. && make -j binary package
cd ..
ccache -z
# Second build: cache hit
rm -rf build && mkdir -p build && cd build
cmake -DENABLE_CCACHE=ON .. && make -j binary package
```

After each build, view cache hit statistics through `ccache --show-stats -v`. You will see the local cache hit rate `Local storage Hits` significantly increase in the second build.

To clear local cache, operate as follows:

```bash
# Clear statistics only, without deleting cache content
ccache -z

# Clear local cache content
ccache -C

# Clear both local cache content and statistics
ccache -Cz
```

### Distributed Cache (ccache + Redis)

Suitable for multi-machine shared cache scenarios: Machine A compiles and pushes results to Redis, Machine B can hit cache from Redis under the same source code, compilation options, and toolchain version, reducing repeated compilation.

In distributed scenarios, `ccache` uses each machine's local cache as the first-level cache and Redis as the shared second-level cache. When Machine A compiles for the first time, it calls the actual compiler and writes results to both local cache and Redis. When Machine B compiles again under the same source code, compilation commands, compiler, and build directory path, it can directly hit the shared cache from Redis, reducing repeated compilation execution. If compilers on two machines have different paths but identical content, it is recommended to set `compiler_check=content`.

For more `ccache` configuration and cache behavior details, refer to the [ccache official documentation](https://ccache.dev/documentation.html).

Network requirements:

- Machine A: First compilation machine, writes cache to Redis, IP is `<A_IP>`
- Machine B: Second compilation machine, verifies shared cache hit from Redis, IP is `<B_IP>`
- Machine C: Redis server, stores shared cache data, IP is `<C_IP>`
- Machines A, B, and C must be on the same network. Both A and B machines must be able to access `C_IP:6379`

It is recommended that Machine A and Machine B use the same source code content, compilation commands, compiler versions, and maintain consistent source code paths and build directory paths. Otherwise, cache misses may occur.

**1. Machine C: Deploy Redis Service**

```bash
apt install redis-server
# Start Redis service
redis-server --daemonize yes --bind 0.0.0.0 --port 6379 --requirepass <PASSWORD>
# Verify Redis connection
redis-cli -h <C_IP> -p 6379 -a <PASSWORD> ping
```

> Note: The above configuration is for controlled test environments only. For shared or production environments, it is recommended to enable access control, authentication, and network isolation.

**2. Machine A / Machine B: Configure ccache**

```bash
apt install redis-tools
# Verify Redis connection
redis-cli -h <C_IP> -p 6379 -a <PASSWORD> ping
# Configure Redis as secondary storage with password authentication
# Format: redis://default:<PASSWORD>@<C_IP>:6379
ccache --set-config=secondary_storage=redis://default:<PASSWORD>@<C_IP>:6379
# Configure compiler content check to avoid cache misses due to path differences
ccache --set-config=compiler_check=content
```

**3. Machine A / Machine B: Execute Build**

```bash
ccache -Cz
rm -rf build && mkdir -p build && cd build
cmake -DENABLE_CCACHE=ON .. && make -j binary package
```

Machine A's build will write to local cache and Redis. Machine B's local cache is empty and can read shared cache from Redis. After Machine A builds, execute the build on Machine B and compare the two `ccache --show-stats -v` results. You will see the remote cache hit rate `Remote storage Hits` significantly increase on Machine B.