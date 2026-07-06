# Presmoke 看护入口

`scripts/run_presmoke.sh` 是 CI 和人工看护入口。该脚本会执行一轮完整的 910B 或 950 presmoke，生成可归档报告，并支持将超时用例切换到其他卡上重试。

如需直接调试 Python CLI，可以使用 `PYTHONPATH=scripts PRESMOKE_PROJECT_ROOT=$PWD python3 -m presmoke ...`。

执行全量看护前，编排脚本会清理当前 CANN `opp/vendors` 目录下的所有内容，保证自定义算子包安装状态可复现。带 `--filter` 或 `--dry-run` 的运行不会清理 `opp/vendors`。

## 快速开始

使用默认参数运行。该命令最短，并保持与旧 `run_presmoke.sh` 调用方约定兼容：

```bash
bash scripts/run_presmoke.sh
```

默认行为：

- `ARCH=dav-2201`，对应 910B。
- `MODES=npu`。
- `NPU_CARDS=auto`，优先检测可见的 `/dev/davinci*` 卡并并行运行。
- `PRIMARY_CARD=0`，仅在自动检测不到 NPU 卡，或强制单轮运行时作为兜底主卡。
- `SCHEDULE=fixed`。
- `JOBS=auto`。
- `NPU_SLOTS=1`。
- `TIMEOUT=120`，只作用于 NPU run 命令。
- 成功时打印 `execute samples success`；失败时打印 `execute samples failed` 并返回非 0。

## 常用命令

只查看计划运行哪些用例，不实际执行：

```bash
bash scripts/run_presmoke.sh --dry-run
```

指定 910B 的某张卡运行：

```bash
NPU_CARDS=7 PRIMARY_CARD=7 bash scripts/run_presmoke.sh
```

运行 950：

```bash
ARCH=dav-3510 PRIMARY_CARD=0 bash scripts/run_presmoke.sh
```

运行单个用例：

```bash
bash scripts/run_presmoke.sh \
  --filter 01_simd_cpp_api/03_basic_api/03_matrix_compute/mmad_unitflag
```

指定报告输出目录：

```bash
OUT_ROOT=/tmp/presmoke_910b_card7 \
PRIMARY_CARD=7 \
bash scripts/run_presmoke.sh
```

显式运行 CPU 模式。CPU 模式不是默认模式：

```bash
MODES=cpu bash scripts/run_presmoke.sh
```

CPU 模式用例较慢时，可以提高 CPU run 超时时间：

```bash
MODES=cpu CPU_RUN_TIMEOUT=1800 bash scripts/run_presmoke.sh
```

## CI 命令

910B：

```bash
PROJECT_ROOT=/path/to/asc-devkit \
ARCH=dav-2201 \
PRIMARY_CARD=7 \
RETRY_CARDS="0 1 2 3 4 5 6" \
bash scripts/run_presmoke.sh
```

950：

```bash
PROJECT_ROOT=/path/to/asc-devkit \
ARCH=dav-3510 \
PRIMARY_CARD=0 \
bash scripts/run_presmoke.sh
```

950 dry-run：

```bash
ARCH=dav-3510 PRIMARY_CARD=0 bash scripts/run_presmoke.sh --dry-run
```

910B dry-run：

```bash
ARCH=dav-2201 PRIMARY_CARD=7 bash scripts/run_presmoke.sh --dry-run
```

## 参数

| 变量 | 默认值 | 含义 |
|---|---|---|
| `PROJECT_ROOT` | 脚本所在仓库根目录 | 要运行的 asc-devkit checkout。 |
| `ASCEND_HOME_DIR` | `/usr/local/Ascend/ascend-toolkit/latest` | CANN toolkit 根目录，目录下应包含 `set_env.sh`。 |
| `ARCH` | `dav-2201` | 目标架构。910B 使用 `dav-2201`，950 使用 `dav-3510`。 |
| `NPU_CARDS` | `auto` | NPU 模式使用的卡。`auto` 优先检测可见的 `/dev/davinci*`，再回退到 `npu-smi`。如需强制单卡运行，可设置 `NPU_CARDS=7`。 |
| `PRIMARY_CARD` | `0` | 自动检测不到卡，或运行被强制为单轮模式时使用的兜底主卡。 |
| `RETRY_CARDS` | 空 | 超时用例重试使用的卡号，多个卡号用空格分隔。 |
| `JOBS` | `auto` | 并行 build worker 数。NPU 模式使用保守的 CPU 数阶梯；纯 CPU 模式使用 `ceil(CPU(s) / 4)`，最大值为 64。 |
| `NPU_SLOTS` | `1` | NPU 队列并发度。看护运行建议保持为 `1`。 |
| `CPU_RUN_SLOTS` | `auto` | 纯 CPU 模式的 run worker 数。`auto` 使用 `lscpu` 中的主机 `CPU(s)`。NPU 模式忽略该参数。 |
| `MAKE_JOBS` | `auto` | 单个 case 的 `make -j` 上限。`auto` 使用 `CPU(s) / resolved JOBS`；显式 `make -j` 命令会被 presmoke wrapper 限制。 |
| `TIMEOUT` | `120` | NPU run 命令超时时间，单位秒。Build 和 verify 命令不使用该超时。 |
| `CPU_RUN_TIMEOUT` | `300` | CPU run 命令超时时间，单位秒。CPU-only 验证较慢时可调大。 |
| `SCHEDULE` | `fixed` | 用例提交顺序。看护运行建议使用 `fixed`。 |
| `SCHEDULE_FILE` | 内置文件 | 可选的固定顺序覆盖文件。 |
| `OUT_ROOT` | `presmoke_reports/presmoke_<arch>_<timestamp>` | CI 归档使用的输出目录。 |
| `MODES` | `npu` | Presmoke 运行模式。 |
| `MODE` | 未设置 | 仅当 `MODES` 未设置时使用的别名，例如 `MODE=cpu`。 |

CPU 调试模式需要显式开启。使用 `MODES=cpu` 或 `MODE=cpu`；默认看护路径仍为 `npu`。CPU 模式默认不使用 NPU fixed schedule，除非显式提供 `SCHEDULE_FILE`。

CPU 模式的默认并行度根据主机 CPU 数计算，不针对特定机器写死。例如，192 CPU 主机会解析为 `JOBS=48`、`MAKE_JOBS=4`、`CPU_RUN_SLOTS=192`；88 CPU 主机会解析为 `JOBS=22`、`MAKE_JOBS=4`、`CPU_RUN_SLOTS=88`。

## 报告

编排脚本会生成：

- `status.txt`：高层进度和最终输出位置。
- `full_card<N>/meta.txt`：环境、卡号、耗时和 `npu-smi` 快照。
- `full_card<N>/stdout.log` 和 `stderr.log`：wrapper 原始日志。
- `full_card<N>/results/report.json`：机器可读的 presmoke 报告。
- `full_card<N>/results/report.md`：详细 Markdown 报告。
- `full_card<N>/results/logs/<case>__<mode>.log`：每个 case 的完整命令日志。
- `full_card<N>/results/logs/stages/<case>__<mode>__<stage>.log`：build、run、verify 等阶段日志。
- `FINAL_REPORT.md`：包含失败和重试结果的简要汇总。
- `ALL_CASE_TIMINGS.tsv`：每个 case 的耗时、NPU 等待时间、步骤详情和实际执行命令。
- `FAILURES.tsv`：主运行中的失败或跳过结果，以及重试结果。

## 固定顺序

默认固定顺序文件为：

- `scripts/presmoke/schedules/dav-2201_npu.txt`
- `scripts/presmoke/schedules/dav-3510_npu.txt`

这些顺序会保留自定义算子依赖约束：

- `custom_op_static_lib` 在 `custom_op` 之前运行。
- `parallel_ops_package` 在 `custom_op_static_lib` 之后运行。
- `aclnn`、`aclop`、`onnx`、`tensorflow` 和 `tiling_sink` 在 `custom_op` 之后运行。

不在 schedule 文件中的新增 case 会追加到末尾，然后再次强制应用自定义算子依赖顺序。
全量看护入口默认开启 fixed schedule 覆盖检查，因此过期或缺失的顺序项会快速失败。

基于上一轮运行结果重新生成固定顺序：

```bash
PYTHONPATH=scripts PRESMOKE_PROJECT_ROOT=$PWD python3 -m presmoke \
  --arch dav-2201 \
  --modes npu \
  --jobs auto \
  --schedule npu-idle-min \
  --schedule-report presmoke_reports/latest/full_card7/results/report.json \
  --export-schedule scripts/presmoke/schedules/dav-2201_npu.txt \
  --dry-run \
  --report-format json \
  --results /tmp/presmoke_schedule_export
```

## 注意事项

- 不要在同一个 checkout 中同时运行多个编排脚本。`.presmoke_locks` 下的锁会防止误重入。
- 全量看护入口会清理当前 CANN `opp/vendors` 目录，请确认 CI 用户有权限删除并重新安装自定义算子 vendor 包。
- 需要 TensorFlow 2.6.5 的 TensorFlow 用例会在 case manifest 中显式跳过。
- 超时重试只针对 timeout 用例。非 timeout 失败会直接汇总，留待进一步定位。
