# Dev Container

Ubuntu 24.04 development environment for AscendC NPU kernel development.

## Prerequisites

| Requirement | With NPU | Without NPU |
|-------------|----------|-------------|
| Docker ≥ 23.0.0 | required | required |
| docker buildx | required | required |
| Ascend driver on host (`/usr/local/Ascend/driver`) | required | not needed |
| NPU device nodes (`/dev/davinciN`, etc.) | required | not needed |

Check your Docker version:
```bash
docker --version
```

Install buildx if missing:
```bash
mkdir -p /usr/local/lib/docker/cli-plugins
ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
curl -fsSL "https://github.com/docker/buildx/releases/latest/download/buildx-linux-${ARCH}" \
     -o /usr/local/lib/docker/cli-plugins/docker-buildx
chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx
```

> The Ascend **driver** directory (`/usr/local/Ascend/driver`) is mounted read-only into the container. The **CANN toolkit and ops packages** should be installed inside the container after startup — not on the host. Installing CANN on the host risks overwriting a shared environment.

## Build the Image

> **Build time:** approximately 5 minutes on a typical connection. The dominant cost is downloading the conda environment and PyTorch. Subsequent rebuilds hit the Docker layer cache and complete in seconds if only non-package layers changed.

```bash
docker buildx build --network host -t ascendc:ubuntu24.04 .devcontainer/
```

## Quick Start (Human Users)

> If you are an AI agent, skip to [Quick Start (AI Agent)](#quick-start-ai-agent).

### With NPU

Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension, then:

```
Ctrl+Shift+P → Dev Containers: Reopen in Container
```

### Without NPU

Copy `devcontainer.json` to a local override and remove all NPU-specific entries:

**In `runArgs`**, remove all `--device` lines (davinci0-7, davinci_manager, devmm_svm, hisi_hdc). If you have no other args, set:
```jsonc
"runArgs": []
```

**In `mounts`**, remove the driver and device-related bind mounts:
```
source=/usr/local/Ascend/driver,...
source=/usr/local/dcmi,...
source=/usr/local/bin/npu-smi,...
source=/etc/ascend_install.info,...
```

Keep the ccache volume and any data directories you've added. Then reopen in container as normal.

## Quick Start (AI Agent)

Agents (e.g. Claude Code, CI runners) can run the container directly without VS Code.

**Before running the container, confirm the following with the user:**

1. **Repository path** — where is the project checked out on the host? (default: current directory `$PWD`)
2. **Data directories** — are there any additional data directories to mount (e.g. datasets, model weights)?
3. **NPU availability** — does the host have Ascend NPUs? (determines which variant to use below)

Then **show the user the exact `docker run` command** you are about to execute and wait for confirmation before proceeding.

> Some steps require user action: CANN toolkit and ops packages must be installed inside the container
> after startup. Ask the user to provide the CANN package paths or installer commands.

### With NPU

```bash
docker run -itd --name ascendc_container \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  ascendc:ubuntu24.04
```

If the user has fewer than 8 NPUs, remove the corresponding `--device=/dev/davinciN` lines (use `ls /dev/davinci*` to check the number).

If the user needs to mount data directories, append each as `-v /host/path:/container/path`.

### Without NPU

```bash
docker run -itd --name ascendc_container ascendc:ubuntu24.04
```

If the user needs to mount data directories, append each as `-v /host/path:/container/path`.

> After starting the container, install CANN toolkit and ops packages manually, or automate via a setup script.

## Python Environments

One conda environment is pre-installed:

| Environment | Python | PyTorch |
|-------------|--------|---------|
| `py312` | 3.12 | 2.7.1 |

Default active environment is `py312`.

## Configuration

### Mirror Sources

To override mirror sources at build time:

```bash
docker buildx build --network host \
  --build-arg APT_MIRROR=mirrors.ustc.edu.cn \
  --build-arg CONDA_MIRROR=https://mirrors.ustc.edu.cn/anaconda \
  --build-arg PYPI_MIRROR=https://pypi.mirrors.ustc.edu.cn/simple \
  -t ascendc:ubuntu24.04 .devcontainer/
```

Available mirrors:

| Build arg | Default | Tsinghua | USTC | Huawei |
|-----------|---------|----------|------|--------|
| `APT_MIRROR` | `mirrors.huaweicloud.com` | `mirrors.tuna.tsinghua.edu.cn` | `mirrors.ustc.edu.cn` | `mirrors.huaweicloud.com` |
| `CONDA_MIRROR` | `mirrors.tuna.tsinghua.edu.cn/anaconda` | `mirrors.tuna.tsinghua.edu.cn/anaconda` | `mirrors.ustc.edu.cn/anaconda` | — |
| `PYPI_MIRROR` | `repo.huaweicloud.com/repository/pypi/simple` | `pypi.tuna.tsinghua.edu.cn/simple` | `pypi.mirrors.ustc.edu.cn/simple` | `repo.huaweicloud.com/repository/pypi/simple` |

### Data Directory

The data directory varies per developer. Add your own in `devcontainer.json`:

```jsonc
// mounts
"source=/your/data,target=/your/data,type=bind"
```

### NPU Device Selection

Device nodes are passed explicitly with `--device=/dev/davinciN`. If the host has fewer than 8 NPUs, remove the corresponding lines. For example, for a 4-NPU host:

```bash
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
```
