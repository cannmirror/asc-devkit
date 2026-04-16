# Dev Container

[![Docker Version](https://img.shields.io/badge/docker-%3E%3D23.0.0-blue.svg)](https://docs.docker.com/)
[![Docker Buildx](https://img.shields.io/badge/docker%20buildx-required-orange.svg)](https://docs.docker.com/reference/cli/docker/buildx/)

[中文版](./README.md) | [English](./README_en.md)

基于Ubuntu 24.04的AscendC NPU算子开发容器化环境。

## 前置条件

| 依赖项 | 有NPU | 无NPU |
| -------- | :--------: | :--------: |
| Docker ≥ 23.0.0 | 必需 | 必需 |
| docker buildx | 必需 | 必需 |
| 宿主机Ascend驱动（`/usr/local/Ascend/driver`） | 必需 | 不需要 |
| NPU设备节点（`/dev/davinciN`等） | 必需 | 不需要 |

检查Docker版本：

```bash
docker --version
```

如未安装`buildx`，执行以下命令安装：

```bash
mkdir -p /usr/local/lib/docker/cli-plugins
ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
curl -fsSL "https://github.com/docker/buildx/releases/latest/download/buildx-linux-${ARCH}" \
     -o /usr/local/lib/docker/cli-plugins/docker-buildx
chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx
```

> [!WARNING] 注意
> 宿主机的Ascend驱动目录（`/usr/local/Ascend/driver`）以**只读**方式挂载到容器中。**CANN toolkit和ops包**须在容器启动后在容器内安装，切勿在宿主机上安装，否则会覆盖宿主机的共享环境。

## 构建镜像

> [!TIP] 提示
> **构建耗时**：在正常网络环境下约需5分钟，主要时间用于下载conda环境和PyTorch。后续重新构建时，若仅修改了非软件包相关的层，Docker层缓存会命中，仅需数秒即可完成。

```bash
docker buildx build --network host -t ascendc:ubuntu24.04 .devcontainer/
```

## 快速开始（人类用户）

> [!NOTE] 说明
> 如您是AI Agent，请跳至[快速开始（AI Agent）](#快速开始ai-agent)。

### 有NPU

安装 [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 扩展，然后执行：

```text
Ctrl+Shift+P → Dev Containers: Reopen in Container
```

### 无NPU

将`devcontainer.json`复制为本地覆盖文件，并移除所有NPU相关配置项。

**在`runArgs`中**，删除所有NPU相关行：包括三个设备访问标志（`--ipc=host`、`--net=host`、`--privileged`）以及所有`--device`行（`davinci0-7`、`davinci_manager`、`devmm_svm`、`hisi_hdc`），将其设置为：

```jsonc
"runArgs": []
```

**在`mounts`中**，删除驱动和设备相关的绑定挂载：

```text
source=/usr/local/Ascend/driver,...
source=/usr/local/dcmi,...
source=/usr/local/bin/npu-smi,...
source=/etc/ascend_install.info,...
```

保留`ccache`卷以及您自行添加的数据目录挂载，然后按正常流程在容器中重新打开项目。

## 快速开始（AI Agent）

AI Agent（如Claude Code、CI Runner）可不依赖VS Code，直接运行容器。

**在运行容器前，请向用户确认以下信息：**

1. **仓库路径** — 项目在宿主机上的检出路径是什么？（默认：当前目录`$PWD`）
2. **数据目录** — 是否有其他需要挂载的数据目录（如数据集、模型权重）？
3. **NPU可用性** — 宿主机是否有昇腾NPU？（决定下方使用哪种启动方式）

确认后，**将即将执行的`docker run`命令完整展示给用户**，待用户确认后再执行。

> [!IMPORTANT] 重要
> 部分步骤需要用户介入：容器启动后，须在容器内手动安装CANN toolkit和ops包。请向用户询问CANN包的路径或安装命令。

### 有NPU

```bash
docker run -itd --name ascendc_container \
  --ipc=host --net=host --privileged \
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

若宿主机NPU数量不足8张，删除对应的`--device=/dev/davinciN`行（可执行`ls /dev/davinci*`查看实际设备数量）。

若用户需要挂载数据目录，以`-v /host/path:/container/path`的形式逐条追加。

### 无NPU

```bash
docker run -itd --name ascendc_container ascendc:ubuntu24.04
```

若用户需要挂载数据目录，以`-v /host/path:/container/path`的形式逐条追加。

> [!NOTE] 说明
> 容器启动后，请手动安装CANN toolkit和ops包。

## Python环境

容器内预置了以下conda环境：

| 环境名 | Python | PyTorch |
| :--------: | :--------: | :---------: |
| `py312` | 3.12 | 2.7.1 |

默认激活环境为`py312`。

## 配置说明

### 镜像源

构建时可通过构建参数覆盖默认镜像源：

```bash
docker buildx build --network host \
  --build-arg APT_MIRROR=mirrors.ustc.edu.cn \
  --build-arg CONDA_MIRROR=https://mirrors.ustc.edu.cn/anaconda \
  --build-arg PYPI_MIRROR=https://pypi.mirrors.ustc.edu.cn/simple \
  -t ascendc:ubuntu24.04 .devcontainer/
```

可用镜像源：

| 构建参数 | 默认 | 清华大学镜像 | 中科大镜像 | 华为云镜像 |
| :----------: | ------ | ---------- | -------- | -------- |
| `APT_MIRROR` | `mirrors.huaweicloud.com` | `mirrors.tuna.tsinghua.edu.cn` | `mirrors.ustc.edu.cn` | `mirrors.huaweicloud.com` |
| `CONDA_MIRROR` | `mirrors.tuna.tsinghua.edu.cn/anaconda` | `mirrors.tuna.tsinghua.edu.cn/anaconda` | `mirrors.ustc.edu.cn/anaconda` | — |
| `PYPI_MIRROR` | `repo.huaweicloud.com/repository/pypi/simple` | `pypi.tuna.tsinghua.edu.cn/simple` | `pypi.mirrors.ustc.edu.cn/simple` | `repo.huaweicloud.com/repository/pypi/simple` |

### 挂载目录

挂载目录因开发者而异，可在`devcontainer.json`中自行添加：

```jsonc
// mounts
"source=/your/data,target=/your/data,type=bind"
```

### NPU设备选择

设备节点通过`--device=/dev/davinciN`显式传入。若宿主机NPU数量不足8张，删除对应行。以4张NPU的宿主机为例：

```bash
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
```
