# 构建
## 前提条件
1. **安装依赖**

   以下所列仅为本开源仓源码编译用到的依赖，其中python、gcc、cmake的安装方法请参见配套版本的[用户手册](https://hiascend.com/document/redirect/CannCommunityInstDepend)，选择安装场景后，参见“安装CANN > 安装依赖”章节进行相关依赖的安装。

   - python >= 3.7.0

   - gcc >= 7.3.0

   - cmake >= 3.16.0

   - googletest（可选，仅执行UT时依赖，建议版本[release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

     下载[googletest源码](https://github.com/google/googletest.git)后，执行以下命令安装：

     ```bash
     mkdir temp && cd temp                 # 在googletest源码根目录下创建临时目录并进入
     cmake .. -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
     make
     make install                         # root用户安装googletest
     # sudo make install                  # 非root用户安装googletest

2. **安装驱动与固件（运行态依赖）**

   运行算子时必须安装驱动与固件，若仅编译算子，可跳过本操作，安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》。

## 环境准备
本项目支持由源码编译，进行源码编译前，请根据如下步骤完成相关环境准备。

1. **安装社区尝鲜版CANN toolkit包**

    根据实际环境，下载对应`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`包，下载链接为[toolkit x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/8.5.0.alpha001/Ascend-cann-toolkit_8.5.0.alpha001_linux-x86_64.run)、[toolkit aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/8.5.0.alpha001/Ascend-cann-toolkit_8.5.0.alpha001_linux-aarch64.run)。
    
    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --full --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

2. **配置环境变量**

	根据实际场景，选择合适的命令。

    ```bash
    # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
    source /usr/local/Ascend/set_env.sh
    # 指定路径安装
    # source ${install_path}/set_env.sh
    ```

3. **下载源码**

    开发者可通过如下命令下载本仓源码：
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/ascend-c-dev.git
    ``` 


## 编译安装

1. 编译

   本开源仓提供一键式编译安装能力，进入本开源仓代码根目录，执行如下命令：

   ```bash
   bash build.sh
   ```

   编译完成后会在`output`目录下生成CANN-ascend_c-*<cann_version>*-linux.*<arch>*.run软件包。
2. 安装

   在开源仓根目录下执行下列命令，根据设置的环境变量路径，将编译生成的run包安装到CANN包的装包路径，同时会覆盖原CANN包中的Ascend C内容。

   ```bash
   # 切换到run包生成路径下
   cd output
   # 安装run包
   ./CANN-ascend_c-<cann_version>-linux.<arch>.run
   ```

## 本地验证

在开源仓根目录执行下列命令之一，将依次批跑tests目录下的用例，得到结果日志，用于看护编译是否正常。

```bash
bash build.sh -t
```

或

```bash
bash build.sh --test
```