# 快速入门
## 前提条件
1. **安装依赖**

   以下所列仅为本开源仓源码编译用到的依赖，其中python、gcc、cmake的安装方法请参见配套版本的[用户手册](https://hiascend.com/document/redirect/CannCommunityInstDepend)，选择安装场景后，参见“安装CANN > 安装依赖”章节进行相关依赖的安装。

   - python >= 3.7.0

   - gcc >= 7.3.0

   - cmake >= 3.16.0

   - ccache >= 4.8.2（可选）

     用于缓存编译结果，避免重复编译，缩短项目构建时间。建议版本[release-v4.8.2](https://gitcode.com/cann-src-third-party/ccache/releases/4.8.2)，x86_64环境[下载链接](https://github.com/ccache/ccache/releases/download/v4.8.2/ccache-4.8.2-linux-x86_64.tar.xz)，aarch64环境[下载链接](https://gitcode.com/cann-src-third-party/ccache/releases/download/4.8.2/ccache-4.8.2.tar.gz)。

     x86_64环境安装步骤如下：
     
     ```bash
     # 在准备安装的路径下创建buildtools目录，如有则忽略
     # 这里以安装路径/opt为例，对安装命令进行说明
     mkdir /opt/buildtools
     # 切换到安装包下载路径，将ccache解压到安装路径
     tar -xf ccache-4.8.2-linux-x86_64.tar.xz -C /opt/buildtools
     chmod 755 /opt/buildtools/ccache-4.8.2-linux-x86_64/ccache
     mkdir -p /usr/local/ccache/bin
     # 建立软链接
     ln -sf /opt/buildtools/ccache-4.8.2-linux-x86_64/ccache /usr/local/bin/ccache
     ln -sf /opt/buildtools/ccache-4.8.2-linux-x86_64/ccache /usr/local/ccache/bin/ccache
     # 将ccache添加到环境变量PATH
     export PATH=/usr/local/ccache/bin:$PATH
     ```
     
     aarch64环境安装步骤如下：
     - 下载编译ccache的依赖项：[zstd](https://gitcode.com/cann-src-third-party/zstd/releases/download/1.5.6/zstd-1.5.6.tar.gz)和[hiredis](https://gitcode.com/cann-src-third-party/hiredis/releases/download/1.3.0/hiredis-1.3.0.tar.gz)。
    
     - 编译安装
     
        ```bash
        # 在准备安装的路径下创建buildtools目录，如有则忽略
        # 这里以安装路径/opt为例，对安装命令进行说明
        mkdir /opt/buildtools
        # 切换到安装包下载路径，将zstd解压到安装路径
        tar -xf zstd-1.5.6.tar.gz -C /opt/buildtools
        cd /opt/buildtools/zstd-1.5.6
        make -j 24
        make install
        cd -
        # 切换到安装包下载路径，将hiredis解压到安装路径
        tar -xf hiredis-1.3.0.tar.gz -C /opt/buildtools
        cd /opt/buildtools/hiredis-1.3.0
        make -j 24 prefix=/opt/buildtools/hiredis-1.3.0 all
        make prefix=/opt/buildtools/hiredis-1.3.0 install
        cd -
        # 切换到安装包下载路径，将ccache解压到安装路径
        tar -xf ccache-4.8.2.tar.gz -C /opt/buildtools
        cd /opt/buildtools/ccache-4.8.2
        mkdir build
        cd build/
        cmake -DCMAKE_BUILD_TYPE=Release -DZSTD_LIBRARY=/usr/local/lib/libzstd.a -DZSTD_INCLUDE_DIR=/usr/local/include -DHIREDIS_LIBRARY=/usr/local/lib/libhiredis.a -DHIREDIS_INCLUDE_DIR=/usr/local/include ..
        make -j 24
        make install
        mkdir -p /usr/local/ccache/bin
        # 建立软链接
        ln -sf /usr/local/bin/ccache /usr/local/ccache/bin/ccache
        # 将ccache添加到环境变量PATH
        export PATH=/usr/local/ccache/bin:$PATH
        ```

   - lcov >= 1.16（可选，仅执行UT时依赖）
   
     下载[lcov源码](https://gitcode.com/cann-src-third-party/lcov/releases/download/v1.16/lcov-1.16.tar.gz)后，执行以下命令安装：
     ```bash
     tar -xf lcov-1.16.tar.gz
     cd lcov-1.16
     make install                         # root用户安装
     # sudo make install                  # 非root用户安装
     ```

   - pytest >= 5.4.2（可选，仅执行UT时依赖）

     执行以下命令安装：
     ```bash
     pip3 install pytest
     ```
   
   - coverage >= 4.5.4（可选，仅执行UT时依赖）

     执行以下命令安装：
     ```bash
     pip3 install coverage
     ```

   - googletest（可选，仅执行UT时依赖，建议版本[release-1.14.0](https://gitcode.com/cann-src-third-party/googletest/releases/v1.14.0)）

     下载[googletest源码](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz)后，执行以下命令安装：

     ```bash
     tar -xf googletest-1.14.0.tar.gz
     cd googletest-1.14.0
     mkdir temp && cd temp                # 在googletest源码根目录下创建临时目录并进入
     cmake .. -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
     make
     make install                         # root用户安装googletest
     # sudo make install                  # 非root用户安装googletest
     ```

2. **安装驱动与固件（运行态依赖）**

   运行算子时必须安装驱动与固件，若仅编译算子，可跳过本操作，安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》。

## 环境准备<a name="prepare&install"></a>
本项目支持由源码编译，进行源码编译前，请根据如下步骤完成相关环境准备。

1. **安装社区尝鲜版CANN toolkit包**

    根据实际环境，下载对应`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`包，下载链接为[toolkit x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ascendc/Ascend-cann-toolkit_8.5.0.alpha001_linux-x86_64.run)、[toolkit aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ascendc/Ascend-cann-toolkit_8.5.0.alpha001_linux-aarch64.run)。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --full --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径。
    - 缺省--install-path时， 则使用默认路径安装。
    若使用root用户安装，安装完成后相关软件存储在“/usr/local/Ascend/latest”路径下；若使用非root用户安装，安装完成后相关软件存储在“$HOME/Ascend/latest”路径下。

2. **安装社区版CANN legacy包（运行态依赖）**

    运行算子前必须安装本包，若仅编译算子，可跳过本操作。

    根据产品型号和环境架构，下载对应`cann-${soc_name}-ops-legacy_${cann_version}_linux-${arch}.run`包，下载链接如下：

    - Ascend 910B：[legacy x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ascendc/cann-910b-ops-legacy_8.5.0.0.0_linux-x86_64.run)、[legacy aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ascendc/cann-910b-ops-legacy_8.5.0.0.0_linux-aarch64.run)。
    - Ascend 910C：[legacy x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ascendc/cann-910_93-ops-legacy_8.5.0.0.0_linux-x86_64.run)、[legacy aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ascendc/cann-910_93-ops-legacy_8.5.0.0.0_linux-aarch64.run)。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x cann-${soc_name}-ops-legacy_${cann_version}_linux-${arch}.run
    # 安装命令
    ./cann-${soc_name}-ops-legacy_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```
    - \$\{soc\_name\}：表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。
    - \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

3. **配置环境变量**

- 默认路径，root用户安装

    ```bash
    source /usr/local/Ascend/latest/bin/setenv.bash
    ```

- 默认路径，非root用户安装
    ```bash
    source $HOME/Ascend/latest/bin/setenv.bash
    ```

- 指定路径安装
    ```bash
    source ${install_path}/latest/bin/setenv.bash
    ```

4. **下载源码**

    开发者可通过如下命令下载本仓源码：
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/asc-devkit.git
    ``` 


## 编译安装<a name="compile&install"></a>

1. 编译

   本开源仓提供一键式编译安装能力，进入本开源仓代码根目录，执行如下命令：

   ```bash
   bash build.sh --pkg
   ```

   编译完成后会在`build_out`目录下生成cann-asc-devkit_*<cann_version>*_linux-*\<arch\>*.run软件包。
2. 安装

   在开源仓根目录下执行下列命令，根据设置的环境变量路径，将编译生成的run包安装到CANN包的装包路径，同时会覆盖原CANN包中的Ascend C内容。

   ```bash
   # 切换到run包生成路径下
   cd build_out
   # 默认路径安装run包
   ./cann-asc-devkit_<cann_version>_linux-<arch>.run --full
   # 指定路径安装run包
   ./cann-asc-devkit_<cann_version>_linux-<arch>.run --full --install-path=${install_path}
   ```

## UT验证

在开源仓根目录执行下列命令，将按各模块依次批跑tests目录下的用例，得到结果日志，用于看护编译是否正常。

```bash
bash build.sh --adv_test                         # 批跑tests目录下adv_api里的用例
bash build.sh --basic_test_one                   # 批跑tests目录下basic_api part-one里的用例
bash build.sh --basic_test_two                   # 批跑tests目录下basic_api part-two里的用例
bash build.sh --basic_test_three                 # 批跑tests目录下basic_api part-three里的用例
```
